// Breaker detection.
//
// The swept wave sheets used to be placed by hand: three hardcoded origin
// blocks that had no connection to the simulation running underneath them.
// This pass closes that loop. Every cell of the shallow-water field is scored
// for how much it is currently breaking — the same steepness / Froude /
// convergence triad the foam source uses — and the scores are reduced onto a
// coarse 8x8 block lattice carrying the position, spread, principal axis and
// mean momentum of each breaking region. The CPU reads the 4 KB summary back,
// merges adjacent blocks into breaker candidates, and feeds the result to the
// wave sheets as placement data. The wave now stands where the water says a
// wave should stand.
//
// One deliberate non-physical term: the score is multiplied by a smooth
// composition prior (an ellipse around a focus point). That is art direction —
// Hokusai's breaker sits mid-frame, not anywhere the numerics happen to boil —
// and it is kept here, explicit and separate, rather than disguised as physics.

const GRID_SIZE: u32 = 256u;
const DOMAIN_SIZE: f32 = 84.0;
const CELL_SIZE: f32 = DOMAIN_SIZE / f32(GRID_SIZE);
const GRAVITY: f32 = 9.81;
const REST_DEPTH: f32 = 1.85;

const BLOCKS: u32 = 8u;
const BLOCK_CELLS: u32 = GRID_SIZE / BLOCKS;
const THREADS: u32 = 8u;
const CELLS_PER_THREAD: u32 = BLOCK_CELLS / THREADS;
const SUMMARY_FLOATS: u32 = 16u;

struct BreakerConfig {
  // Composition prior: score inside this ellipse survives, outside fades.
  focus: vec2f,
  radii: vec2f,
  // Suppress breakers that would stand under the camera.
  cameraPos: vec2f,
  minDistance: f32,
  // Scores below the floor carry no placement weight.
  scoreFloor: f32,
  time: f32,
  pad: vec2f,
}

struct OceanPoint {
  displacementFoam: vec4f,
  normalJacobian: vec4f,
}

@group(0) @binding(0) var<uniform> config: BreakerConfig;
@group(0) @binding(1) var<storage, read> ocean: array<OceanPoint>;
@group(0) @binding(2) var<storage, read> dynamic: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> scores: array<f32>;
@group(0) @binding(4) var<storage, read_write> summary: array<f32>;

fn cellIndex(x: i32, y: i32) -> u32 {
  return u32(clamp(y, 0, i32(GRID_SIZE) - 1)) * GRID_SIZE + u32(clamp(x, 0, i32(GRID_SIZE) - 1));
}

fn cellWorld(x: u32, y: u32) -> vec2f {
  return (vec2f(f32(x), f32(y)) + 0.5) / f32(GRID_SIZE) * DOMAIN_SIZE - vec2f(DOMAIN_SIZE * 0.5);
}

fn depthOf(state: vec4f) -> f32 {
  return max(0.24, REST_DEPTH + state.x);
}

@compute @workgroup_size(8, 8)
fn scoreBreakers(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= GRID_SIZE || id.y >= GRID_SIZE) {
    return;
  }
  let x = i32(id.x);
  let y = i32(id.y);
  let center = dynamic[cellIndex(x, y)];
  let west = dynamic[cellIndex(x - 1, y)];
  let east = dynamic[cellIndex(x + 1, y)];
  let south = dynamic[cellIndex(x, y - 1)];
  let north = dynamic[cellIndex(x, y + 1)];

  let gradient = vec2f(
    (east.x - west.x) / (2.0 * CELL_SIZE),
    (north.x - south.x) / (2.0 * CELL_SIZE)
  );
  let flowCenter = center.yz / depthOf(center);
  let flowWest = west.yz / depthOf(west);
  let flowEast = east.yz / depthOf(east);
  let flowSouth = south.yz / depthOf(south);
  let flowNorth = north.yz / depthOf(north);
  let divergence = (flowEast.x - flowWest.x + flowNorth.y - flowSouth.y) / (2.0 * CELL_SIZE);

  // A breaker begins when the front exceeds the limiting steepness while the
  // flow converges; the Froude number separates an energetic bore from chop.
  let steepnessExcess = smoothstep(0.15, 0.45, length(gradient));
  let bore = smoothstep(0.10, 0.40, length(flowCenter) / sqrt(GRAVITY * depthOf(center)));
  let convergence = smoothstep(0.04, 0.50, max(0.0, -divergence));

  // Choppiness in the resolved spectral band sharpens the same event.
  let jacobian = ocean[cellIndex(x, y)].normalJacobian.w;
  let compression = saturate((1.0 - jacobian) * 1.5);

  var score = steepnessExcess * bore * convergence * (0.55 + 0.45 * compression);

  // Bootstrap: existing foam is evidence of recent breaking even if the
  // instantaneous slope has relaxed.
  let foamSignal = ocean[cellIndex(x, y)].displacementFoam.w;
  score = max(score, foamSignal * 0.35 * compression);

  // Art direction, kept explicit: the print wants its breaker near the focus.
  let world = cellWorld(id.x, id.y);
  let ellipse = length((world - config.focus) / config.radii);
  let prior = mix(0.30, 1.0, smoothstep(1.15, 0.30, ellipse));
  let proximity = smoothstep(config.minDistance * 0.5, config.minDistance * 1.5, distance(world, config.cameraPos));
  score *= prior * proximity;

  scores[id.y * GRID_SIZE + id.x] = score;
}

// Per-block partials, one slot per thread, collapsed by a tree reduction.
var<workgroup> rMax: array<f32, 64>;
var<workgroup> rSum: array<f32, 64>;
var<workgroup> rMeanX: array<f32, 64>;
var<workgroup> rMeanZ: array<f32, 64>;
var<workgroup> rCxx: array<f32, 64>;
var<workgroup> rCzz: array<f32, 64>;
var<workgroup> rCxz: array<f32, 64>;
var<workgroup> rMomX: array<f32, 64>;
var<workgroup> rMomZ: array<f32, 64>;

@compute @workgroup_size(8, 8)
fn reduceBreakers(@builtin(global_invocation_id) blockId: vec3u,
                  @builtin(local_invocation_id) localId: vec3u) {
  let block = blockId.y * BLOCKS + blockId.x;
  let tid = localId.y * THREADS + localId.x;
  let baseCell = vec2u(blockId.x, blockId.y) * vec2u(BLOCK_CELLS);
  let threadCell = baseCell + vec2u(localId.x, localId.y) * vec2u(CELLS_PER_THREAD);
  let blockCenter = (vec2f(f32(blockId.x), f32(blockId.y)) + 0.5) * f32(BLOCK_CELLS)
    / f32(GRID_SIZE) * DOMAIN_SIZE - vec2f(DOMAIN_SIZE * 0.5);

  var localMax = 0.0;
  var sum = 0.0;
  var sumX = 0.0;
  var sumZ = 0.0;
  var sumXX = 0.0;
  var sumZZ = 0.0;
  var sumXZ = 0.0;
  var sumMomX = 0.0;
  var sumMomZ = 0.0;

  for (var dy = 0u; dy < CELLS_PER_THREAD; dy = dy + 1u) {
    for (var dx = 0u; dx < CELLS_PER_THREAD; dx = dx + 1u) {
      let cell = threadCell + vec2u(dx, dy);
      let index = cell.y * GRID_SIZE + cell.x;
      let score = scores[index];
      localMax = max(localMax, score);
      if (score > config.scoreFloor) {
        let relative = cellWorld(cell.x, cell.y) - blockCenter;
        let state = dynamic[index];
        sum += score;
        sumX += score * relative.x;
        sumZ += score * relative.y;
        sumXX += score * relative.x * relative.x;
        sumZZ += score * relative.y * relative.y;
        sumXZ += score * relative.x * relative.y;
        sumMomX += score * state.y;
        sumMomZ += score * state.z;
      }
    }
  }

  rMax[tid] = localMax;
  rSum[tid] = sum;
  rMeanX[tid] = sumX;
  rMeanZ[tid] = sumZ;
  rCxx[tid] = sumXX;
  rCzz[tid] = sumZZ;
  rCxz[tid] = sumXZ;
  rMomX[tid] = sumMomX;
  rMomZ[tid] = sumMomZ;
  workgroupBarrier();

  for (var stride = 32u; stride >= 1u; stride = stride >> 1u) {
    if (tid < stride) {
      rMax[tid] = max(rMax[tid], rMax[tid + stride]);
      rSum[tid] = rSum[tid] + rSum[tid + stride];
      rMeanX[tid] = rMeanX[tid] + rMeanX[tid + stride];
      rMeanZ[tid] = rMeanZ[tid] + rMeanZ[tid + stride];
      rCxx[tid] = rCxx[tid] + rCxx[tid + stride];
      rCzz[tid] = rCzz[tid] + rCzz[tid + stride];
      rCxz[tid] = rCxz[tid] + rCxz[tid + stride];
      rMomX[tid] = rMomX[tid] + rMomX[tid + stride];
      rMomZ[tid] = rMomZ[tid] + rMomZ[tid + stride];
    }
    workgroupBarrier();
  }

  if (tid == 0u) {
    let out = block * SUMMARY_FLOATS;
    summary[out + 0u] = rMax[0u];
    if (rSum[0u] > 0.0) {
      let total = rSum[0u];
      let meanRelX = rMeanX[0u] / total;
      let meanRelZ = rMeanZ[0u] / total;
      summary[out + 1u] = total;
      summary[out + 2u] = blockCenter.x + meanRelX;
      summary[out + 3u] = blockCenter.y + meanRelZ;
      summary[out + 4u] = rCxx[0u] / total - meanRelX * meanRelX;
      summary[out + 5u] = rCzz[0u] / total - meanRelZ * meanRelZ;
      summary[out + 6u] = rCxz[0u] / total - meanRelX * meanRelZ;
      summary[out + 7u] = rMomX[0u];
      summary[out + 8u] = rMomZ[0u];
    }
  }
}
