const GRID_SIZE: u32 = 256u;
const NORMALIZATION: f32 = 1.0 / 65536.0;
const GRAVITY: f32 = 9.81;
const REST_DEPTH: f32 = 1.85;

struct ResolveParams {
  time: f32,
  delta: f32,
  foamDecay: f32,
  frame: f32,
}

struct OceanPoint {
  displacementFoam: vec4f,
  normalJacobian: vec4f,
}

@group(0) @binding(0) var<uniform> params: ResolveParams;
@group(0) @binding(1) var<storage, read> spatialSpectrum: array<vec4f>;
@group(0) @binding(2) var<storage, read> previousOcean: array<OceanPoint>;
@group(0) @binding(3) var<storage, read_write> currentOcean: array<OceanPoint>;
@group(0) @binding(4) var<storage, read> dynamicWater: array<vec4f>;

fn wrappedIndex(x: i32, y: i32) -> u32 {
  let size = i32(GRID_SIZE);
  let wrappedX = (x % size + size) % size;
  let wrappedY = (y % size + size) % size;
  return u32(wrappedY) * GRID_SIZE + u32(wrappedX);
}

fn heights(x: i32, y: i32) -> vec2f {
  let value = spatialSpectrum[wrappedIndex(x, y)];
  return vec2f(value.x, value.z) * NORMALIZATION;
}

fn dynamicState(x: i32, y: i32) -> vec4f {
  return dynamicWater[wrappedIndex(x, y)];
}

@compute @workgroup_size(8, 8)
fn resolveOcean(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= GRID_SIZE || id.y >= GRID_SIZE) {
    return;
  }

  let x = i32(id.x);
  let y = i32(id.y);
  let center = heights(x, y);
  let left = heights(x - 1, y);
  let right = heights(x + 1, y);
  let down = heights(x, y - 1);
  let up = heights(x, y + 1);
  let lowerLeft = heights(x - 1, y - 1);
  let upperRight = heights(x + 1, y + 1);
  let dynamicCenter = dynamicState(x, y);
  let dynamicLeft = dynamicState(x - 1, y);
  let dynamicRight = dynamicState(x + 1, y);
  let dynamicDown = dynamicState(x, y - 1);
  let dynamicUp = dynamicState(x, y + 1);
  let dynamicUpperRight = dynamicState(x + 1, y + 1);

  let cellLarge = 84.0 / f32(GRID_SIZE);
  let cellDetail = 15.0 / f32(GRID_SIZE);
  let gradientLarge = vec2f(
    (right.x - left.x) / (2.0 * cellLarge),
    (up.x - down.x) / (2.0 * cellLarge)
  );
  let gradientDetail = vec2f(
    (right.y - left.y) / (2.0 * cellDetail),
    (up.y - down.y) / (2.0 * cellDetail)
  );
  let cellDynamic = 84.0 / f32(GRID_SIZE);
  let gradientDynamic = vec2f(
    (dynamicRight.x - dynamicLeft.x) / (2.0 * cellDynamic),
    (dynamicUp.x - dynamicDown.x) / (2.0 * cellDynamic)
  );
  let gradient = gradientLarge * 0.82 + gradientDetail * 0.62 + gradientDynamic * 1.18;

  let hxx = (left.x - 2.0 * center.x + right.x) / (cellLarge * cellLarge)
    + (left.y - 2.0 * center.y + right.y) / (cellDetail * cellDetail) * 0.42;
  let hzz = (down.x - 2.0 * center.x + up.x) / (cellLarge * cellLarge)
    + (down.y - 2.0 * center.y + up.y) / (cellDetail * cellDetail) * 0.42;
  let hxz = (
    upperRight.x - right.x - up.x + center.x
  ) / (cellLarge * cellLarge) + (
    upperRight.y - right.y - up.y + center.y
  ) / (cellDetail * cellDetail) * 0.42;
  let dynamicHxx = (dynamicLeft.x - 2.0 * dynamicCenter.x + dynamicRight.x) / (cellDynamic * cellDynamic);
  let dynamicHzz = (dynamicDown.x - 2.0 * dynamicCenter.x + dynamicUp.x) / (cellDynamic * cellDynamic);
  let dynamicHxz = (dynamicUpperRight.x - dynamicRight.x - dynamicUp.x + dynamicCenter.x) / (cellDynamic * cellDynamic);

  let choppiness = 1.58;
  let combinedHxx = hxx * 0.72 + dynamicHxx * 1.08;
  let combinedHzz = hzz * 0.72 + dynamicHzz * 1.08;
  let combinedHxz = hxz * 0.72 + dynamicHxz * 1.08;
  let jacobian = (1.0 - choppiness * combinedHxx) * (1.0 - choppiness * combinedHzz)
    - choppiness * choppiness * combinedHxz * combinedHxz;
  let steepness = length(gradient);
  let compressionFoam = 1.0 - smoothstep(-0.34, 0.16, jacobian);
  let crestFoam = smoothstep(1.05, 2.25, steepness);
  let physicalFoam = dynamicCenter.w;
  let dynamicDepth = max(0.24, REST_DEPTH + dynamicCenter.x);
  let dynamicVelocity = dynamicCenter.yz / dynamicDepth;
  let dynamicFroude = length(dynamicVelocity) / sqrt(GRAVITY * dynamicDepth);
  let dynamicBreaking = smoothstep(0.34, 0.74, length(gradientDynamic))
    * smoothstep(0.18, 0.78, dynamicFroude);
  let foamSource = clamp(
    compressionFoam * 0.35 + crestFoam * 0.15 + physicalFoam * 1.0 + dynamicBreaking * 0.55,
    0.0,
    1.0
  );

  let index = id.y * GRID_SIZE + id.x;
  let drift = clamp(dynamicVelocity, vec2f(-2.2), vec2f(2.2));
  let backtrace = vec2i(round(drift * params.delta / cellDynamic));
  let previousIndex = wrappedIndex(x - backtrace.x, y - backtrace.y);
  let previousFoam = previousOcean[previousIndex].displacementFoam.w;
  let decay = exp(-params.delta * params.foamDecay);
  let foam = max(foamSource, previousFoam * decay);

  let dynamicHeight = dynamicCenter.x;
  let boundHarmonic = dynamicHeight * abs(dynamicHeight) * 0.24;
  let height = center.x * 0.82 + center.y * 0.52 + dynamicHeight * 1.58 + boundHarmonic;
  let horizontal = -gradient * choppiness;
  let normal = normalize(vec3f(-gradient.x, 1.0, -gradient.y));
  currentOcean[index].displacementFoam = vec4f(horizontal.x, height, horizontal.y, foam);
  currentOcean[index].normalJacobian = vec4f(normal, jacobian);
}
