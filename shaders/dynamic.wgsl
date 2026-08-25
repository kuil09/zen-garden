const GRID_SIZE: i32 = 256;
const DOMAIN_SIZE: f32 = 84.0;
const CELL_SIZE: f32 = DOMAIN_SIZE / f32(GRID_SIZE);
const GRAVITY: f32 = 9.81;
const REST_DEPTH: f32 = 1.85;

struct DynamicParams {
  time: f32,
  delta: f32,
  frame: u32,
  substep: u32,
  pointer: vec2f,
  pointerEnergy: f32,
  motion: f32,
}

// Conservative state: free-surface elevation, x momentum, z momentum,
// and transported foam concentration.
struct WaterState {
  elevation: f32,
  momentumX: f32,
  momentumZ: f32,
  foam: f32,
}

@group(0) @binding(0) var<uniform> params: DynamicParams;
@group(0) @binding(1) var<storage, read> previousState: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> nextState: array<vec4f>;

fn clampedIndex(x: i32, y: i32) -> u32 {
  return u32(clamp(y, 0, GRID_SIZE - 1) * GRID_SIZE + clamp(x, 0, GRID_SIZE - 1));
}

fn stateAt(x: i32, y: i32) -> WaterState {
  let state = previousState[clampedIndex(x, y)];
  return WaterState(state.x, state.y, state.z, state.w);
}

fn packed(state: WaterState) -> vec4f {
  return vec4f(state.elevation, state.momentumX, state.momentumZ, state.foam);
}

fn depth(state: WaterState) -> f32 {
  return max(0.24, REST_DEPTH + state.elevation);
}

fn velocity(state: WaterState) -> vec2f {
  return vec2f(state.momentumX, state.momentumZ) / depth(state);
}

fn conserved(state: WaterState) -> vec3f {
  return vec3f(depth(state), state.momentumX, state.momentumZ);
}

fn fluxX(state: WaterState) -> vec3f {
  let waterDepth = depth(state);
  let flow = velocity(state);
  return vec3f(
    state.momentumX,
    state.momentumX * flow.x + 0.5 * GRAVITY * waterDepth * waterDepth,
    state.momentumX * flow.y
  );
}

fn fluxZ(state: WaterState) -> vec3f {
  let waterDepth = depth(state);
  let flow = velocity(state);
  return vec3f(
    state.momentumZ,
    state.momentumZ * flow.x,
    state.momentumZ * flow.y + 0.5 * GRAVITY * waterDepth * waterDepth
  );
}

fn hllX(left: WaterState, right: WaterState) -> vec3f {
  let leftVelocity = velocity(left).x;
  let rightVelocity = velocity(right).x;
  let leftCelerity = sqrt(GRAVITY * depth(left));
  let rightCelerity = sqrt(GRAVITY * depth(right));
  let leftSignal = min(leftVelocity - leftCelerity, rightVelocity - rightCelerity);
  let rightSignal = max(leftVelocity + leftCelerity, rightVelocity + rightCelerity);
  if (leftSignal >= 0.0) {
    return fluxX(left);
  }
  if (rightSignal <= 0.0) {
    return fluxX(right);
  }
  return (
    rightSignal * fluxX(left) - leftSignal * fluxX(right)
      + leftSignal * rightSignal * (conserved(right) - conserved(left))
  ) / max(0.0001, rightSignal - leftSignal);
}

fn hllZ(south: WaterState, north: WaterState) -> vec3f {
  let southVelocity = velocity(south).y;
  let northVelocity = velocity(north).y;
  let southCelerity = sqrt(GRAVITY * depth(south));
  let northCelerity = sqrt(GRAVITY * depth(north));
  let southSignal = min(southVelocity - southCelerity, northVelocity - northCelerity);
  let northSignal = max(southVelocity + southCelerity, northVelocity + northCelerity);
  if (southSignal >= 0.0) {
    return fluxZ(south);
  }
  if (northSignal <= 0.0) {
    return fluxZ(north);
  }
  return (
    northSignal * fluxZ(south) - southSignal * fluxZ(north)
      + southSignal * northSignal * (conserved(north) - conserved(south))
  ) / max(0.0001, northSignal - southSignal);
}

fn bilinearFoam(coordinates: vec2f) -> f32 {
  let bounded = clamp(coordinates, vec2f(0.0), vec2f(f32(GRID_SIZE - 1)));
  let base = vec2i(floor(bounded));
  let local = fract(bounded);
  let a = stateAt(base.x, base.y).foam;
  let b = stateAt(base.x + 1, base.y).foam;
  let c = stateAt(base.x, base.y + 1).foam;
  let d = stateAt(base.x + 1, base.y + 1).foam;
  return mix(mix(a, b, local.x), mix(c, d, local.x), local.y);
}

fn periodicDistance(value: f32, period: f32) -> f32 {
  return fract(value / period + 0.5) * period - period * 0.5;
}

fn pressureGradient(distanceToFront: f32, width: f32, direction: vec2f, amplitude: f32) -> vec2f {
  let normalized = distanceToFront / width;
  let pressure = amplitude * exp(-0.5 * normalized * normalized);
  return direction * (-normalized / width) * pressure;
}

@compute @workgroup_size(8, 8)
fn evolveDynamicWater(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= u32(GRID_SIZE) || id.y >= u32(GRID_SIZE)) {
    return;
  }

  let x = i32(id.x);
  let y = i32(id.y);
  let center = stateAt(x, y);
  let west = stateAt(x - 1, y);
  let east = stateAt(x + 1, y);
  let south = stateAt(x, y - 1);
  let north = stateAt(x, y + 1);
  let uv = (vec2f(id.xy) + 0.5) / f32(GRID_SIZE);
  let world = (uv - 0.5) * DOMAIN_SIZE;
  let dt = min(params.delta, 0.0085);
  if (dt <= 0.0 || params.motion <= 0.0) {
    nextState[id.y * u32(GRID_SIZE) + id.x] = packed(center);
    return;
  }

  // The HLL approximate Riemann solver preserves the characteristic wave
  // speeds while resolving colliding bores more sharply than scalar diffusion.
  let fluxWest = hllX(west, center);
  let fluxEast = hllX(center, east);
  let fluxSouth = hllZ(south, center);
  let fluxNorth = hllZ(center, north);
  let updated = conserved(center)
    - dt / CELL_SIZE * ((fluxEast - fluxWest) + (fluxNorth - fluxSouth));

  var elevation = updated.x - REST_DEPTH;
  var momentum = updated.yz;

  // Two moving atmospheric-pressure fronts generate waves in the same
  // conservative field. Their waves therefore collide and exchange momentum.
  let directionA = normalize(vec2f(0.24, -0.971));
  let directionB = normalize(vec2f(-0.79, -0.613));
  let distanceA = periodicDistance(dot(world, directionA) - params.time * 3.75 + 40.0, 72.0);
  let distanceB = periodicDistance(dot(world, directionB) + params.time * 2.85 + 13.0, 53.0);
  let lateralA = 0.72 + 0.28 * sin(dot(world, vec2f(-directionA.y, directionA.x)) * 0.11 + params.time * 0.09);
  let lateralB = 0.74 + 0.26 * sin(dot(world, vec2f(-directionB.y, directionB.x)) * 0.085 - params.time * 0.07 + 1.8);
  let pressureA = pressureGradient(distanceA, 1.48, directionA, 21.5 * lateralA);
  let pressureB = pressureGradient(distanceB, 1.86, directionB, 15.6 * lateralB);
  momentum -= max(0.24, REST_DEPTH + elevation) * (pressureA + pressureB) * dt;

  // Incoming boundary characteristics continuously replenish a wind sea.
  let farBand = smoothstep(0.875, 0.945, uv.y) * (1.0 - smoothstep(0.968, 0.995, uv.y));
  let farPhase = params.time * 2.05 + world.x * 0.20 + sin(world.x * 0.071 + params.time * 0.17) * 0.58;
  let farElevation = (sin(farPhase) + 0.22 * sin(farPhase * 2.0 - 0.7)) * 0.58;
  let farFlow = directionA * farElevation * sqrt(GRAVITY * REST_DEPTH);
  let farBlend = farBand * min(1.0, dt * 7.5);
  elevation = mix(elevation, farElevation, farBlend);
  momentum = mix(momentum, farFlow, farBlend);

  let sideBand = (1.0 - smoothstep(0.015, 0.085, uv.x))
    * smoothstep(0.14, 0.34, uv.y) * (1.0 - smoothstep(0.78, 0.94, uv.y));
  let sidePhase = params.time * 2.48 + world.y * 0.23 + sin(world.y * 0.08 - params.time * 0.13) * 0.42;
  let sideElevation = (sin(sidePhase) + 0.18 * sin(sidePhase * 2.0 + 0.9)) * 0.42;
  let sideFlow = vec2f(1.0, 0.12) * sideElevation * sqrt(GRAVITY * REST_DEPTH);
  let sideBlend = sideBand * min(1.0, dt * 6.2);
  elevation = mix(elevation, sideElevation, sideBlend);
  momentum = mix(momentum, sideFlow, sideBlend);

  let pointerCoordinates = vec2f(params.pointer.x, 1.0 - params.pointer.y) * f32(GRID_SIZE);
  let pointerVector = (vec2f(id.xy) - pointerCoordinates) / 16.0;
  let pointerRadius = length(pointerVector);
  let pointerPressure = exp(-pointerRadius * pointerRadius * 1.8) * params.pointerEnergy;
  if (pointerRadius > 0.001) {
    momentum += normalize(pointerVector) * pointerPressure * dt * 5.0;
  }

  // Bottom drag removes energy after a bore has formed. Numerical dissipation
  // at the HLL shock is the macroscopic analogue of unresolved turbulence.
  momentum *= exp(-dt * 0.115);
  elevation = clamp(elevation, -1.46, 2.35);
  let waterDepth = max(0.24, REST_DEPTH + elevation);
  momentum = clamp(momentum, vec2f(-12.0), vec2f(12.0));
  let flow = momentum / waterDepth;

  let westFlow = velocity(west);
  let eastFlow = velocity(east);
  let southFlow = velocity(south);
  let northFlow = velocity(north);
  let gradient = vec2f(east.elevation - west.elevation, north.elevation - south.elevation) / (2.0 * CELL_SIZE);
  let divergence = (eastFlow.x - westFlow.x + northFlow.y - southFlow.y) / (2.0 * CELL_SIZE);
  let froude = length(flow) / sqrt(GRAVITY * waterDepth);
  let slope = length(gradient);
  let frontCompression = max(0.0, -divergence);

  // A breaker begins when the front exceeds the limiting steepness while the
  // flow converges. Froude number distinguishes an energetic bore from noise.
  let steepnessExcess = smoothstep(0.25, 0.60, slope);
  let bore = smoothstep(0.15, 0.55, froude);
  let convergence = smoothstep(0.08, 0.70, frontCompression);
  let breakerSource = steepnessExcess * bore * convergence;

  // Solve dF/dt + u·grad(F) = S(1-F) - F/tau with mild diffusion.
  let backtrace = vec2f(id.xy) - flow * dt / CELL_SIZE;
  var foam = bilinearFoam(backtrace) * exp(-dt / 3.4);
  foam += (1.0 - foam) * breakerSource * dt * 3.50;
  let neighborFoam = (west.foam + east.foam + south.foam + north.foam) * 0.25;
  foam = mix(foam, neighborFoam, min(0.16, dt * 6.0));

  let edgeDistance = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
  let sourceMask = max(farBand, sideBand);
  let absorption = mix(0.88, 1.0, max(smoothstep(0.0, 0.065, edgeDistance), sourceMask));
  elevation *= absorption;
  momentum *= absorption;

  nextState[id.y * u32(GRID_SIZE) + id.x] = vec4f(
    elevation,
    momentum.x,
    momentum.y,
    clamp(foam, 0.0, 1.0)
  );
}
