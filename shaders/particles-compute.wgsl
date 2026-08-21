const GRID_SIZE: u32 = 256u;
const DOMAIN_SIZE: f32 = 84.0;
const GRAVITY: f32 = 9.81;
const REST_DEPTH: f32 = 1.85;

struct Uniforms {
  viewProjection: mat4x4f,
  cameraTime: vec4f,
  resolutionMotion: vec4f,
  pointerEnergy: vec4f,
  frameExposure: vec4f,
  sunDirection: vec4f,
}

struct Particle {
  positionLife: vec4f,
  velocityLifetime: vec4f,
}

struct OceanPoint {
  displacementFoam: vec4f,
  normalJacobian: vec4f,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(2) var<storage, read> ocean: array<OceanPoint>;
@group(0) @binding(3) var<storage, read> dynamicWater: array<vec4f>;

fn hashU32(value: u32) -> u32 {
  var x = value;
  x = ((x >> 16u) ^ x) * 0x45d9f3bu;
  x = ((x >> 16u) ^ x) * 0x45d9f3bu;
  return (x >> 16u) ^ x;
}

fn random(value: u32) -> f32 {
  return f32(hashU32(value)) / 4294967295.0;
}

fn eventStrength(point: OceanPoint, dynamic: vec4f) -> f32 {
  let compression = max(0.0, 1.0 - point.normalJacobian.w);
  let normal = normalize(point.normalJacobian.xyz);
  let slope = length(normal.xz) / max(normal.y, 0.08);
  let waterDepth = max(0.24, REST_DEPTH + dynamic.x);
  let flow = dynamic.yz / waterDepth;
  let froude = length(flow) / sqrt(GRAVITY * waterDepth);
  let breakingCrest = smoothstep(0.48, 1.42, point.displacementFoam.y)
    * smoothstep(0.20, 0.62, slope)
    * smoothstep(0.12, 0.58, froude);
  return dynamic.w * 0.46
    + smoothstep(0.28, 1.42, compression) * 0.24
    + breakingCrest * 1.18;
}

@compute @workgroup_size(256)
fn updateParticles(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= arrayLength(&particles)) {
    return;
  }

  let frame = u32(u.frameExposure.w);
  let seed = hashU32(id.x * 747796405u + frame * 2891336453u);
  let randomA = random(seed);
  let randomB = random(seed ^ 0x9e3779b9u);
  let randomC = random(seed ^ 0x85ebca6bu);
  var particle = particles[id.x];
  let delta = min(u.frameExposure.x, 0.034) * u.resolutionMotion.z;

  if (particle.positionLife.w <= 0.0 || particle.positionLife.w > particle.velocityLifetime.w) {
    var selectedIndex = seed % (GRID_SIZE * GRID_SIZE);
    var selectedPoint = ocean[selectedIndex];
    var selectedDynamic = dynamicWater[selectedIndex];
    var selectedStrength = eventStrength(selectedPoint, selectedDynamic);
    for (var attempt = 1u; attempt < 9u; attempt = attempt + 1u) {
      let candidateIndex = hashU32(seed + attempt * 0x9e3779b9u) % (GRID_SIZE * GRID_SIZE);
      let candidatePoint = ocean[candidateIndex];
      let candidateDynamic = dynamicWater[candidateIndex];
      let candidateStrength = eventStrength(candidatePoint, candidateDynamic);
      if (candidateStrength > selectedStrength) {
        selectedIndex = candidateIndex;
        selectedPoint = candidatePoint;
        selectedDynamic = candidateDynamic;
        selectedStrength = candidateStrength;
      }
    }

    let grid = vec2u(selectedIndex % GRID_SIZE, selectedIndex / GRID_SIZE);
    let fieldUv = (vec2f(grid) + vec2f(randomB, randomC)) / f32(GRID_SIZE);
    let worldXZ = (fieldUv - 0.5) * DOMAIN_SIZE;
    let normal = normalize(selectedPoint.normalJacobian.xyz);
    let tangent = normalize(cross(normal, vec3f(0.0, 1.0, 0.01)) + vec3f(0.001, 0.0, 0.0));
    let waterDepth = max(0.24, REST_DEPTH + selectedDynamic.x);
    let surfaceFlow = selectedDynamic.yz / waterDepth;
    let birthChance = smoothstep(0.36, 1.02, selectedStrength);
    let accepted = select(0.0, 1.0, randomA < birthChance * 0.38);
    let lifetime = mix(0.24, 1.18, randomC * randomC) * accepted;
    let source = vec3f(worldXZ.x, -0.52 + selectedPoint.displacementFoam.y, worldXZ.y);
    let resolvedFlow = vec3f(surfaceFlow.x, 0.0, surfaceFlow.y);
    let ejectionSpeed = mix(1.8, 7.4, randomB * randomB) * birthChance;
    var velocity = resolvedFlow
      + normal * ejectionSpeed
      + tangent * (randomC - 0.5) * mix(0.35, 1.65, birthChance);
    velocity.y += max(0.0, selectedDynamic.y * normal.x + selectedDynamic.z * normal.z) / waterDepth * 0.28;
    particle.positionLife = vec4f(source + normal * mix(0.015, 0.11, randomA), lifetime);
    particle.velocityLifetime = vec4f(velocity, max(lifetime, 0.05));
  } else {
    let position = particle.positionLife.xyz;
    let velocity = particle.velocityLifetime.xyz;
    let windVelocity = vec3f(1.15, 0.0, -0.34);
    let aerodynamicDrag = (windVelocity - velocity) * 0.24;
    let acceleration = vec3f(0.0, -GRAVITY, 0.0) + aerodynamicDrag;
    particle.velocityLifetime = vec4f(velocity + acceleration * delta, particle.velocityLifetime.w);
    particle.positionLife = vec4f(position + particle.velocityLifetime.xyz * delta, particle.positionLife.w - delta);
  }

  particles[id.x] = particle;
}
