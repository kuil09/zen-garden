const PI: f32 = 3.141592653589793;

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

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;

struct ParticleOutput {
  @builtin(position) position: vec4f,
  @location(0) local: vec2f,
  @location(1) color: vec3f,
  @location(2) opacity: f32,
  @location(3) kind: f32,
  @location(4) variation: f32,
}

fn saturate(value: f32) -> f32 {
  return clamp(value, 0.0, 1.0);
}

fn hashU32(value: u32) -> u32 {
  var x = value;
  x = ((x >> 16u) ^ x) * 0x45d9f3bu;
  x = ((x >> 16u) ^ x) * 0x45d9f3bu;
  return (x >> 16u) ^ x;
}

fn random(value: u32) -> f32 {
  return f32(hashU32(value)) / 4294967295.0;
}

fn rotate(point: vec2f, angle: f32) -> vec2f {
  let cosine = cos(angle);
  let sine = sin(angle);
  return vec2f(point.x * cosine - point.y * sine, point.x * sine + point.y * cosine);
}

@vertex
fn particleVertex(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> ParticleOutput {
  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
  );
  let particle = particles[instanceIndex];
  let local = corners[vertexIndex];
  let seed = hashU32(instanceIndex * 277803737u + 1013904223u);
  let variation = random(seed);
  let variationB = random(seed ^ 0x9e3779b9u);
  let life = saturate(particle.positionLife.w / max(0.001, particle.velocityLifetime.w));
  let birth = smoothstep(0.0, 0.13, 1.0 - life);
  let death = smoothstep(0.0, 0.20, life);
  let speed = length(particle.velocityLifetime.xyz);

  var kind = 0.0;
  var size = mix(0.007, 0.034, variationB * variationB);
  var aspect = mix(1.0, 1.38, saturate(speed * 0.10));
  if (variation > 0.88) {
    kind = 1.0;
    size = mix(0.014, 0.050, variationB);
    aspect = mix(1.8, 3.6, saturate(speed * 0.16));
  }

  let motionAngle = atan2(particle.velocityLifetime.y, particle.velocityLifetime.x + 0.001);
  let angle = mix(-0.22, 0.22, variationB) + motionAngle * select(0.16, 0.82, kind > 0.5);
  let rotated = rotate(vec2f(local.x, local.y * aspect), angle);
  let cameraRight = vec3f(1.0, 0.0, 0.0);
  let cameraUp = vec3f(0.0, 1.0, 0.022);
  let billboard = cameraRight * rotated.x * size + cameraUp * rotated.y * size;
  let ivory = vec3f(0.90, 0.98, 1.04);
  let cobaltShadow = vec3f(0.25, 0.62, 0.86);
  let color = mix(cobaltShadow, ivory, 0.68 + variationB * 0.28);

  var output: ParticleOutput;
  output.position = u.viewProjection * vec4f(particle.positionLife.xyz + billboard, 1.0);
  output.local = local;
  output.color = color;
  output.opacity = birth * death * mix(0.18, 0.58, variationB);
  output.kind = kind;
  output.variation = variationB;
  return output;
}

fn dropletShape(point: vec2f) -> f32 {
  let shifted = point + vec2f(0.0, 0.16);
  let body = exp(-dot(shifted * vec2f(1.2, 0.78), shifted * vec2f(1.2, 0.78)) * 3.8);
  let taper = smoothstep(-1.0, 0.45, point.y) * (1.0 - smoothstep(0.56, 1.0, point.y + abs(point.x) * 0.42));
  return body * mix(0.62, 1.0, taper);
}

fn streakShape(point: vec2f) -> f32 {
  let core = exp(-dot(point * vec2f(1.9, 0.58), point * vec2f(1.9, 0.58)) * 4.2);
  let split = 0.72 + 0.28 * sin((point.y + point.x * 0.35) * 7.0);
  return core * split;
}

@fragment
fn particleFragment(input: ParticleOutput) -> @location(0) vec4f {
  var shape = dropletShape(input.local);
  if (input.kind > 0.5) {
    shape = streakShape(input.local);
  }
  if (shape < 0.025) {
    discard;
  }
  let alpha = shape * input.opacity;
  let brightCore = pow(shape, 2.2);
  return vec4f(input.color * (0.62 + brightCore * 0.88), alpha);
}
