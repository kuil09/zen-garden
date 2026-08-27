// Spray particle system for breaking wave crests.
//
// Particles are spawned along the crest line of each active breaker (driven by
// the CPU-side breaker anchors) and simulated with simple ballistic physics:
// gravity, air drag, and life decay. The render side draws them as soft points
// that depth-test against the ocean surface so they read as spray thrown off the
// crest rather than floating decals.
//
// This shader owns two compute entry points:
//   spawnSpray   - seeds new particles from active breaker crests
//   updateSpray  - integrates alive particles one step
// A third entry point is intentionally omitted; dead particles are simply left
// with life <= 0 and skipped by both the update and the render vertex stage.

const MAX_PARTICLES: u32 = 8192u;
const WAVE_INSTANCES: u32 = 3u;

const GRAVITY: f32 = 9.81;
const AIR_DRAG: f32 = 0.985;
const PARTICLE_LIFETIME_MIN: f32 = 0.6;
const PARTICLE_LIFETIME_MAX: f32 = 1.6;

struct SprayParams {
  elapsed: f32,
  delta: f32,
  spawnRate: f32,
  instanceCount: f32,
  pad: vec4f,
}

// Per-breaker crest description written by the CPU each frame. Layout matches
// the CPU-side packing in writeBreakerParams() but trimmed to what spray needs.
struct BreakerCrest {
  // Segment A (start) and B (end) of the crest line in world XZ.
  startX: f32,
  startY: f32,
  startZ: f32,
  endX: f32,
  endY: f32,
  endZ: f32,
  radius: f32,
  heightGain: f32,
  curlRate: f32,
  curlWaves: f32,
  phaseOffset: f32,
  crestPeak: f32,
  crestWidth: f32,
  thetaSpan: f32,
  taper: f32,
  envelope: f32,
}

struct Particle {
  position: vec4f,  // xyz world space, w = life (seconds remaining)
  velocity: vec4f,  // xyz velocity, w = size
  color: vec4f,     // rgb tint, w = alpha
  spin: vec4f,      // x = isSpray flag, y = current spin, z = spin rate, w = unused
}

@group(0) @binding(0) var<uniform> params: SprayParams;
@group(0) @binding(1) var<storage, read> breakers: array<BreakerCrest>;
@group(0) @binding(2) var<storage, read_write> particles: array<Particle>;

fn hash21(seed: vec2f) -> vec2f {
  let q = fract(seed * vec2f(0.1031, 0.11369));
  let r = q + dot(q, q.yx + 19.19);
  return fract(vec2f((r.x + r.y), (r.x + r.y) * r.y));
}

fn randomDir(base: vec3f, spread: f32, seed: vec2f) -> vec3f {
  let rand = hash21(seed);
  let theta = rand.x * 6.2831853;
  let phi = (rand.y * 0.5 + 0.5) * spread;
  let tangent = normalize(cross(base, vec3f(0.0, 1.0, 0.0)) + vec3f(0.0001, 0.0, 0.0));
  let bitangent = normalize(cross(base, tangent));
  let dir = normalize(base * cos(phi) + (tangent * cos(theta) + bitangent * sin(theta)) * sin(phi));
  return dir;
}

@compute @workgroup_size(64)
fn spawnSpray(@builtin(global_invocation_id) id: vec3u) {
  let idx = id.x;
  if (idx >= WAVE_INSTANCES) {
    return;
  }
  let breaker = breakers[idx];
  if (breaker.envelope < 0.1) {
    return;
  }

  // Crest line geometry in world space.
  let originA = vec3f(breaker.startX, breaker.startY, breaker.startZ);
  let originB = vec3f(breaker.endX, breaker.endY, breaker.endZ);
  let along = normalize(originB - originA);
  let axis = normalize(cross(along, vec3f(0.0, 1.0, 0.0)) + vec3f(0.0001, 0.0, 0.0));
  let span = length(originB - originA);

  // Number of particles to emit from this crest this frame.
  let particlesPerSegment = 24u;
  let numSegments = 16u;
  let totalToSpawn = particlesPerSegment * numSegments;
  let baseIdx = idx * (MAX_PARTICLES / WAVE_INSTANCES);

  let spawnScale = breaker.envelope * params.spawnRate;
  let seedBase = breaker.phaseOffset * 13.7 + params.elapsed * 7.3;

  for (var seg = 0u; seg < numSegments; seg = seg + 1u) {
    let u = f32(seg) / f32(numSegments);
    let crestPos = originA + along * (u * span);
    let crestHeight = breaker.heightGain * 2.5 * breaker.crestPeak;

    for (var p = 0u; p < particlesPerSegment; p = p + 1u) {
      let particleIdx = baseIdx + seg * particlesPerSegment + p;
      if (particleIdx >= MAX_PARTICLES) {
        continue;
      }

      let seed = f32(particleIdx) * 17.3 + seedBase;
      let rand = hash21(vec2f(seed, seed * 3.7));

      // Only spawn a fraction of the slots per frame, scaled by envelope/rate.
      if (rand.x > spawnScale * 0.5) {
        continue;
      }

      let offsetAlong = (rand.x - 0.5) * 0.8;
      let offsetAxis = (rand.y - 0.5) * 0.6;
      let spawnPos = crestPos + along * offsetAlong + axis * offsetAxis + vec3f(0.0, crestHeight + rand.x * 0.3, 0.0);

      let baseVel = vec3f(along.x * 2.0, 3.5 + rand.y * 2.5, along.z * 2.0);
      let spread = randomDir(normalize(baseVel), 0.6, rand) * (1.5 + rand.x * 2.5);
      let velocity = baseVel + spread * 0.7;

      let life = PARTICLE_LIFETIME_MIN + rand.x * (PARTICLE_LIFETIME_MAX - PARTICLE_LIFETIME_MIN);
      let size = 0.08 + rand.y * 0.14;

      let isSpray = rand.y > 0.55;
      let particleColor = select(vec4f(1.0, 1.0, 1.0, 0.95), vec4f(1.0, 1.0, 1.0, 0.65), isSpray);

      particles[particleIdx] = Particle(
        vec4f(spawnPos, life),
        vec4f(velocity, size),
        particleColor,
        vec4f(select(0.0, 1.0, isSpray), rand.x * 6.2831853, (rand.y - 0.5) * 8.0, 0.0)
      );
    }
  }
}

@compute @workgroup_size(64)
fn updateSpray(@builtin(global_invocation_id) id: vec3u) {
  let idx = id.x;
  if (idx >= MAX_PARTICLES) {
    return;
  }

  var p = particles[idx];
  var life = p.position.w;
  if (life <= 0.0) {
    return;
  }

  let dt = min(params.delta, 0.034);

  // Integrate position (reconstruct vec4f: WGSL forbids swizzle assignment).
  let newPos = p.position.xyz + p.velocity.xyz * dt;
  p.position = vec4f(newPos, p.position.w);

  // Integrate velocity with gravity + air drag.
  let newVel = vec3f(p.velocity.x * AIR_DRAG, p.velocity.y - GRAVITY * dt, p.velocity.z * AIR_DRAG);
  p.velocity = vec4f(newVel, p.velocity.w);

  // Spin update.
  p.spin.y = p.spin.y + p.spin.z * dt;

  // Life decay.
  life = life - dt;
  p.position.w = life;

  // Fade alpha near end of life.
  let fade = smoothstep(0.0, 0.3, life);
  p.color.w = p.color.w * fade;

  particles[idx] = p;
}
