// Parametric plunging-breaker geometry.
//
// A height field cannot describe an overturning wave: the lip hangs forward over
// the face, so the surface is multi-valued in (x, z). Instead the wave is a swept
// parametric sheet P(u, v): u runs along the crest line, v walks around the
// cross-section profile — up the back, over the crest, forward through the lip and
// down the plunging tongue. The spiral in v is what produces the overhang.

// Profile constants, in units of the curl radius. Trough sits at y = 0 so the
// wave meets the surrounding sea; the crest tops out near y = 2.9.
const WAVE_BARREL_Y: f32 = 4.50;
const WAVE_CREST_RADIUS: f32 = 3.50;
const WAVE_TROUGH_X: f32 = 0.25;
const WAVE_TROUGH_Y: f32 = 0.0;
const WAVE_SKIRT_END: f32 = 0.08;
const WAVE_CREST_V: f32 = 0.48;
// Where aerated water starts on the profile; the claws drag this line downward.
const WAVE_LIP_V: f32 = 0.78;
// Where the claw strip is rooted on the profile: the outer top of the lip.
const WAVE_CLAW_BASE_V: f32 = 0.55;
const WAVE_CLAW_REACH: f32 = 1.80;

// Placement data, produced by shaders/breakers.wgsl + the CPU anchor manager.
// One slot per wave-sheet instance; the CPU writes all slots every frame.
// originA/originB carry the crest line endpoints (y = 0, on the sea plane);
// radius and heightGain already include the spawn/despawn envelope.
struct Breaker {
  originA: vec4f,
  originB: vec4f,
  // heightGain, curlRate, curlWaves, phaseOffset
  params: vec4f,
  // crestPeak, crestWidth, thetaSpan, taper
  shape: vec4f,
  // throwGain, detailGain, active, seed
  extras: vec4f,
}

struct WaveParams {
  originA: vec3f,
  originB: vec3f,
  radius: f32,
  heightGain: f32,
  crestPeak: f32,
  crestWidth: f32,
  thetaSpan: f32,
  taper: f32,
  throwGain: f32,
  curlRate: f32,
  curlWaves: f32,
  detailGain: f32,
  phaseOffset: f32,
}

@group(0) @binding(2) var<storage, read> breakers: array<Breaker>;

fn waveParams(instance: u32) -> WaveParams {
  let placed = breakers[instance];
  var params: WaveParams;
  params.originA = placed.originA.xyz;
  params.originB = placed.originB.xyz;
  params.radius = placed.originB.w * placed.extras.z;
  params.heightGain = placed.params.x;
  params.crestPeak = placed.shape.x;
  params.crestWidth = placed.shape.y;
  params.thetaSpan = placed.shape.z;
  params.taper = placed.shape.w;
  params.throwGain = placed.extras.x;
  params.curlRate = placed.params.y;
  params.curlWaves = placed.params.z;
  params.detailGain = placed.extras.y;
  params.phaseOffset = placed.params.w;
  return params;
}

// Cross-section of the breaker in (forward, up), in units of the curl radius.
// Three segments: submerged skirt, near-vertical concave face, tight spiral lip.
// The spiral goes PAST vertical and curls BACK inward — deep barrel.
fn waveProfile(v: f32, curl: f32, params: WaveParams) -> vec2f {
  let trough = vec2f(WAVE_TROUGH_X, WAVE_TROUGH_Y);
  if (v < WAVE_SKIRT_END) {
    let k = smoothstep(0.0, 1.0, v / WAVE_SKIRT_END);
    return mix(vec2f(2.20, -0.80), trough, k);
  }
  if (v < WAVE_CREST_V) {
    let a = (v - WAVE_SKIRT_END) / (WAVE_CREST_V - WAVE_SKIRT_END);
    // Steep cliff face leaning slightly forward (~72° from horizontal, Hokusai).
    let crestY = WAVE_BARREL_Y + WAVE_CREST_RADIUS;
    let rise = pow(a, 0.62);
    // Forward lean: face pushes forward as it rises (not perfectly vertical).
    let lean = 0.55 * (a * a);
    return vec2f(
      WAVE_TROUGH_X * pow(1.0 - a, 2.4) + lean, // tighter x convergence + forward lean
      mix(WAVE_TROUGH_Y, crestY, rise),
    );
  }
  // Vertical cliff cap with pronounced forward overhang at the lip.
  // Hokusai's wave: height approx equals width, lip juts forward ~28% of height.
  let t = (v - WAVE_CREST_V) / (1.0 - WAVE_CREST_V);
  let capBaseY = WAVE_BARREL_Y + WAVE_CREST_RADIUS;
  // Modest extra rise keeps the cliff tall but not needle-thin (aspect H/W ~1.2).
  let extraRise = (WAVE_CREST_RADIUS * 0.18) * params.heightGain;
  let y = capBaseY + extraRise * smoothstep(0.0, 1.0, t);
  // Strong forward overhang (cliff lip, non-collapsing): Hokusai ~28% of height.
  let overhang = 0.75 * smoothstep(0.55, 1.0, t) * (0.4 + 0.6 * curl);
  return vec2f(overhang, y);
}

// How far through the break each slice of the crest is. Travelling this phase
// along u staggers the claws the way Hokusai stacks them, and it loops cleanly.
// Added: per-instance variation via seed, secondary harmonics for less homogeneity.
fn waveCurl(u: f32, params: WaveParams, time: f32) -> f32 {
  let phase = u * params.curlWaves - time * params.curlRate + params.phaseOffset;
  let staggered = 0.5 + 0.5 * sin(phase);
  let secondary = 0.5 + 0.5 * sin(phase * 0.41 + 1.7);
  let tertiary = 0.5 + 0.5 * sin(phase * 0.19 + 3.1); // extra variation
  let base = 0.30 + 0.55 * staggered + 0.18 * secondary + 0.10 * tertiary;
  // Subtle per-instance variation using phaseOffset as seed proxy
  let varSeed = fract(params.phaseOffset * 12.9898);
  let variation = (hash11(varSeed + u * 7.3) - 0.5) * 0.12;
  return clamp(base + variation, 0.0, 1.0);
}

// Deliberately asymmetric: the wave rears up over a short stretch of crest and
// then trails away. A symmetric bump reads as a hill, not as a wave about to fall.
fn waveCrestScale(u: f32, params: WaveParams) -> f32 {
  let rise = smoothstep(0.01, params.crestPeak * 0.75, u);
  let hold = params.crestPeak + params.crestWidth * 0.35;
  let decay = exp(-max(0.0, u - hold) / params.crestWidth);
  let shoulder = 0.10 + 0.90 * decay;
  let ends = smoothstep(0.0, 0.08, 1.0 - u);
  let baseScale = params.heightGain * rise * shoulder * ends;
  // Subtle variation along crest using phaseOffset as seed
  let varSeed = fract(params.phaseOffset * 45.6789);
  let variation = 1.0 + (hash11(varSeed + u * 11.3) - 0.5) * 0.18;
  return baseScale * variation;
}

struct WaveSample {
  position: vec3f,
  curl: f32,
  scale: f32,
  crestDistance: f32,
}

fn waveSample(uv: vec2f, params: WaveParams, time: f32) -> WaveSample {
  let along = normalize(params.originB - params.originA);
  let span = length(params.originB - params.originA);
  let axis = vec3f(along.z, 0.0, -along.x);

  let curl = waveCurl(uv.x, params, time);
  let scale = waveCrestScale(uv.x, params) * params.radius;
  let profile = waveProfile(uv.y, curl, params);

  var position = params.originA
    + along * (uv.x * span)
    + axis * (profile.x * scale)
    + vec3f(0.0, profile.y * scale, 0.0);

  // The existing spectral / shallow-water field is still doing the work: it rides
  // on the sheet as surface detail so the wave never looks like a static sculpture.
  let field = sampleOceanWorld(position.xz);
  let detail = field.displacementFoam.y * 0.30 * params.detailGain
    * smoothstep(0.0, 0.30, uv.y);
  position.y += detail;
  position += axis * detail * 0.35;

  var result: WaveSample;
  result.position = position;
  result.curl = curl;
  result.scale = scale;
  result.crestDistance = uv.y;
  return result;
}

@vertex
fn waveVertex(@location(0) uv: vec2f, @builtin(instance_index) instance: u32) -> SurfaceOutput {
  let params = waveParams(instance);
  let time = u.cameraTime.w;
  // Dead instance guard: inactive slots have zero radius. Clip them out entirely.
  if (params.radius <= 0.0 || params.heightGain <= 0.0) {
    var discardOut: SurfaceOutput;
    discardOut.position = vec4f(0.0, 0.0, 0.0, 0.0);
    return discardOut;
  }
  let here = waveSample(uv, params, time);

  let stepU = 1.0 / 320.0;
  let stepV = 1.0 / 200.0;
  let neighbourU = waveSample(vec2f(clamp(uv.x + stepU, 0.0, 1.0), uv.y), params, time);
  let neighbourV = waveSample(vec2f(uv.x, clamp(uv.y + stepV, 0.0, 1.0)), params, time);
  let tangentU = neighbourU.position - here.position;
  let tangentV = neighbourV.position - here.position;
  var normal = normalize(cross(tangentV, tangentU));
  if (length(tangentU) < 1e-6 || length(tangentV) < 1e-6) {
    normal = vec3f(0.0, 1.0, 0.0);
  }

  let field = sampleOceanWorld(here.position.xz);
  // The lip and the tongue are where a real breaker aerates.
  let lip = smoothstep(WAVE_CREST_V - 0.06, WAVE_CREST_V + 0.14, uv.y);
  let tongue = smoothstep(WAVE_CREST_V + 0.10, 1.0, uv.y);

  var output: SurfaceOutput;
  output.position = u.viewProjection * vec4f(here.position, 1.0);
  // Frustum cull: cull vertices far left of view (NDC x < -1.1 = slightly outside left edge)
  if (output.position.x < -1.1 * output.position.w) {
    output.position = vec4f(0.0, 0.0, 0.0, 0.0);
    return output;
  }
  output.worldPosition = here.position;
  output.normal = normal;
  output.fieldCoordinates = here.position.xz;
  output.foam = clamp(here.curl * 0.55 + field.displacementFoam.w * 0.45, 0.0, 1.0);
  output.compression = clamp(here.curl * max(lip, tongue), 0.0, 1.0);
  output.waveHeight = here.position.y;
  output.sheetCoordinates = uv;
  output.sheetWeight = 1.0;
  return output;
}

// Hokusai's foam is not shading on the wave, it is silhouette: fingers of white
// water rising off the lip against open sky. That has to be real geometry, so the
// claws get their own strip rooted on the outer top of the lip and growing along
// the outward radial. The fragment stage cuts the fingers out of it.
@vertex
fn clawVertex(@location(0) uv: vec2f, @builtin(instance_index) instance: u32) -> SurfaceOutput {
  let params = waveParams(instance);
  let time = u.cameraTime.w;
  if (params.radius <= 0.0 || params.heightGain <= 0.0) {
    var discardOut: SurfaceOutput;
    discardOut.position = vec4f(0.0, 0.0, 0.0, 0.0);
    return discardOut;
  }

  let along = normalize(params.originB - params.originA);
  let span = length(params.originB - params.originA);
  let axis = vec3f(along.z, 0.0, -along.x);

  let curl = waveCurl(uv.x, params, time);
  let bodyScale = waveCrestScale(uv.x, params);
  let scale = bodyScale * params.radius;
  // Claws stay large down the trailing crest even as the body tapers away.
  let clawScale = mix(bodyScale, 1.0, 0.55) * params.radius;
  let root = waveProfile(WAVE_CLAW_BASE_V, curl, params);
  // Claws rise UP off the crest against the sky (Hokusai foam fingers), no plunge.
  let outward = normalize(vec2f(0.32, 1.0));

  let reach = WAVE_CLAW_REACH * mix(0.5, 1.0, curl);
  let extent = uv.y * reach;
  // Fingers fan apart as they rise — more variation per finger.
  let fingerIndex = floor(uv.x * 8.0);
  let fan = (hash11(fingerIndex * 7.3 + 3.7 + params.phaseOffset * 13.0) - 0.5) * 1.1 * uv.y;
  // Variable reach per finger
  let reachVar = 0.7 + hash11(fingerIndex * 11.7 + params.phaseOffset * 17.0) * 0.6;
  let thisReach = reach * reachVar;
  let thisExtent = uv.y * thisReach;

  var position = params.originA
    + along * (uv.x * span + fan * scale)
    + axis * (root.x * scale + outward.x * thisExtent * clawScale)
    + vec3f(0.0, root.y * scale + outward.y * thisExtent * clawScale, 0.0);
  // Minimal gravity — claws rise, they do not plunge back down.
  let gravityVar = 0.8 + hash11(fingerIndex * 19.3 + params.phaseOffset * 23.0) * 0.4;
  position.y -= uv.y * uv.y * 0.03 * clawScale * gravityVar;
  // Slight vertical jitter for organic feel
  position.y += (hash11(fingerIndex * 31.0 + uv.y * 17.0) - 0.5) * 0.08 * clawScale;

  var output: SurfaceOutput;
  output.position = u.viewProjection * vec4f(position, 1.0);
  // Frustum cull: cull vertices far left of view
  if (output.position.x < -1.1 * output.position.w) {
    output.position = vec4f(0.0, 0.0, 0.0, 0.0);
    return output;
  }
  output.worldPosition = position;
  output.normal = normalize(vec3f(axis.x * outward.x, outward.y, axis.z * outward.x));
  output.fieldCoordinates = position.xz;
  output.foam = 1.0;
  output.compression = curl;
  output.waveHeight = position.y;
  // sheetCoordinates.y is the fraction of the way up the strip, so the fragment
  // cuts the same silhouette no matter how long this slice's fingers are.
  output.sheetCoordinates = uv;
  output.sheetWeight = 2.0;
  return output;
}

// ============================================================================
// #12 — Curvature-gated meso fracture (additive foundation).
//
// Replaces uniform global noise with 5–12 sparse fracture features (scallop /
// wedge notch / split tongue / broken island) placed in crest-local coordinates
// and activated only where curvature, region, break phase and the crest envelope
// all agree. Positions are fixed in object space (hero seed) so they never
// slide with the camera; phase only changes their depth/severity.
// The live sheet stays unchanged; fractureGate()/FractureFeature feed the meso
// displacement pass that #6/#7 will expose once the macro geometry lands.
// ============================================================================
struct FractureFeature {
  // centreU, centreV, width, depth
  region: vec4f,
  // direction, skew, kind, seed
  shape: vec4f,
  // birthPhase, peakPhase, deathPhase, lodClass
  life: vec4f,
}

// Gate: 1.0 only where the fracture is allowed to bite. Keeps meso roughness on
// the crest/hook/tongue boundaries and off the flat face and submerged skirt.
fn fractureGate(
  profileCurvature: f32,
  regionMask: f32,
  breakPhase: f32,
  crestEnvelope: f32,
  feature: FractureFeature
) -> f32 {
  let curvatureGate = smoothstep(0.15, 0.55, profileCurvature);
  let phaseGate = smoothstep(feature.life.x, feature.life.y, breakPhase)
               * (1.0 - smoothstep(feature.life.z * 0.85, feature.life.z, breakPhase));
  return clamp(curvatureGate * regionMask * phaseGate * crestEnvelope, 0.0, 1.0);
}

// Depth contribution of a feature at a sample point, in profile (u, v) space.
// Returns 0 outside the feature footprint; the meso pass turns this into a real
// notch/tongue split on the actual geometry.
fn fractureDepth(sampleU: f32, sampleV: f32, feature: FractureFeature, gate: f32) -> f32 {
  let du = (sampleU - feature.region.x) / max(feature.region.z, 1e-3);
  let dv = (sampleV - feature.region.y) / max(feature.region.z, 1e-3);
  let r2 = du * du + dv * dv;
  if (r2 > 1.0) { return 0.0; }
  // soft circular footprint, deepest at centre, with a directional skew
  let footprint = (1.0 - r2) * (1.0 + feature.shape.y * du);
  return feature.region.w * clamp(footprint, 0.0, 1.0) * gate;
}
