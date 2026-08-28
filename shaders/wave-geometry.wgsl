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
  // CrestCurve: p0, p1, p2, p3 (each xy), shape (vec4f)
  curve0: vec4f,
  curve1: vec4f,
  curve2: vec4f,
  curve3: vec4f,
  curve4: vec4f,
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
  curve: CrestCurve,
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
  // Extract CrestCurve from packed fields (curve0..3 = p0..p3, curve4 = shape)
  params.curve.p0 = placed.curve0;
  params.curve.p1 = placed.curve1;
  params.curve.p2 = placed.curve2;
  params.curve.p3 = placed.curve3;
  params.curve.shape = placed.curve4;
  return params;
}

// #6 — Cross-section of the breaker in (forward, up), in units of the curl
// radius. Rebuilt as FIVE explicit segments so the silhouette has intentional
// sharpness instead of one smooth vertical cap:
//   1. submerged skirt        (v: 0      .. SKIRT_END)
//   2. concave face           (v: SKIRT_END .. FACE_END)
//   3. crest bulb (thick)     (v: FACE_END   .. CREST_END)
//   4. forward hook           (v: CREST_END  .. HOOK_END)
//   5. returning tongue       (v: HOOK_END    .. 1.0)
// The hook juts forward (~20-35% of height) and the tongue drops back BELOW the
// crest maximum, opening the barrel / negative-space the original cap lacked.
// NOTE: param renaming (faceConcavity/crestMass/hookReach/...) is deferred until
// the CPU Breaker packing in main.js is updated; the shape still reads the
// existing fields.
const WAVE_FACE_END: f32 = 0.40;
const WAVE_CREST_END: f32 = 0.52;
const WAVE_HOOK_END: f32 = 0.74;


// ============================================================================
// #7 — Asymmetric 3D crest skeleton (additive foundation).
//
// Replaces the straight originA–originB sweep with a 3D cubic Bézier
// centreline + a parallel-transport frame so the hero ridge bows toward the
// camera and the flanks attenuate asymmetrically. The existing waveSample()
// still drives the live sheet; waveSampleCrest() is the 3D replacement that the
// vertex stage will switch to once main.js packs a CrestCurve per instance.
// ============================================================================
struct CrestCurve {
  p0: vec4f,
  p1: vec4f,
  p2: vec4f,
  p3: vec4f,
  // peakU, forwardBow, bank, seed
  shape: vec4f,
}

fn cubicBezier(p0: vec3f, p1: vec3f, p2: vec3f, p3: vec3f, t: f32) -> vec3f {
  let u = 1.0 - t;
  let w0 = u * u * u;
  let w1 = 3.0 * u * u * t;
  let w2 = 3.0 * u * t * t;
  let w3 = t * t * t;
  return p0 * w0 + p1 * w1 + p2 * w2 + p3 * w3;
}

fn cubicBezierTangent(p0: vec3f, p1: vec3f, p2: vec3f, p3: vec3f, t: f32) -> vec3f {
  let u = 1.0 - t;
  return normalize(
    3.0 * u * u * (p1 - p0) +
    6.0 * u * t * (p2 - p1) +
    3.0 * t * t * (p3 - p2)
  );
}

// Centreline of the hero ridge: bows forward (camera side) near peakU and
// trails asymmetrically (short flank on one side, long on the other).
fn crestCentreline(curve: CrestCurve, u: f32) -> vec3f {
  // bias u so the dominant hook sits at curve.shape.x (peakU) and one flank is
  // shorter than the other (asymmetry), without a periodic sine repeat.
  let biased = clamp(u, 0.0, 1.0);
  var c = cubicBezier(curve.p0.xyz, curve.p1.xyz, curve.p2.xyz, curve.p3.xyz, biased);
  // forward bow toward camera (assume +Z is camera-ish forward here)
  c += vec3f(0.0, 0.0, curve.shape.y);
  return c;
}

// Parallel-transport frame: rotate the previous frame by the minimal rotation
// that aligns its tangent with the new tangent, avoiding the 180° twist/flip a
// world-up cross product would produce on steep crests.
struct CrestFrame {
  tangent: vec3f,
  normal: vec3f,
  binormal: vec3f,
}
fn parallelTransportFrame(tangent: vec3f, prevNormal: vec3f) -> CrestFrame {
  let t = normalize(tangent);
  // project previous normal onto plane perpendicular to t
  let n = normalize(prevNormal - t * dot(prevNormal, t));
  let b = normalize(cross(t, n));
  var f: CrestFrame;
  f.tangent = t;
  f.normal = n;
  f.binormal = b;
  return f;
}

fn waveProfile(v: f32, curl: f32, params: WaveParams) -> vec2f {
  let trough = vec2f(WAVE_TROUGH_X, WAVE_TROUGH_Y);
  // curl (0..1) drives how aggressive the hook/tongue are.
  let curlAmt = 0.45 + 0.55 * curl;

  // --- Segment 1: submerged skirt -----------------------------------------
  if (v < WAVE_SKIRT_END) {
    let k = smoothstep(0.0, 1.0, v / WAVE_SKIRT_END);
    return mix(vec2f(2.20, -0.80), trough, k);
  }

  // --- Segment 2: concave (near-vertical, slightly hollow) face -----------
  if (v < WAVE_FACE_END) {
    let a = (v - WAVE_SKIRT_END) / (WAVE_FACE_END - WAVE_SKIRT_END);
    let crestY = WAVE_BARREL_Y + WAVE_CREST_RADIUS;
    let rise = pow(a, 0.70);
    // Clearly concave (hollow) face: bow the mid-face inward so the silhouette
    // reads as a scooped wall, not a straight ramp. Amplified for visibility.
    let lean = -0.30 * sin(a * 3.14159);
    return vec2f(WAVE_TROUGH_X * pow(1.0 - a, 2.2) + lean, mix(WAVE_TROUGH_Y, crestY, rise));
  }

  // --- Segment 3: crest bulb (short, thick mass) --------------------------
  let capBaseY = WAVE_BARREL_Y + WAVE_CREST_RADIUS;
  let crestMaxY = capBaseY + (WAVE_CREST_RADIUS * 0.20) * params.heightGain;
  if (v < WAVE_CREST_END) {
    let a = (v - WAVE_FACE_END) / (WAVE_CREST_END - WAVE_FACE_END);
    // Bulb: rise to the crest maximum with a slight forward bulge.
    let y = mix(capBaseY, crestMaxY, smoothstep(0.0, 1.0, a));
    let x = 0.06 * sin(a * 3.14159) * curlAmt;
    return vec2f(x, y);
  }

  // --- Segment 4: forward hook --------------------------------------------
  if (v < WAVE_HOOK_END) {
    let a = (v - WAVE_CREST_END) / (WAVE_HOOK_END - WAVE_CREST_END);
    // Hook juts forward (~25-35% of height) so the lip clearly curls over.
    // u.debugMode.y scales the hook reach live (test-mode slider; default 0 = unchanged).
    let hookReach = (0.90 + 0.50 * curlAmt) * (params.heightGain + 0.5) * (1.0 + u.debugMode.y);
    let x = hookReach * smoothstep(0.0, 1.0, a);
    let y = crestMaxY + 0.10 * params.heightGain * sin(a * 3.14159);
    return vec2f(x, y);
  }

  // --- Segment 5: returning tongue (drops below crest max -> barrel) ------
  let a = (v - WAVE_HOOK_END) / (1.0 - WAVE_HOOK_END);
  let hookX = (0.90 + 0.50 * curlAmt) * (params.heightGain + 0.5);
  // Tongue curls back inward AND drops below the crest maximum, opening the
  // barrel / negative-space the original vertical cap lacked.
  let tipX = hookX * (1.0 - 0.75 * a);
  let tipY = crestMaxY - (1.60 + 0.60 * curlAmt) * (params.heightGain + 0.4) * smoothstep(0.0, 1.0, a) * (1.0 + u.debugMode.z);
  return vec2f(tipX, tipY);
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
  // #6: foam masks now align with the new segment boundaries (crest bulb /
  // forward hook / returning tongue) instead of the old single vertical cap.
  let lip = smoothstep(WAVE_CREST_END - 0.04, WAVE_CREST_END + 0.08, uv.y);
  let tongue = smoothstep(WAVE_HOOK_END - 0.02, 1.0, uv.y);

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

// #8 — Hierarchical foam-claw ribbon graph (additive foundation).
//
// Replaces the single flat claw strip (fragment discard) with an instanced
// cubic Bézier ribbon graph: primary fingers define the silhouette rhythm,
// secondary branches hang off a parent's t, and droplets emit from tips.
// The live clawVertex() above is unchanged; foamFingerPoint() is the geometry
// the foam vertex stage will switch to once a deterministic finger graph is
// generated on the CPU (see spray.wgsl FoamFinger + #10 deterministic seed).
// ============================================================================
struct FoamFinger {
  // rootU, rootV, parentIndex, generation
  root: vec4f,
  // tangent angle, normal lift, length, base width
  shapeA: vec4f,
  // hook, taper, twist, phase
  shapeB: vec4f,
  // deterministic seed and visibility envelope
  life: vec4f,
}

// Cubic Bézier ribbon centreline for one finger, evaluated at parameter t.
// root tangent comes from the #6 crest/hook tangent; primary fingers lift up
// and forward then curl back toward the barrel in the second half.
fn foamFingerPoint(finger: FoamFinger, profilePos: vec3f, crestTangent: vec3f, t: f32) -> vec3f {
  let up = vec3f(0.0, 1.0, 0.0);
  let side = normalize(cross(crestTangent, up) + vec3f(0.0001, 0.0, 0.0));
  let dir = normalize(crestTangent * cos(finger.shapeA.x) + up * sin(finger.shapeA.x));
  let root = profilePos;
  let shoulder = root + dir * (finger.shapeA.z * 0.35) + up * finger.shapeA.y;
  // hook: bend back toward the barrel in the latter half
  let hook = shoulder + dir * (finger.shapeA.z * 0.45)
              - side * (finger.shapeB.x * finger.shapeA.z);
  let tip = hook + dir * (finger.shapeA.z * 0.20)
            - side * (finger.shapeB.x * finger.shapeA.z * 1.4);
  let u = clamp(t, 0.0, 1.0);
  let w0 = (1.0 - u) * (1.0 - u) * (1.0 - u);
  let w1 = 3.0 * (1.0 - u) * (1.0 - u) * u;
  let w2 = 3.0 * (1.0 - u) * u * u;
  let w3 = u * u * u;
  return root * w0 + shoulder * w1 + hook * w2 + tip * w3;
}

// Ribbon half-width at parameter t (tapered, with a hard screen-space minimum
// applied later in the vertex stage so tips never collapse to a 1px line).
fn foamFingerWidth(finger: FoamFinger, t: f32) -> f32 {
  let taper = mix(1.0, finger.shapeB.y, clamp(t, 0.0, 1.0));
  return max(finger.shapeA.w * taper, 0.0015);
}
