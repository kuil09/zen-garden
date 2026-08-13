// Parametric plunging-breaker geometry.
//
// A height field cannot describe an overturning wave: the lip hangs forward over
// the face, so the surface is multi-valued in (x, z). Instead the wave is a swept
// parametric sheet P(u, v): u runs along the crest line, v walks around the
// cross-section profile — up the back, over the crest, forward through the lip and
// down the plunging tongue. The spiral in v is what produces the overhang.

// Profile constants, in units of the curl radius. Trough sits at y = 0 so the
// wave meets the surrounding sea; the crest tops out near y = 2.9.
const WAVE_BARREL_Y: f32 = 1.45;
const WAVE_CREST_RADIUS: f32 = 1.40;
const WAVE_TROUGH_X: f32 = 0.30;
const WAVE_TROUGH_Y: f32 = 0.0;
const WAVE_SKIRT_END: f32 = 0.10;
const WAVE_CREST_V: f32 = 0.58;
// Where aerated water starts on the profile; the claws drag this line downward.
const WAVE_LIP_V: f32 = 0.84;
// Where the claw strip is rooted on the profile: the outer top of the lip.
const WAVE_CLAW_BASE_V: f32 = 0.60;
const WAVE_CLAW_REACH: f32 = 0.92;

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
}

fn waveParams(instance: u32) -> WaveParams {
  // The great wave: the crest line runs from near-left to far-right so the tall
  // end reads against the top-left of the plate and the crest recedes rightward.
  var params: WaveParams;
  params.originA = vec3f(-24.0, 0.0, 4.0);
  params.originB = vec3f(3.0, 0.0, 31.0);
  params.radius = 7.60;
  params.heightGain = 1.0;
  params.crestPeak = 0.30;
  params.crestWidth = 0.55;
  params.thetaSpan = 4.90;
  params.taper = 0.40;
  params.throwGain = 1.0;
  params.curlRate = 0.42;
  params.curlWaves = 3.4;
  params.detailGain = 1.0;

  if (instance == 1u) {
    // Foreground counter-wave, lower right, smaller and less far through its break.
    params.originA = vec3f(-38.0, 0.0, -8.0);
    params.originB = vec3f(10.0, 0.0, 6.0);
    params.radius = 1.75;
    params.heightGain = 0.42;
    params.crestPeak = 0.55;
    params.crestWidth = 0.52;
    params.thetaSpan = 3.45;
    params.throwGain = 0.75;
    params.curlRate = 0.61;
    params.curlWaves = 8.2;
    params.detailGain = 1.25;
  } else if (instance == 2u) {
    // Mid-distance swell that fills the gap between the two masses.
    params.originA = vec3f(-6.0, 0.0, 26.0);
    params.originB = vec3f(48.0, 0.0, 50.0);
    params.radius = 2.60;
    params.heightGain = 0.74;
    params.crestPeak = 0.40;
    params.crestWidth = 0.58;
    params.thetaSpan = 2.95;
    params.throwGain = 0.55;
    params.curlRate = 0.33;
    params.curlWaves = 6.9;
    params.detailGain = 0.7;
  }
  return params;
}

// Cross-section of the breaker in (forward, up), in units of the curl radius.
// Three segments: a submerged skirt, the steep concave face rising out of the
// trough, and a shrinking spiral that carries the lip past vertical and throws
// the tongue forward and down. The spiral is what makes the surface multi-valued
// in (x, z) — the thing a height field cannot do.
fn waveProfile(v: f32, curl: f32, params: WaveParams) -> vec2f {
  let trough = vec2f(WAVE_TROUGH_X, WAVE_TROUGH_Y);
  if (v < WAVE_SKIRT_END) {
    let k = smoothstep(0.0, 1.0, v / WAVE_SKIRT_END);
    return mix(vec2f(2.40, -0.70), trough, k);
  }
  if (v < WAVE_CREST_V) {
    let a = (v - WAVE_SKIRT_END) / (WAVE_CREST_V - WAVE_SKIRT_END);
    let crestY = WAVE_BARREL_Y + WAVE_CREST_RADIUS;
    let rise = pow(a, 0.86);
    return vec2f(
      WAVE_TROUGH_X * pow(1.0 - a, 1.55),
      mix(WAVE_TROUGH_Y, crestY, rise),
    );
  }
  let t = (v - WAVE_CREST_V) / (1.0 - WAVE_CREST_V);
  let theta = params.thetaSpan * mix(0.42, 1.0, curl) * t;
  let radius = WAVE_CREST_RADIUS * (1.0 - params.taper * pow(t, 1.10));
  let throwOut = vec2f(1.85, -1.95) * params.throwGain * curl
    * smoothstep(0.40, 1.0, t);
  return vec2f(0.0, WAVE_BARREL_Y) + radius * vec2f(sin(theta), cos(theta)) + throwOut;
}

// How far through the break each slice of the crest is. Travelling this phase
// along u staggers the claws the way Hokusai stacks them, and it loops cleanly.
fn waveCurl(u: f32, params: WaveParams, time: f32) -> f32 {
  let phase = u * params.curlWaves - time * params.curlRate;
  let staggered = 0.5 + 0.5 * sin(phase);
  let secondary = 0.5 + 0.5 * sin(phase * 0.41 + 1.7);
  return clamp(0.34 + 0.52 * staggered + 0.20 * secondary, 0.0, 1.0);
}

// Deliberately asymmetric: the wave rears up over a short stretch of crest and
// then trails away. A symmetric bump reads as a hill, not as a wave about to fall.
fn waveCrestScale(u: f32, params: WaveParams) -> f32 {
  let rise = smoothstep(0.02, params.crestPeak * 0.85, u);
  let hold = params.crestPeak + params.crestWidth * 0.45;
  let decay = exp(-max(0.0, u - hold) / params.crestWidth);
  let shoulder = 0.16 + 0.84 * decay;
  let ends = smoothstep(0.0, 0.10, 1.0 - u);
  return params.heightGain * rise * shoulder * ends;
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

  let along = normalize(params.originB - params.originA);
  let span = length(params.originB - params.originA);
  let axis = vec3f(along.z, 0.0, -along.x);

  let curl = waveCurl(uv.x, params, time);
  let bodyScale = waveCrestScale(uv.x, params);
  let scale = bodyScale * params.radius;
  // Claws stay large down the trailing crest even as the body tapers away.
  let clawScale = mix(bodyScale, 1.0, 0.55) * params.radius;
  let root = waveProfile(WAVE_CLAW_BASE_V, curl, params);
  // Up and forward, following the direction the lip is already throwing water.
  let outward = normalize(root - vec2f(0.0, WAVE_BARREL_Y) + vec2f(0.72, 0.30));

  let reach = WAVE_CLAW_REACH * mix(0.45, 1.0, curl);
  let extent = uv.y * reach;
  // Fingers fan apart slightly as they rise, the way thrown water separates.
  let fan = (hash11(floor(uv.x * 5.2) + 3.7) - 0.5) * 0.55 * uv.y;

  var position = params.originA
    + along * (uv.x * span + fan * scale)
    + axis * (root.x * scale + outward.x * extent * clawScale)
    + vec3f(0.0, root.y * scale + outward.y * extent * clawScale, 0.0);
  // Gravity bends the tips back over.
  position.y -= uv.y * uv.y * 0.20 * clawScale;

  var output: SurfaceOutput;
  output.position = u.viewProjection * vec4f(position, 1.0);
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
