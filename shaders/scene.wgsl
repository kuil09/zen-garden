const GRID_SIZE: i32 = 256;
const DOMAIN_SIZE: f32 = 84.0;

struct Uniforms {
  viewProjection: mat4x4f,
  cameraTime: vec4f,
  resolutionMotion: vec4f,
  flowFocus: vec4f,
  frameExposure: vec4f,
  sunDirection: vec4f,
  camRight: vec4f,
  camUp: vec4f,
  camForward: vec4f,
}

struct OceanPoint {
  displacementFoam: vec4f,
  normalJacobian: vec4f,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> ocean: array<OceanPoint>;

fn saturate(value: f32) -> f32 {
  return clamp(value, 0.0, 1.0);
}

fn hash21(point: vec2f) -> f32 {
  let p = fract(point * vec2f(123.34, 456.21));
  let shifted = p + dot(p, p + 45.32);
  return fract(shifted.x * shifted.y);
}

fn valueNoise(point: vec2f) -> f32 {
  let cell = floor(point);
  let local = fract(point);
  let blend = local * local * (3.0 - 2.0 * local);
  let a = hash21(cell);
  let b = hash21(cell + vec2f(1.0, 0.0));
  let c = hash21(cell + vec2f(0.0, 1.0));
  let d = hash21(cell + vec2f(1.0, 1.0));
  return mix(mix(a, b, blend.x), mix(c, d, blend.x), blend.y);
}

fn fbm(point: vec2f) -> f32 {
  var p = point;
  var value = 0.0;
  var amplitude = 0.54;
  for (var octave = 0; octave < 5; octave = octave + 1) {
    value += valueNoise(p) * amplitude;
    p = vec2f(p.x * 1.67 - p.y * 1.13, p.x * 1.13 + p.y * 1.67) + 7.13;
    amplitude *= 0.49;
  }
  return value;
}

fn wrapped(value: i32) -> u32 {
  return u32((value % GRID_SIZE + GRID_SIZE) % GRID_SIZE);
}

fn oceanPointAt(x: i32, y: i32) -> OceanPoint {
  return ocean[wrapped(y) * u32(GRID_SIZE) + wrapped(x)];
}

fn sampleOceanWorld(worldXZ: vec2f) -> OceanPoint {
  let coordinates = fract(worldXZ / DOMAIN_SIZE + vec2f(0.5)) * f32(GRID_SIZE);
  let base = vec2i(floor(coordinates));
  let local = fract(coordinates);
  let a = oceanPointAt(base.x, base.y);
  let b = oceanPointAt(base.x + 1, base.y);
  let c = oceanPointAt(base.x, base.y + 1);
  let d = oceanPointAt(base.x + 1, base.y + 1);
  return OceanPoint(
    mix(mix(a.displacementFoam, b.displacementFoam, local.x), mix(c.displacementFoam, d.displacementFoam, local.x), local.y),
    mix(mix(a.normalJacobian, b.normalJacobian, local.x), mix(c.normalJacobian, d.normalJacobian, local.x), local.y)
  );
}

struct SurfaceOutput {
  @builtin(position) position: vec4f,
  @location(0) worldPosition: vec3f,
  @location(1) normal: vec3f,
  @location(2) fieldCoordinates: vec2f,
  @location(3) foam: f32,
  @location(4) compression: f32,
  @location(5) waveHeight: f32,
  @location(6) sheetCoordinates: vec2f,
  @location(7) sheetWeight: f32,
}

@vertex
fn oceanVertex(@location(0) uv: vec2f) -> SurfaceOutput {
  // The grid follows the camera and fans out toward the horizon. Sampling the
  // simulation in world space keeps the water fixed while geometry follows.
  let depthProgress = pow(uv.y, 1.42);
  let distanceForward = mix(0.62, 460.0, depthProgress);
  let halfWidth = mix(12.0, 360.0, pow(uv.y, 0.86));
  let baseXZ = vec2f(
    u.cameraTime.x + (uv.x - 0.5) * halfWidth * 2.0,
    u.cameraTime.z + distanceForward
  );
  let field = sampleOceanWorld(baseXZ);
  let distanceFade = mix(1.0, 0.34, smoothstep(40.0, 170.0, distanceForward));
  var worldPosition = vec3f(baseXZ.x, -0.52, baseXZ.y);
  worldPosition.x += field.displacementFoam.x * distanceFade;
  worldPosition.y += field.displacementFoam.y * distanceFade;
  worldPosition.z += field.displacementFoam.z * distanceFade;
  let crestEvent = smoothstep(0.38, 1.18, field.displacementFoam.y)
    * smoothstep(0.045, 0.44, 1.0 - field.normalJacobian.y);
  let crestDirection = normalize(vec2f(-field.normalJacobian.x, -field.normalJacobian.z) + vec2f(0.001, 0.0));
  worldPosition.x += crestDirection.x * crestEvent * 1.48 * distanceFade;
  worldPosition.y += crestEvent * 0.46 * distanceFade;
  worldPosition.z += crestDirection.y * crestEvent * 1.48 * distanceFade;

  var output: SurfaceOutput;
  output.position = u.viewProjection * vec4f(worldPosition, 1.0);
  output.worldPosition = worldPosition;
  output.normal = normalize(mix(field.normalJacobian.xyz, vec3f(0.0, 1.0, 0.0), 1.0 - distanceFade));
  output.fieldCoordinates = baseXZ;
  output.foam = field.displacementFoam.w * distanceFade;
  output.compression = max(0.0, 1.0 - field.normalJacobian.w) * distanceFade;
  output.waveHeight = field.displacementFoam.y * distanceFade;
  output.sheetCoordinates = uv;
  output.sheetWeight = 0.0;
  return output;
}

fn microNormal(worldPosition: vec3f, normal: vec3f, foam: f32) -> vec3f {
  let time = u.cameraTime.w * u.resolutionMotion.z;
  let phaseWarp = valueNoise(worldPosition.xz * 0.19 + vec2f(time * 0.03, -time * 0.02)) * 2.4;
  let shortA = sin(dot(worldPosition.xz, vec2f(7.4, 5.1)) - time * 3.1 + phaseWarp);
  let shortB = sin(dot(worldPosition.xz, vec2f(-11.2, 8.7)) - time * 4.4 + 1.7 - phaseWarp * 0.6);
  let shortC = sin(dot(worldPosition.xz, vec2f(18.5, 3.9)) - time * 5.7 - 0.8 + phaseWarp * 0.35);
  let distanceFade = 1.0 - smoothstep(28.0, 92.0, distance(worldPosition, u.cameraTime.xyz));
  let perturbation = vec3f(shortA * 0.014 + shortB * 0.008, 0.0, shortB * 0.012 + shortC * 0.006);
  return normalize(normal + perturbation * distanceFade * (1.0 - foam * 0.82));
}

// Woodblock palette. Six inks, in the order a printer would lay them down.
const INK_PAPER: vec3f = vec3f(0.945, 0.918, 0.847);
const INK_MIST: vec3f = vec3f(0.741, 0.792, 0.808);
const INK_PALE: vec3f = vec3f(0.451, 0.596, 0.667);
const INK_FOAM: vec3f = vec3f(0.98, 0.99, 1.0); // bright white foam
const INK_MID: vec3f = vec3f(0.188, 0.376, 0.545);
const INK_PRUSSIAN: vec3f = vec3f(0.063, 0.180, 0.341);
const INK_SUMI: vec3f = vec3f(0.043, 0.075, 0.129);

// Four flat tones, no gradient between them. A woodblock has one plate per tone.
fn waterRamp(level: f32) -> vec3f {
  if (level < 0.26) { return INK_PRUSSIAN; }
  if (level < 0.50) { return INK_MID; }
  if (level < 0.74) { return INK_PALE; }
  return INK_MIST;
}

fn hash11(value: f32) -> f32 {
  return fract(sin(value * 78.233) * 43758.5453);
}

@fragment
fn surfaceFragment(input: SurfaceOutput) -> @location(0) vec4f {
  let viewDirection = normalize(u.cameraTime.xyz - input.worldPosition);
  var normal = normalize(input.normal);
  let backFacing = dot(normal, viewDirection) < 0.0;
  if (backFacing) {
    normal = -normal;
  }
  normal = microNormal(input.worldPosition, normal, input.foam);

  let lightDirection = normalize(u.sunDirection.xyz);
  let nol = dot(normal, lightDirection);
  let sun = saturate(nol * 0.5 + 0.5);
  let sky = saturate(normal.y * 0.5 + 0.5);

  // The spectral field only moves the sea by a metre or so, so map that range
  // across the whole ramp; otherwise every distant plate prints the same tone.
  var level = 0.02
    + smoothstep(-1.10, 1.45, input.waveHeight) * 0.50
    + sun * 0.09
    + sky * 0.07;
  // Stretched along the swell so the open sea reads as ranks of water rather
  // than as mottling.
  level += (fbm(input.fieldCoordinates * vec2f(0.10, 0.62)) - 0.5) * 0.26;
  level = saturate(level);
  var color = waterRamp(level);

  // Hokusai's crests are lit from within: a jade band right under the lip.
  let crest = smoothstep(0.28, 0.80, input.compression);
  color = mix(color, vec3f(0.353, 0.612, 0.596), crest * 0.38);

  // Foam is unprinted paper: a hard edge, never a gradient.
  let breakup = fbm(input.fieldCoordinates * 1.5);
  let fine = fbm(input.fieldCoordinates * 6.5 + vec2f(11.3, -4.1));
  let porosity = breakup * 0.65 + fine * 0.35;
  var foamMask = step(0.4, input.foam * 2.3 - porosity * 0.75 + 0.12);
  var foamEdge = step(0.4, input.foam * 2.3 - porosity * 0.75 - 0.01);
  // Suppress foam too close to camera (boundary waves entering view edge).
  let distToCam = length(u.cameraTime.xyz - input.worldPosition);
  let nearFade = smoothstep(15.0, 35.0, distToCam);
  foamMask *= nearFade;
  foamEdge *= nearFade;
  // Suppress foam on far left of view (camera-relative X).
  let toWorld = input.worldPosition - u.cameraTime.xyz;
  let viewX = dot(toWorld, u.camRight.xyz);
  let leftFade = smoothstep(-35.0, -25.0, viewX);
  foamMask *= leftFade;
  foamEdge *= leftFade;
  color = mix(color, INK_FOAM, foamEdge * 1.0);
  color = mix(color, INK_PAPER, foamMask * 1.2);

  let distance = length(u.cameraTime.xyz - input.worldPosition);
  color = mix(color, INK_PALE, smoothstep(70.0, 220.0, distance) * 0.55);
  return vec4f(color, 1.0);
}


struct BackgroundOutput {
  @builtin(position) position: vec4f,
}

@vertex
fn backgroundVertex(@builtin(vertex_index) index: u32) -> BackgroundOutput {
  var positions = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f(3.0, -1.0),
    vec2f(-1.0, 3.0)
  );
  var output: BackgroundOutput;
  output.position = vec4f(positions[index], 0.9999, 1.0);
  return output;
}

// Ray through this background pixel, built from the camera basis. NDC x is
// right, y is up; camRight.w / camUp.w carry tan(halfFov).
fn backgroundRay(ndc: vec2f) -> vec3f {
  return normalize(u.camForward.xyz
    + u.camRight.xyz * (ndc.x * u.camRight.w)
    + u.camUp.xyz * (ndc.y * u.camUp.w));
}

@fragment
fn backgroundFragment(input: BackgroundOutput) -> @location(0) vec4f {
  let uv = input.position.xy / u.resolutionMotion.xy;
  let ndc = vec2f((uv.x - 0.5) * 2.0, (0.5 - uv.y) * 2.0);
  let direction = backgroundRay(ndc);

  // Banded sky, keyed off the ray elevation so the bands sit at the horizon
  // whatever the camera does. Banding is the point — a print has no
  // continuous gradient.
  let bandHeight = saturate(direction.y * 1.9 + 0.16);
  let band = floor(bandHeight * 5.0) / 5.0;
  var color = mix(INK_PAPER, INK_MIST, smoothstep(0.30, 1.0, band) * 0.40);

  // Plate clouds: flat masses with a hard edge, not volumetric fbm. Anchored
  // to the ray bearing so they hold still as the camera drifts.
  let time = u.cameraTime.w * u.resolutionMotion.z;
  let bearing = atan2(direction.x, direction.z);
  let cloudCoordinates = vec2f(bearing * 1.9, direction.y * 4.6) + vec2f(time * 0.006, 0.0);
  let cloudShape = fbm(cloudCoordinates + vec2f(fbm(cloudCoordinates * 0.5) * 1.4, 0.0));
  let cloudMask = step(0.62, cloudShape) * smoothstep(0.005, 0.10, direction.y);
  color = mix(color, INK_PAPER, cloudMask * 0.72);

  return vec4f(max(color, vec3f(0.0)), 1.0);
}
