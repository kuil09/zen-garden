const GRID_SIZE: i32 = 256;
const DOMAIN_SIZE: f32 = 84.0;
const PI: f32 = 3.141592653589793;

struct Uniforms {
  viewProjection: mat4x4f,
  cameraTime: vec4f,
  resolutionMotion: vec4f,
  pointerEnergy: vec4f,
  frameExposure: vec4f,
  sunDirection: vec4f,
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
}

@vertex
fn oceanVertex(@location(0) uv: vec2f) -> SurfaceOutput {
  // The grid follows the camera and fans out toward the horizon. Sampling the
  // simulation in world space keeps the water fixed while geometry follows.
  let depthProgress = pow(uv.y, 1.42);
  let distanceForward = mix(0.62, 118.0, depthProgress);
  let halfWidth = mix(12.0, 92.0, pow(uv.y, 0.86));
  let baseXZ = vec2f(
    u.cameraTime.x + (uv.x - 0.5) * halfWidth * 2.0,
    u.cameraTime.z + distanceForward
  );
  let field = sampleOceanWorld(baseXZ);
  let distanceFade = mix(1.0, 0.48, smoothstep(40.0, 118.0, distanceForward));
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
  return output;
}

fn skyRadiance(directionInput: vec3f) -> vec3f {
  let direction = normalize(directionInput);
  let elevation = saturate(direction.y * 0.5 + 0.5);
  let horizon = pow(saturate(1.0 - abs(direction.y)), 7.0);
  let zenith = pow(saturate(direction.y), 0.48);
  var color = mix(vec3f(0.002, 0.014, 0.074), vec3f(0.022, 0.145, 0.43), elevation);
  color += vec3f(0.008, 0.075, 0.25) * zenith;
  color += vec3f(0.072, 0.35, 0.59) * horizon * 0.76;

  let lightDirection = normalize(u.sunDirection.xyz);
  let sunCosine = max(0.0, dot(direction, lightDirection));
  let sun = pow(sunCosine, 1050.0);
  let innerHalo = pow(sunCosine, 76.0);
  let outerHalo = pow(sunCosine, 11.0);
  color += vec3f(4.2, 2.35, 0.96) * sun;
  color += vec3f(0.66, 0.25, 0.055) * innerHalo * 0.48;
  color += vec3f(0.18, 0.08, 0.026) * outerHalo * 0.24;
  return color;
}

fn dielectricFresnel(cosine: f32) -> f32 {
  let eta = 1.0 / 1.333;
  let transmittedSineSquared = eta * eta * max(0.0, 1.0 - cosine * cosine);
  if (transmittedSineSquared >= 1.0) {
    return 1.0;
  }
  let transmittedCosine = sqrt(max(0.0, 1.0 - transmittedSineSquared));
  let parallel = (cosine - 1.333 * transmittedCosine) / max(cosine + 1.333 * transmittedCosine, 0.0001);
  let perpendicular = (transmittedCosine - 1.333 * cosine) / max(transmittedCosine + 1.333 * cosine, 0.0001);
  return 0.5 * (parallel * parallel + perpendicular * perpendicular);
}

fn distributionGGX(noh: f32, roughness: f32) -> f32 {
  let alpha = roughness * roughness;
  let alphaSquared = alpha * alpha;
  let denominator = noh * noh * (alphaSquared - 1.0) + 1.0;
  return alphaSquared / max(PI * denominator * denominator, 0.0001);
}

fn visibilitySmith(nov: f32, nol: f32, roughness: f32) -> f32 {
  let r = roughness + 1.0;
  let k = r * r * 0.125;
  return (nov / mix(nov, 1.0, k)) * (nol / mix(nol, 1.0, k));
}

fn windGlitter(normal: vec3f, viewDirection: vec3f, lightDirection: vec3f) -> f32 {
  let halfVector = normalize(viewDirection + lightDirection);
  let ndh = max(dot(normal, halfVector), 0.001);
  let wind = normalize(vec3f(0.73, 0.0, -0.68));
  let tangent = normalize(wind - normal * dot(wind, normal));
  let bitangent = normalize(cross(normal, tangent));
  let along = dot(halfVector, tangent) / ndh;
  let across = dot(halfVector, bitangent) / ndh;
  let alongVariance = 0.041;
  let acrossVariance = 0.024;
  let slopePdf = exp(-0.5 * (along * along / alongVariance + across * across / acrossVariance))
    / (2.0 * PI * sqrt(alongVariance * acrossVariance));
  return min(slopePdf / max(ndh * ndh * ndh * ndh, 0.001), 18.0);
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

@fragment
fn surfaceFragment(input: SurfaceOutput) -> @location(0) vec4f {
  let viewDirection = normalize(u.cameraTime.xyz - input.worldPosition);
  var normal = normalize(input.normal);
  if (dot(normal, viewDirection) < 0.0) {
    normal = -normal;
  }
  normal = microNormal(input.worldPosition, normal, input.foam);

  let lightDirection = normalize(u.sunDirection.xyz);
  let halfVector = normalize(viewDirection + lightDirection);
  let nov = max(0.001, dot(normal, viewDirection));
  let nol = max(0.0, dot(normal, lightDirection));
  let noh = max(0.0, dot(normal, halfVector));
  let fresnel = dielectricFresnel(nov);
  let roughness = mix(0.045, 0.16, saturate(input.foam * 0.9 + input.compression * 0.12));

  let reflected = skyRadiance(reflect(-viewDirection, normal));
  let opticalDepth = mix(0.75, 7.5, pow(1.0 - nov, 1.4)) + max(0.0, -input.waveHeight) * 0.9;
  let transmission = exp(-vec3f(0.53, 0.115, 0.032) * opticalDepth);
  let deepWater = vec3f(0.001, 0.028, 0.12);
  let scatter = vec3f(0.002, 0.235, 0.39) * (vec3f(1.0) - transmission);
  let refractedSky = skyRadiance(refract(-viewDirection, normal, 1.0 / 1.333)) * vec3f(0.024, 0.09, 0.15);
  var color = mix(deepWater * transmission + scatter + refractedSky, reflected, clamp(fresnel * 0.96 + 0.018, 0.0, 0.94));

  let specular = distributionGGX(noh, roughness) * visibilitySmith(nov, nol, roughness) * fresnel * nol;
  let glitter = windGlitter(normal, viewDirection, lightDirection) * pow(nol, 2.0) * (1.0 - input.foam);
  color += vec3f(8.8, 7.1, 4.8) * specular;
  color += vec3f(1.0, 0.72, 0.32) * glitter * 0.018;

  let crest = smoothstep(0.34, 1.28, input.waveHeight) * smoothstep(0.10, 0.74, 1.0 - normal.y);
  let forwardScatter = pow(saturate(dot(-lightDirection, viewDirection) * 0.5 + 0.5), 4.0);
  let backlit = pow(saturate(dot(-lightDirection, normal) * 0.5 + 0.5), 3.0);
  color += vec3f(0.018, 0.72, 0.78) * crest * (0.26 + backlit * 1.05 + forwardScatter * 0.28);

  // The solver supplies transported foam concentration and the resolved
  // surface supplies the instantaneous breaking event. Multi-scale noise only
  // models unresolved bubble porosity; it cannot create foam by itself.
  let time = u.cameraTime.w * u.resolutionMotion.z;
  let breakup = valueNoise(input.fieldCoordinates * 0.42 + vec2f(time * 0.026, -time * 0.018)) * 0.58
    + valueNoise(input.fieldCoordinates * 1.63 + vec2f(-time * 0.051, time * 0.039)) * 0.29
    + valueNoise(input.fieldCoordinates * 5.2 + vec2f(time * 0.11, -time * 0.08)) * 0.13;
  let fineBreakup = valueNoise(input.fieldCoordinates * 2.8 + vec2f(-time * 0.07, time * 0.05));
  let microBubbles = valueNoise(input.fieldCoordinates * 18.0 + vec2f(time * 0.22, -time * 0.17));
  let crestSelection = smoothstep(0.24, 0.98, input.waveHeight);
  let surfaceSlope = 1.0 - normal.y;
  let freshBreaking = crestSelection
    * smoothstep(0.08, 0.48, surfaceSlope)
    * smoothstep(-0.08, 0.46, input.compression);
  let porositySample = breakup * 0.52 + fineBreakup * 0.25 + microBubbles * 0.23;
  let bubblePorosity = exp((porositySample - 0.54) * 2.35);
  let foamSupport = mix(0.32, 1.0, smoothstep(0.05, 0.40, surfaceSlope))
    * mix(0.42, 1.0, crestSelection);
  let transportedFoam = smoothstep(0.10, 0.58, max(0.0, input.foam)) * foamSupport;
  let opticalDensity = (transportedFoam * 1.82 + freshBreaking * 1.32) * bubblePorosity;
  let foam = 1.0 - exp(-opticalDensity);
  let foamShadow = vec3f(0.42, 0.62, 0.72);
  let foamLight = vec3f(0.96, 1.00, 1.02) + vec3f(0.16, 0.085, 0.022) * pow(nol, 4.0);
  let foamColor = mix(foamShadow, foamLight, 0.78 + 0.22 * saturate(nol + backlit));
  color = mix(color, foamColor, smoothstep(0.12, 0.88, foam) * 0.78);

  let distanceToEye = distance(input.worldPosition, u.cameraTime.xyz);
  let aerial = smoothstep(48.0, 122.0, distanceToEye);
  color = mix(color, vec3f(0.055, 0.25, 0.39), aerial * 0.72);
  return vec4f(max(color, vec3f(0.0)), 1.0);
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

@fragment
fn backgroundFragment(input: BackgroundOutput) -> @location(0) vec4f {
  let uv = input.position.xy / u.resolutionMotion.xy;
  let aspect = u.resolutionMotion.x / u.resolutionMotion.y;
  var point = vec2f((uv.x - 0.5) * aspect * 2.0, (0.5 - uv.y) * 2.0);
  point += (u.pointerEnergy.xy - 0.5) * vec2f(0.035, -0.022);
  let direction = normalize(vec3f(point.x * 0.82, point.y * 0.78 + 0.035, 1.0));
  var color = skyRadiance(direction);

  let time = u.cameraTime.w * u.resolutionMotion.z;
  let polar = vec2f(atan2(point.y - 0.26, point.x + 0.22), length(point - vec2f(-0.22, 0.26)));
  let swirlCoordinates = vec2f(polar.x * 1.7 - polar.y * 3.4 - time * 0.012, polar.y * 4.1 + time * 0.004);
  let cloudWarp = fbm(swirlCoordinates * 0.58 + vec2f(3.1, -1.7));
  let cloud = fbm(swirlCoordinates + vec2f(cloudWarp * 2.3, -cloudWarp * 1.1));
  let cloudMask = smoothstep(0.60, 0.83, cloud) * smoothstep(-0.18, 0.9, point.y);
  color += vec3f(0.055, 0.22, 0.43) * cloudMask * 0.52;
  color *= 1.0 - cloudMask * 0.17;

  let horizon = exp(-abs(point.y + 0.018) * 31.0);
  let sunPoint = vec2f(0.36, 0.145);
  let sunDistance = length((point - sunPoint) * vec2f(1.0, 1.15));
  let sunDisc = 1.0 - smoothstep(0.012, 0.025, sunDistance);
  let sunHalo = exp(-sunDistance * 9.0);
  color += vec3f(2.8, 1.55, 0.58) * sunDisc;
  color += vec3f(0.34, 0.12, 0.018) * sunHalo * 0.26;
  color += vec3f(0.16, 0.45, 0.62) * horizon * 0.58;
  return vec4f(max(color, vec3f(0.0)), 1.0);
}
