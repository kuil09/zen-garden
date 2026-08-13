struct Uniforms {
  viewProjection: mat4x4f,
  cameraTime: vec4f,
  resolutionMotion: vec4f,
  pointerEnergy: vec4f,
  frameExposure: vec4f,
  sunDirection: vec4f,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var sceneSampler: sampler;
@group(0) @binding(2) var sceneTexture: texture_2d<f32>;

struct PostOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
}

@vertex
fn postVertex(@builtin(vertex_index) index: u32) -> PostOutput {
  var positions = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f(3.0, -1.0),
    vec2f(-1.0, 3.0)
  );
  var output: PostOutput;
  output.position = vec4f(positions[index], 0.0, 1.0);
  output.uv = positions[index] * vec2f(0.5, -0.5) + 0.5;
  return output;
}

fn luminance(color: vec3f) -> f32 {
  return dot(color, vec3f(0.2126, 0.7152, 0.0722));
}

fn hash12(point: vec2f) -> f32 {
  let p = fract(point * vec2f(123.34, 345.45));
  return fract(p.x * p.y * (p.x + p.y + 34.345));
}

fn valueNoise2(point: vec2f) -> f32 {
  let cell = floor(point);
  let local = fract(point);
  let blend = local * local * (3.0 - 2.0 * local);
  let a = hash12(cell);
  let b = hash12(cell + vec2f(1.0, 0.0));
  let c = hash12(cell + vec2f(0.0, 1.0));
  let d = hash12(cell + vec2f(1.0, 1.0));
  return mix(mix(a, b, blend.x), mix(c, d, blend.x), blend.y);
}

fn tap(uv: vec2f) -> vec3f {
  return textureSample(sceneTexture, sceneSampler, uv).rgb;
}

// The scene pass already prints in flat tones, so a Sobel over the frame finds
// exactly the boundaries a carver would cut: one line per plate edge.
fn inkEdge(uv: vec2f, texel: vec2f) -> f32 {
  let tl = luminance(tap(uv + texel * vec2f(-1.0, -1.0)));
  let tc = luminance(tap(uv + texel * vec2f( 0.0, -1.0)));
  let tr = luminance(tap(uv + texel * vec2f( 1.0, -1.0)));
  let ml = luminance(tap(uv + texel * vec2f(-1.0,  0.0)));
  let mr = luminance(tap(uv + texel * vec2f( 1.0,  0.0)));
  let bl = luminance(tap(uv + texel * vec2f(-1.0,  1.0)));
  let bc = luminance(tap(uv + texel * vec2f( 0.0,  1.0)));
  let br = luminance(tap(uv + texel * vec2f( 1.0,  1.0)));
  let gx = (tr + 2.0 * mr + br) - (tl + 2.0 * ml + bl);
  let gy = (bl + 2.0 * bc + br) - (tl + 2.0 * tc + tr);
  return sqrt(gx * gx + gy * gy);
}

@fragment
fn postFragment(input: PostOutput) -> @location(0) vec4f {
  let resolution = u.resolutionMotion.xy;
  let texel = 1.0 / resolution;
  let pixel = input.uv * resolution;

  // Plate misregistration: each colour block was pulled by hand and none of them
  // land quite on top of one another.
  let slip = texel * 0.85;
  var color = vec3f(
    tap(input.uv + slip * vec2f( 0.9, -0.4)).r,
    tap(input.uv).g,
    tap(input.uv + slip * vec2f(-0.7,  0.5)).b
  );

  let edge = inkEdge(input.uv, texel);
  // Brush weight: the line thins and thickens along its length.
  let weight = 0.55 + 0.45 * valueNoise2(pixel * 0.055);
  let ink = smoothstep(0.055, 0.20, edge * weight);
  color = mix(color, vec3f(0.043, 0.075, 0.129), ink * 0.78);

  // Paper: long fibres in the sheet plus a coarser mottle where the ink sat.
  let fibre = valueNoise2(vec2f(pixel.x * 0.9, pixel.y * 0.07));
  let mottle = valueNoise2(pixel * 0.012);
  color *= 1.0 - (fibre - 0.5) * 0.055 - (mottle - 0.5) * 0.10;
  color += (hash12(pixel + u.frameExposure.w) - 0.5) * 0.012;

  // The block never inks evenly to the edge of the sheet.
  let centred = input.uv - 0.5;
  let edgeFade = smoothstep(0.86, 0.36, length(centred * vec2f(1.06, 1.0)));
  color = mix(vec3f(0.902, 0.862, 0.769), color, 0.62 + 0.38 * edgeFade);

  return vec4f(clamp(color, vec3f(0.0), vec3f(1.0)), 1.0);
}
