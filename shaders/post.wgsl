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

fn aces(color: vec3f) -> vec3f {
  let a = 2.51;
  let b = 0.03;
  let c = 2.43;
  let d = 0.59;
  let e = 0.14;
  return clamp((color * (a * color + b)) / (color * (c * color + d) + e), vec3f(0.0), vec3f(1.0));
}

fn hash12(point: vec2f) -> f32 {
  let p = fract(point * vec2f(123.34, 345.45));
  return fract(p.x * p.y * (p.x + p.y + 34.345));
}

@fragment
fn postFragment(input: PostOutput) -> @location(0) vec4f {
  let uv = input.uv;
  let pixel = 1.0 / u.resolutionMotion.xy;
  let radialDirection = uv - 0.5;
  let aberration = radialDirection * dot(radialDirection, radialDirection) * 0.0022;
  let center = textureSample(sceneTexture, sceneSampler, uv).rgb;
  var color = vec3f(
    textureSample(sceneTexture, sceneSampler, uv + aberration).r,
    center.g,
    textureSample(sceneTexture, sceneSampler, uv - aberration).b
  );
  let localBlur = (
    textureSample(sceneTexture, sceneSampler, uv + vec2f(pixel.x, 0.0)).rgb
      + textureSample(sceneTexture, sceneSampler, uv - vec2f(pixel.x, 0.0)).rgb
      + textureSample(sceneTexture, sceneSampler, uv + vec2f(0.0, pixel.y)).rgb
      + textureSample(sceneTexture, sceneSampler, uv - vec2f(0.0, pixel.y)).rgb
  ) * 0.25;
  color += (center - localBlur) * 0.46;

  var bloom = vec3f(0.0);
  var weight = 0.0;
  for (var ring = 0; ring < 3; ring = ring + 1) {
    let radius = f32(ring * ring + 1) * mix(3.0, 9.0, f32(ring) / 2.0);
    for (var directionIndex = 0; directionIndex < 8; directionIndex = directionIndex + 1) {
      let angle = f32(directionIndex) * 0.78539816339 + f32(ring) * 0.31;
      let offset = vec2f(cos(angle), sin(angle)) * pixel * radius;
      let sampleColor = textureSample(sceneTexture, sceneSampler, uv + offset).rgb;
      let brightness = smoothstep(0.72, 2.4, luminance(sampleColor));
      let sampleWeight = mix(0.7, 0.24, f32(ring) / 2.0);
      bloom += sampleColor * brightness * sampleWeight;
      weight += sampleWeight;
    }
  }
  bloom /= max(weight, 0.001);
  color += bloom * 0.24;

  color *= u.frameExposure.y;
  color = aces(color);
  color = pow(color, vec3f(0.94));
  color *= vec3f(0.96, 1.01, 1.055);
  let gradedLuminance = luminance(color);
  color = mix(vec3f(gradedLuminance), color, 1.17);
  color += vec3f(-0.012, 0.018, 0.032) * smoothstep(0.08, 0.72, gradedLuminance);

  let vignetteDistance = length((uv - 0.5) * vec2f(0.82, 1.0));
  let vignette = 1.0 - smoothstep(0.38, 0.78, vignetteDistance) * 0.42;
  color *= vignette;
  let grain = hash12(input.position.xy + u.cameraTime.w * 61.0) - 0.5;
  color += grain * (0.0065 / max(u.resolutionMotion.w, 0.75));
  return vec4f(clamp(color, vec3f(0.0), vec3f(1.0)), 1.0);
}
