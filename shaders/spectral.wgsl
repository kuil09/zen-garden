const GRID_SIZE: u32 = 256u;
const PI: f32 = 3.141592653589793;
const GRAVITY: f32 = 9.81;

struct ComputeParams {
  time: f32,
  delta: f32,
  mode: u32,
  stage: u32,
  direction: u32,
  frame: u32,
  padding0: f32,
  padding1: f32,
}

@group(0) @binding(0) var<uniform> params: ComputeParams;
@group(0) @binding(1) var<storage, read> initialSpectrum: array<vec4f>;
@group(0) @binding(2) var<storage, read> inputField: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> outputField: array<vec4f>;

fn complexMultiply(a: vec2f, b: vec2f) -> vec2f {
  return vec2f(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn signedFrequency(index: u32) -> f32 {
  if (index <= GRID_SIZE / 2u) {
    return f32(index);
  }
  return f32(i32(index) - i32(GRID_SIZE));
}

fn evolvedComponent(h0: vec2f, h0Mirror: vec2f, omega: f32) -> vec2f {
  let phase = omega * params.time;
  let positive = vec2f(cos(phase), sin(phase));
  let negative = vec2f(positive.x, -positive.y);
  return complexMultiply(h0, positive) + complexMultiply(vec2f(h0Mirror.x, -h0Mirror.y), negative);
}

@compute @workgroup_size(8, 8)
fn evolveSpectrum(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= GRID_SIZE || id.y >= GRID_SIZE) {
    return;
  }

  let index = id.y * GRID_SIZE + id.x;
  let mirrorX = (GRID_SIZE - id.x) % GRID_SIZE;
  let mirrorY = (GRID_SIZE - id.y) % GRID_SIZE;
  let mirrorIndex = mirrorY * GRID_SIZE + mirrorX;
  let waveNumber = vec2f(signedFrequency(id.x), signedFrequency(id.y));

  let kLarge = length(waveNumber * (2.0 * PI / 84.0));
  let kDetail = length(waveNumber * (2.0 * PI / 15.0));
  let omegaLarge = sqrt(GRAVITY * kLarge);
  let omegaDetail = sqrt(GRAVITY * kDetail);
  let source = initialSpectrum[index];
  let mirror = initialSpectrum[mirrorIndex];

  let large = evolvedComponent(source.xy, mirror.xy, omegaLarge);
  let detail = evolvedComponent(source.zw, mirror.zw, omegaDetail);
  outputField[index] = vec4f(large, detail);
}

fn reversedIndex(value: u32) -> u32 {
  return reverseBits(value) >> 24u;
}

fn transformPair(value: vec4f, twiddle: vec2f) -> vec4f {
  return vec4f(
    complexMultiply(value.xy, twiddle),
    complexMultiply(value.zw, twiddle)
  );
}

@compute @workgroup_size(8, 8)
fn fftPass(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= GRID_SIZE || id.y >= GRID_SIZE) {
    return;
  }

  let outputIndex = id.y * GRID_SIZE + id.x;
  if (params.mode == 0u) {
    let sourceX = select(id.x, reversedIndex(id.x), params.direction == 0u);
    let sourceY = select(id.y, reversedIndex(id.y), params.direction == 1u);
    outputField[outputIndex] = inputField[sourceY * GRID_SIZE + sourceX];
    return;
  }

  let axisIndex = select(id.x, id.y, params.direction == 1u);
  let span = 1u << (params.stage + 1u);
  let halfSpan = span >> 1u;
  let positionInSpan = axisIndex & (span - 1u);
  let butterflyIndex = positionInSpan & (halfSpan - 1u);
  let spanStart = axisIndex - positionInSpan;
  let evenAxis = spanStart + butterflyIndex;
  let oddAxis = evenAxis + halfSpan;

  var evenIndex: u32;
  var oddIndex: u32;
  if (params.direction == 0u) {
    evenIndex = id.y * GRID_SIZE + evenAxis;
    oddIndex = id.y * GRID_SIZE + oddAxis;
  } else {
    evenIndex = evenAxis * GRID_SIZE + id.x;
    oddIndex = oddAxis * GRID_SIZE + id.x;
  }

  let angle = 2.0 * PI * f32(butterflyIndex) / f32(span);
  let twiddle = vec2f(cos(angle), sin(angle));
  let evenValue = inputField[evenIndex];
  let oddValue = transformPair(inputField[oddIndex], twiddle);
  outputField[outputIndex] = select(evenValue + oddValue, evenValue - oddValue, positionInSpan >= halfSpan);
}
