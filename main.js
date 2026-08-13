const canvas = document.querySelector('#garden');
const notice = document.querySelector('#notice');
const motionToggle = document.querySelector('#motionToggle');
const buttonText = motionToggle.querySelector('.button-text');
const intensitySlider = document.querySelector('#intensity');
const quietToggle = document.querySelector('#quietToggle');
const clock = document.querySelector('#clock');
const page = document.querySelector('main');

const shader = /* wgsl */ `
struct Uniforms {
  resolution: vec2f,
  time: f32,
  intensity: f32,
  pointer: vec2f,
  pointerActive: f32,
  padding: f32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;

@vertex
fn vertexMain(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
  var positions = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  return vec4f(positions[index], 0.0, 1.0);
}

fn hash(p: vec2f) -> f32 {
  return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453);
}

fn ripple(p: vec2f, source: vec2f, frequency: f32, speed: f32) -> f32 {
  let d = length(p - source);
  return sin(d * frequency - u.time * speed) * exp(-d * 1.25);
}

@fragment
fn fragmentMain(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let uv = pos.xy / u.resolution;
  let ratio = u.resolution.x / u.resolution.y;
  var p = (uv - 0.5) * vec2f(ratio, 1.0);
  let t = u.time * (0.15 + u.intensity * 0.24);

  let current = vec2f(
    0.32 * sin(t * 0.9) + 0.10 * sin(t * 2.7),
    0.18 * cos(t * 0.7) + 0.08 * sin(t * 1.9)
  );
  let companion = vec2f(-0.34 + 0.18 * cos(t * 1.2), 0.22 * sin(t * 0.65));
  let hand = (u.pointer - 0.5) * vec2f(ratio, 1.0);
  let pointerWave = ripple(p, hand, 19.0, 2.8) * u.pointerActive;
  let water = ripple(p, current, 15.0, 1.4) * 0.82 + ripple(p, companion, 11.0, 1.05) * 0.56 + pointerWave * 0.58;

  let bend = p + vec2f(sin(p.y * 6.0 + t) * 0.07, cos(p.x * 5.0 - t) * 0.05);
  let lines = sin((bend.x + bend.y * 0.72 + water * 0.22) * 15.0 + t * 0.6);
  let softLines = smoothstep(0.84, 1.0, lines) * (0.24 + u.intensity * 0.32);
  let glow = exp(-length(p - current) * 1.4) * 0.42 + exp(-length(p - companion) * 1.8) * 0.18;
  let horizon = smoothstep(-0.65, 0.8, p.y + water * 0.08);
  let grain = (hash(floor(pos.xy * 0.5)) - 0.5) * 0.028;

  let night = vec3f(0.045, 0.065, 0.115);
  let blue = vec3f(0.075, 0.20, 0.31);
  let coral = vec3f(1.0, 0.37, 0.19);
  var color = mix(night, blue, horizon * 0.8 + 0.1);
  color += coral * (glow * 0.72 + softLines * 0.72);
  color += vec3f(0.88, 0.67, 0.41) * max(water, 0.0) * 0.10;
  color += grain;
  let vignette = 1.0 - smoothstep(0.35, 1.2, length(p * vec2f(0.8, 1.0))) * 0.48;
  return vec4f(color * vignette, 1.0);
}
`;

let device;
let context;
let uniformBuffer;
let bindGroup;
let pipeline;
const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
let paused = false;
let intensity = 0.52;
let pointer = [0.5, 0.5];
let pointerActive = 0;
let startTime = performance.now();
let pausedElapsed = 0;

function updateClock() {
  clock.textContent = new Intl.DateTimeFormat('ko-KR', { hour: '2-digit', minute: '2-digit', hour12: false }).format(new Date());
}

function setPaused(next) {
  if (next === paused) return;
  paused = next;
  if (paused) pausedElapsed = performance.now() - startTime;
  else startTime = performance.now() - pausedElapsed;
  motionToggle.setAttribute('aria-pressed', String(paused));
  buttonText.textContent = paused ? '다시 흐르게 하기' : '잠시 멈추기';
}

function setQuiet(next) {
  page.classList.toggle('quiet', next);
  quietToggle.setAttribute('aria-pressed', String(next));
  quietToggle.textContent = next ? '글자 다시 보기' : '글자 숨기기';
}

function resize() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
  const height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
    context.configure({ device, format: navigator.gpu.getPreferredCanvasFormat(), alphaMode: 'opaque' });
  }
}

async function initialize() {
  if (!navigator.gpu) throw new Error('WebGPU is unavailable');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No GPU adapter available');
  device = await adapter.requestDevice();
  context = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'opaque' });

  uniformBuffer = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const module = device.createShaderModule({ code: shader });
  pipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module, entryPoint: 'vertexMain' },
    fragment: { module, entryPoint: 'fragmentMain', targets: [{ format }] },
    primitive: { topology: 'triangle-list' },
  });
  bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: uniformBuffer } }] });
  resize();
  requestAnimationFrame(draw);
}

function draw(now) {
  resize();
  const elapsed = (paused ? pausedElapsed : now - startTime) / 1000;
  const data = new Float32Array([canvas.width, canvas.height, elapsed, intensity, pointer[0], pointer[1], pointerActive, 0]);
  device.queue.writeBuffer(uniformBuffer, 0, data);
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({ colorAttachments: [{ view: context.getCurrentTexture().createView(), clearValue: { r: 0.04, g: 0.05, b: 0.09, a: 1 }, loadOp: 'clear', storeOp: 'store' }] });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.draw(3);
  pass.end();
  device.queue.submit([encoder.finish()]);
  requestAnimationFrame(draw);
}

function trackPointer(event) {
  const rect = canvas.getBoundingClientRect();
  pointer = [(event.clientX - rect.left) / rect.width, 1 - (event.clientY - rect.top) / rect.height];
  pointerActive = 1;
}

canvas.addEventListener('pointermove', trackPointer);
canvas.addEventListener('pointerdown', trackPointer);
canvas.addEventListener('pointerleave', () => { pointerActive = 0; });
window.addEventListener('resize', resize);
motionToggle.addEventListener('click', () => setPaused(!paused));
intensitySlider.addEventListener('input', (event) => { intensity = Number(event.target.value) / 100; });
quietToggle.addEventListener('click', () => setQuiet(!page.classList.contains('quiet')));
window.addEventListener('keydown', (event) => {
  if (event.key.toLowerCase() === 'h' && event.target === document.body) setQuiet(!page.classList.contains('quiet'));
  if (event.key === 'Escape') setQuiet(false);
});
updateClock();
setInterval(updateClock, 30_000);
if (prefersReducedMotion) setPaused(true);

initialize().catch((error) => {
  console.warn(error);
  notice.hidden = false;
});
