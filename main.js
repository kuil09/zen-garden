import spectralShader from './shaders/spectral.wgsl?raw';
import dynamicShader from './shaders/dynamic.wgsl?raw';
import resolveShader from './shaders/resolve.wgsl?raw';
import breakersShader from './shaders/breakers.wgsl?raw';
import sceneShader from './shaders/scene.wgsl?raw';
import waveGeometryShader from './shaders/wave-geometry.wgsl?raw';
import postShader from './shaders/post.wgsl?raw';
import sprayShader from './shaders/spray.wgsl?raw';
import { toggleOceanSound, updateBreakerPosition, setListenerPosition } from './ocean-sound.js';

const MAX_PARTICLES = 8192;
const SPRAY_WORKGROUP_SIZE = 64;

// Inline WGSL for the spray point-sprite render pass. Kept here (not a .wgsl
// file) because it shares the Particle struct layout with shaders/spray.wgsl.
const SPRAY_VERTEX_WGSL = `
  struct Uniforms {
    viewProjection: mat4x4f,
    cameraPos: vec4f,
    viewport: vec4f,
    pointer: vec4f,
    misc: vec4f,
    sun: vec4f,
    camRight: vec4f,
    camUp: vec4f,
    forward: vec4f,
  }
  struct Particle {
    position: vec4f,  // xyz world space, w = life (seconds remaining)
    velocity: vec4f,  // xyz velocity, w = size
    color: vec4f,     // rgb tint, w = alpha
    spin: vec4f,      // x = isSpray flag, y = spin, z = spin rate, w = unused
  }
  @group(0) @binding(0) var<uniform> u: Uniforms;
  @group(0) @binding(1) var<storage, read> particles: array<Particle>;

  struct VSOut {
    @builtin(position) position: vec4f,
    @location(0) color: vec4f,
    @location(1) uv: vec2f,
  }

  // Two triangles forming a camera-facing quad in local [-1,1] space.
  const QUAD = array<vec2f, 6>(
    vec2f(-1.0, -1.0),
    vec2f( 1.0, -1.0),
    vec2f(-1.0,  1.0),
    vec2f(-1.0,  1.0),
    vec2f( 1.0, -1.0),
    vec2f( 1.0,  1.0),
  );

  @vertex
  fn main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
    let p = particles[ii];
    var out: VSOut;
    if (p.position.w <= 0.0) {
      out.position = vec4f(0.0, 0.0, 0.0, 0.0);
      out.color = vec4f(0.0);
      out.uv = vec2f(0.0);
      return out;
    }

    let corner = QUAD[vi];
    let angle = p.spin.y;
    let ca = cos(angle);
    let sa = sin(angle);
    let rc = vec2f(corner.x * ca - corner.y * sa, corner.x * sa + corner.y * ca);

    // Local billboard-space offset, in units of world-space particle size.
    var local = rc;
    if (p.spin.x > 0.5) {
      // Shard: elongate along the particle's velocity, projected onto the
      // billboard plane, so it reads as a streak rather than a dot.
      var dir = vec3f(0.0, 1.0, 0.0);
      let velLen = length(p.velocity.xyz);
      if (velLen > 1e-4) { dir = p.velocity.xyz / velLen; }
      let vRight = dot(dir, u.camRight.xyz);
      let vUp = dot(dir, u.camUp.xyz);
      let vlen = max(length(vec2f(vRight, vUp)), 1e-3);
      let along = vec2f(vRight, vUp) / vlen;
      let perp = vec2f(-along.y, along.x);
      let comp = vec2f(dot(rc, along), dot(rc, perp));
      local = comp.x * along * 1.9 + comp.y * perp * 0.7;
    }

    let worldOffset = (u.camRight.xyz * local.x + u.camUp.xyz * local.y) * p.velocity.w;
    let worldPos = p.position.xyz + worldOffset;
    out.position = u.viewProjection * vec4f(worldPos, 1.0);
    out.uv = rc;
    out.color = p.color;
    return out;
  }
`;
const SPRAY_FRAGMENT_WGSL = `
  struct VSOut {
    @builtin(position) position: vec4f,
    @location(0) color: vec4f,
    @location(1) uv: vec2f,
  }
  @fragment
  fn main(in: VSOut) -> @location(0) vec4f {
    // Hard elliptical cut: no soft radial alpha, just a flat plate with a
    // separate keyline so it reads as woodblock ink rather than a glow.
    let r = length(in.uv);
    if (r > 1.0) { discard; }
    let edge = smoothstep(0.80, 0.96, r);
    let rgb = mix(in.color.rgb, vec3f(0.12, 0.22, 0.45), edge);
    return vec4f(rgb, in.color.a);
  }
`;

const canvas = document.querySelector('#ocean');
const experience = document.querySelector('#experience');
const motionToggle = document.querySelector('#motionToggle');
const soundToggle = document.querySelector('#soundToggle');
const status = document.querySelector('#status');
const notice = document.querySelector('#notice');

const SIMULATION_SIZE = 256;
const SIMULATION_CELLS = SIMULATION_SIZE * SIMULATION_SIZE;
const FFT_STAGES = Math.log2(SIMULATION_SIZE);
const OCEAN_COLUMNS = 360;
const OCEAN_ROWS = 248;
const WAVE_COLUMNS = 288;
const WAVE_ROWS = 176;
const WAVE_INSTANCES = 3;
const CLAW_COLUMNS = 1024;
const CLAW_ROWS = 40;
const DYNAMIC_SUBSTEPS = 4;
const UNIFORM_FLOATS = 52;
const UNIFORM_SIZE = UNIFORM_FLOATS * Float32Array.BYTES_PER_ELEMENT;
const COMPUTE_PARAMS_SIZE = 32;
const RESOLVE_PARAMS_SIZE = 16;
const HDR_FORMAT = 'rgba16float';
const SAMPLE_COUNT = 4;

// Breaker placement. The wave sheets are placed from simulation data: a
// compute pass scores every cell for how hard it is breaking, a reduction
// collapses the scores onto a coarse block lattice, and the CPU reads the
// lattice back and maintains up to one breaker per wave-sheet instance.
const BREAKER_DOMAIN = 84;
const BREAKER_BLOCKS = 8;
const BLOCK_FLOATS = 16;
const SUMMARY_FLOATS = BREAKER_BLOCKS * BREAKER_BLOCKS * BLOCK_FLOATS;
const READBACK_INTERVAL = 20;
const BREAKER_MATCH_RADIUS = 26;
const BREAKER_ENVELOPE_SECONDS = 2.2;

const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

// ============================================================================
// #10 — Deterministic Hero Wave state machine + development capture mode.
//
// The hero wave is a deterministic preset driven by an explicit normalized
// phase (0..1), separate from the simulation detection that still feeds the
// ambient breakers. Same preset/seed/phase/camera must always produce the same
// scene so visual regressions can be measured (#5).
// ============================================================================

// Deterministic 32-bit RNG (mulberry32) is defined later in this file; reused
// here so the hero curves stay free of Math.random().

function clamp01(x) { return x < 0 ? 0 : x > 1 ? 1 : x; }
function lerp(a, b, t) { return a + (b - a) * t; }
function smoothstep(e0, e1, x) {
  const t = clamp01((x - e0) / (e1 - e0));
  return t * t * (3 - 2 * t);
}
// Piecewise-linear interpolation across a list of [phase, value] knots.
function curveAt(knots, phase) {
  if (phase <= knots[0][0]) return knots[0][1];
  const last = knots[knots.length - 1];
  if (phase >= last[0]) return last[1];
  for (let i = 0; i < knots.length - 1; i++) {
    const [p0, v0] = knots[i];
    const [p1, v1] = knots[i + 1];
    if (phase >= p0 && phase <= p1) {
      return lerp(v0, v1, smoothstep(p0, p1, phase));
    }
  }
  return last[1];
}

// Art-directed phase curve sets. Each phase segment (gather/rise/hook/suspend/
// tear/settle) is a list of [phase, value] knots driving the breaker art params.
const HERO_PRESETS = {
  kanagawa: {
    cycleSeconds: 18,
    anchor: { mode: 'camera-relative', x: -0.18, y: -0.08, depth: 42 },
    height:        [[0.00, 0.30], [0.16, 0.55], [0.36, 0.95], [0.58, 1.00], [0.72, 0.95], [0.86, 0.60], [1.00, 0.30]],
    faceConcavity: [[0.00, 0.10], [0.36, 0.55], [0.58, 0.70], [0.86, 0.30], [1.00, 0.10]],
    crestMass:     [[0.00, 0.20], [0.36, 0.60], [0.58, 0.85], [0.72, 0.80], [1.00, 0.20]],
    hookReach:     [[0.00, 0.00], [0.36, 0.45], [0.58, 0.80], [0.72, 0.75], [1.00, 0.00]],
    tongueDrop:    [[0.00, 0.00], [0.36, 0.20], [0.58, 0.55], [0.72, 0.50], [1.00, 0.00]],
    ridgeBow:      [[0.00, 0.00], [0.36, 0.35], [0.58, 0.60], [0.86, 0.20], [1.00, 0.00]],
    foamVisibility:[[0.00, 0.00], [0.36, 0.40], [0.58, 0.90], [0.72, 1.00], [0.86, 0.50], [1.00, 0.00]],
    secondaryRidge:[[0.00, 0.00], [0.36, 0.30], [0.58, 0.65], [0.72, 0.55], [1.00, 0.00]],
    surfaceFlow:   [[0.00, 0.10], [0.36, 0.60], [0.58, 0.85], [0.86, 0.40], [1.00, 0.10]],
  },
};

// Global development/hero configuration parsed from the URL. Defaults are set so
// the existing simulation-driven composition still runs unchanged when no flags
// are present.
const DEV = {
  hero: false,
  preset: 'kanagawa',
  seed: 7,
  phase: null,      // null => drive from elapsed time (non-deterministic)
  camera: 'default', // 'default' | 'print' (front) | 'yaw-20' | 'yaw+20'
  capture: false,    // ?capture=1 deterministic capture mode (#5)
  debug: 0,         // 0=off, 1=regions tint (see docs/art-direction-params.md)
  test: false,      // ?test=1 => live slider panel for empirical tuning
};

// Test-mode live overrides. Sliders write these; they replace the simulation-
// derived breaker targets so each value can be tuned by eye. Null = use default.
const TEST = {
  active: false,
  heightGain: null, crestPeak: null, crestWidth: null,
  curlWaves: null, taper: null, thetaSpan: null, throwGain: null, detailGain: null,
  radius: null, hookScale: 0, tongueScale: 0,
};

function parseDevParams() {
  const q = new URLSearchParams(window.location.search);
  if (q.has('hero')) DEV.hero = q.get('hero') === '1' || q.get('hero') === 'true';
  if (q.has('preset')) DEV.preset = q.get('preset');
  if (q.has('seed')) DEV.seed = parseInt(q.get('seed'), 10) || 0;
  if (q.has('phase')) DEV.phase = clamp01(parseFloat(q.get('phase')));
  if (q.has('camera')) DEV.camera = q.get('camera');
  if (q.has('capture')) DEV.capture = q.get('capture') === '1' || q.get('capture') === 'true';
  if (q.has('debug')) DEV.debug = q.get('debug') === 'regions' ? 1 : (parseInt(q.get('debug'), 10) || 0);
  if (q.has('test')) DEV.test = q.get('test') === '1' || q.get('test') === 'true';
  TEST.active = DEV.test;
}

// Build a live slider panel for empirical art-direction tuning. Injected when
// ?test=1 is present. All strings use single quotes to avoid JSON escaping
// issues inside the edit tool.
function buildTestPanel() {
  const panel = document.createElement('div');
  panel.style.cssText = 'position:fixed;top:10px;right:10px;width:280px;max-height:90vh;overflow-y:auto;background:rgba(0,0,0,0.85);color:#eee;padding:10px;border-radius:8px;font:12px/1.4 monospace;z-index:9999;';

  const title = document.createElement('b');
  title.textContent = '🎨 art-direction test mode';
  panel.appendChild(title);

  panel.appendChild(document.createElement('br'));
  const sub = document.createElement('small');
  sub.textContent = '?test=1 — 슬라이더로 실시간 튜닝';
  panel.appendChild(sub);

  const hr = document.createElement('hr');
  hr.style.borderColor = '#444';
  panel.appendChild(hr);

  // 슬라이더 설정 (함수 내 closure로 유지)
  const SLIDER_CFGS = [
    { key: 'heightGain', label: '높이(heightGain)', min: 0, max: 2, step: 0.01, def: 1.0 },
    { key: 'radius', label: '반경(radius)', min: 2, max: 12, step: 0.1, def: 8 },
    { key: 'crestPeak', label: '능선위치(crestPeak)', min: 0, max: 1, step: 0.01, def: 0.55 },
    { key: 'crestWidth', label: '능선폭(crestWidth)', min: 0.2, max: 1, step: 0.01, def: 0.55 },
    { key: 'curlWaves', label: '말림횟수(curlWaves)', min: 2, max: 8, step: 0.1, def: 5.5 },
    { key: 'taper', label: '테이퍼(taper)', min: 0, max: 1, step: 0.01, def: 0.5 },
    { key: 'thetaSpan', label: '각도범위(thetaSpan)', min: 1, max: 6, step: 0.1, def: 5 },
    { key: 'throwGain', label: '던지기(throwGain)', min: 0, max: 3, step: 0.05, def: 1.8 },
    { key: 'detailGain', label: '디테일(detailGain)', min: 0, max: 2, step: 0.05, def: 1.5 },
    { key: 'hookScale', label: '훅스케일(hookScale)', min: -0.5, max: 1.5, step: 0.05, def: 0 },
    { key: 'tongueScale', label: '혀스케일(tongueScale)', min: -0.5, max: 1.5, step: 0.05, def: 0 },
  ];

  const inputRefs = [];  // {input, valSpan, cfg}

  SLIDER_CFGS.forEach(cfg => {
    const row = document.createElement('div');
    row.style.marginBottom = '4px';

    const label = document.createElement('label');
    label.style.cssText = 'display:flex;align-items:center;gap:4px;';

    const nameSpan = document.createElement('span');
    nameSpan.style.flex = '1';
    nameSpan.textContent = cfg.label;
    label.appendChild(nameSpan);

    const input = document.createElement('input');
    input.type = 'range';
    input.min = String(cfg.min);
    input.max = String(cfg.max);
    input.step = String(cfg.step);
    input.value = String(cfg.def);
    input.style.cssText = 'flex:2;width:80px;';
    input.dataset.key = cfg.key;

    const valSpan = document.createElement('span');
    valSpan.style.cssText = 'width:32px;text-align:right;font-size:11px;';
    valSpan.textContent = cfg.def.toFixed(2);

    input.oninput = () => {
      valSpan.textContent = parseFloat(input.value).toFixed(2);
      TEST[cfg.key] = parseFloat(input.value);
    };

    label.appendChild(input);
    label.appendChild(valSpan);
    row.appendChild(label);
    panel.appendChild(row);
    inputRefs.push({ input, valSpan, cfg });
  });

  if (DEV.hero) {
    const pRow = document.createElement('div');
    pRow.style.marginTop = '8px';
    const pLabel = document.createElement('label');
    pLabel.style.cssText = 'display:flex;align-items:center;gap:4px;';
    const pName = document.createElement('span');
    pName.style.flex = '1';
    pName.textContent = 'phase';
    pLabel.appendChild(pName);
    const pInput = document.createElement('input');
    pInput.type = 'range';
    pInput.min = '0';
    pInput.max = '1';
    pInput.step = '0.01';
    pInput.value = String(DEV.phase ?? 0.5);
    pInput.style.cssText = 'flex:2;width:80px;';
    const pVal = document.createElement('span');
    pVal.style.cssText = 'width:32px;text-align:right;font-size:11px;';
    pVal.textContent = (DEV.phase ?? 0.5).toFixed(2);
    pInput.oninput = () => {
      DEV.phase = parseFloat(pInput.value);
      pVal.textContent = DEV.phase.toFixed(2);
    };
    pLabel.appendChild(pInput);
    pLabel.appendChild(pVal);
    pRow.appendChild(pLabel);
    panel.appendChild(pRow);
  }

  const resetBtn = document.createElement('button');
  resetBtn.textContent = 'Reset defaults';
  resetBtn.style.cssText = 'margin-top:8px;padding:3px 10px;cursor:pointer;';
  resetBtn.onclick = () => {
    // TEST 초기화
    SLIDER_CFGS.forEach(cfg => {
      TEST[cfg.key] = cfg.key === 'hookScale' || cfg.key === 'tongueScale' ? 0 : null;
    });
    // 슬라이더 입력값도 기본값으로
    inputRefs.forEach(ref => {
      ref.input.value = String(ref.cfg.def);
      ref.valSpan.textContent = ref.cfg.def.toFixed(2);
    });
  };
  panel.appendChild(resetBtn);

  document.body.appendChild(panel);
}


// Returns the art-parameter set for a given normalized phase, or null if the
// preset is unknown. Consumed by the breaker placement code so the hero macro
// silhouette is driven by phase curves rather than live simulation detection.
function heroArtParams(phase) {
  const preset = HERO_PRESETS[DEV.preset];
  if (!preset) return null;
  const p = clamp01(phase);
  return {
    height:        curveAt(preset.height, p),
    faceConcavity: curveAt(preset.faceConcavity, p),
    crestMass:     curveAt(preset.crestMass, p),
    hookReach:     curveAt(preset.hookReach, p),
    tongueDrop:    curveAt(preset.tongueDrop, p),
    ridgeBow:      curveAt(preset.ridgeBow, p),
    foamVisibility:curveAt(preset.foamVisibility, p),
    secondaryRidge:curveAt(preset.secondaryRidge, p),
    surfaceFlow:   curveAt(preset.surfaceFlow, p),
  };
}

let device;
// #10: current normalized hero phase (0..1). Driven by DEV.phase when set for
// deterministic capture, otherwise loops from elapsed time.
let heroPhase = 0;
let context;
let format;

// ---- GPU resources: core buffers ----
let uniformBuffer;
let evolveParamsBuffer;
let dynamicParamsBuffer;
let resolveParamsBuffer;
let initialSpectrumBuffer;
let fftBuffers;
let dynamicBuffers;
let oceanBuffers;

// ---- GPU resources: breaker detection ----
let breakerConfigBuffer;
let breakerScoreBuffer;
let breakerSummaryBuffer;
let breakerStagingBuffer;
let breakerParamsBuffer;
let foamFingerBuffer;
let breakerBindGroups;

// ---- Pipelines: compute ----
let spectralPipeline;
let fftPipeline;
let dynamicPipeline;
let resolvePipeline;
let scoreBreakersPipeline;
let reduceBreakersPipeline;
let evolveBindGroup;
let fftPasses;
let dynamicBindGroups;
let resolveBindGroups;

// ---- Pipelines: scene render ----
let backgroundPipeline;
let oceanPipeline;
let postPipeline;
let postBindGroup;
let surfaceBindGroups;
let waveBindGroups;
let oceanGrid;
let waveGrid;
let clawGrid;
let clawPipeline;

// ---- Spray particle system (compute spawn/update + point render) ----
let sprayModule;
let spraySpawnPipeline;
let sprayUpdatePipeline;
let sprayBuffer;
let sprayParamsBuffer;
let sprayCrestBuffer;
let sprayBindGroup;
let sprayRenderPipeline;
let sprayRenderBindGroup;

// Camera world position for breaker spawn filtering
let cameraWorldPos = [0.0, 0.0, 0.0];
let wavePipeline;
let foamPipeline;
let sceneTexture;
let multisampleTexture;
let depthTexture;
let sceneSampler;
let frameRequest;
let startTime = performance.now();
let pausedElapsed = 0;
let previousFrame = performance.now();
let moving = true;
let pointer = [0.5, 0.5];
let smoothPointer = [0.5, 0.5];
let interactionEnergy = 0;
let renderScale = window.innerWidth < 720 ? 0.84 : 0.96;
let averageFrameTime = 16.7;
let adaptationElapsed = 0;
let oceanBufferIndex = 0;
let dynamicBufferIndex = 0;
let frameIndex = 0;
let dynamicTime = 0;

// ---- Breaker anchor state (CPU side of shaders/breakers.wgsl) ----
const breakerAnchors = Array.from({ length: 3 }, (_, index) => ({
  active: false,
  centerX: 0,
  centerZ: 0,
  dirX: 1,
  dirZ: 0,
  extent: 22,
  radius: 8,
  heightGain: 2.5,
  curlRate: 0.70,
  curlWaves: 5.5,
  phaseOffset: index * 2.4,
  crestPeak: 0.55,
  crestWidth: 0.55,
  thetaSpan: 5.0,
  taper: 0.5,
  throwGain: 1.8,
  detailGain: 1.5,
  envelope: 0,
  targetEnvelope: 0,
  claimed: false,
  component: null,
}));
let breakerReadbackPending = false;
let breakerSummary = null;
let breakerFrameCounter = 0;
let breakerActivity = 0;

// #6 debug: ?forcebreaker=1 forces a single dominant breaker in front of the
// camera so the waveProfile() silhouette change can be verified in isolation,
// regardless of whether the live simulation detection happens to spawn one.
const FORCE_BREAKER = new URLSearchParams(window.location.search).has('forcebreaker');

function normalize3(vector) {
  const length = Math.hypot(vector[0], vector[1], vector[2]) || 1;
  return [vector[0] / length, vector[1] / length, vector[2] / length];
}

function subtract3(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function cross3(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function dot3(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function perspectiveLeftHanded(fieldOfView, aspect, near, far) {
  const focal = 1 / Math.tan(fieldOfView / 2);
  const range = far / (far - near);
  return new Float32Array([
    focal / aspect, 0, 0, 0,
    0, focal, 0, 0,
    0, 0, range, 1,
    0, 0, -near * range, 0,
  ]);
}

function lookAtLeftHanded(eye, target, worldUp = [0, 1, 0]) {
  const forward = normalize3(subtract3(target, eye));
  const right = normalize3(cross3(worldUp, forward));
  const up = cross3(forward, right);
  return new Float32Array([
    right[0], up[0], forward[0], 0,
    right[1], up[1], forward[1], 0,
    right[2], up[2], forward[2], 0,
    -dot3(right, eye), -dot3(up, eye), -dot3(forward, eye), 1,
  ]);
}

function multiplyMatrices(a, b) {
  const result = new Float32Array(16);
  for (let column = 0; column < 4; column += 1) {
    for (let row = 0; row < 4; row += 1) {
      let value = 0;
      for (let index = 0; index < 4; index += 1) {
        value += a[index * 4 + row] * b[column * 4 + index];
      }
      result[column * 4 + row] = value;
    }
  }
  return result;
}

function createGrid(columns, rows) {
  const vertices = new Float32Array(columns * rows * 2);
  let vertexOffset = 0;
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      vertices[vertexOffset] = column / (columns - 1);
      vertices[vertexOffset + 1] = row / (rows - 1);
      vertexOffset += 2;
    }
  }

  const indices = new Uint32Array((columns - 1) * (rows - 1) * 6);
  let indexOffset = 0;
  for (let row = 0; row < rows - 1; row += 1) {
    for (let column = 0; column < columns - 1; column += 1) {
      const topLeft = row * columns + column;
      const topRight = topLeft + 1;
      const bottomLeft = topLeft + columns;
      const bottomRight = bottomLeft + 1;
      indices.set([topLeft, bottomLeft, topRight, topRight, bottomLeft, bottomRight], indexOffset);
      indexOffset += 6;
    }
  }

  return {
    vertexBuffer: createGpuBuffer(vertices, GPUBufferUsage.VERTEX),
    indexBuffer: createGpuBuffer(indices, GPUBufferUsage.INDEX),
    indexCount: indices.length,
  };
}

function createGpuBuffer(data, usage) {
  const buffer = device.createBuffer({
    size: Math.ceil(data.byteLength / 4) * 4,
    usage,
    mappedAtCreation: true,
  });
  const constructor = data instanceof Float32Array ? Float32Array : Uint32Array;
  new constructor(buffer.getMappedRange()).set(data);
  buffer.unmap();
  return buffer;
}

function createZeroedBuffer(size, usage) {
  const buffer = device.createBuffer({ size, usage, mappedAtCreation: true });
  new Uint8Array(buffer.getMappedRange()).fill(0);
  buffer.unmap();
  return buffer;
}

// ---- Breaker anchor manager ----
// Turns the GPU block summary into wave-sheet placement. Blocks are merged
// into connected components (the domain is a wrapping torus), each component
// becomes a candidate breaker with a crest line, spread and throw direction,
// and candidates are matched to persistent anchors with hysteresis so a
// momentary lull in the numerics does not pop a wave out of existence.

function wrapDelta(from, to) {
  let delta = from - to;
  delta -= BREAKER_DOMAIN * Math.round(delta / BREAKER_DOMAIN);
  return delta;
}

function wrapDistance(ax, az, bx, bz) {
  return Math.hypot(wrapDelta(ax, bx), wrapDelta(az, bz));
}

function wrapIntoDomain(value) {
  return (((value / BREAKER_DOMAIN + 0.5) % 1) + 1) % 1 * BREAKER_DOMAIN - BREAKER_DOMAIN / 2;
}

function extractBreakerComponents(summary) {
  const blockCount = BREAKER_BLOCKS * BREAKER_BLOCKS;
  const blocks = new Array(blockCount);
  let globalMax = 0;
  for (let index = 0; index < blockCount; index += 1) {
    const offset = index * BLOCK_FLOATS;
    const block = {
      maxScore: summary[offset],
      sumScore: summary[offset + 1],
      meanX: summary[offset + 2],
      meanZ: summary[offset + 3],
      cxx: summary[offset + 4],
      czz: summary[offset + 5],
      cxz: summary[offset + 6],
      momX: summary[offset + 7],
      momZ: summary[offset + 8],
      normMax: 0,
    };
    globalMax = Math.max(globalMax, block.maxScore);
    blocks[index] = block;
  }
  if (globalMax <= 1e-6) {
    return [];
  }
  for (const block of blocks) {
    block.normMax = block.maxScore / globalMax;
  }

  // Connected components over the block lattice, wrapping in both axes.
  const componentOf = new Map();
  const components = [];
  const neighborOffsets = [[1, 0], [-1, 0], [0, 1], [0, -1]];
  for (let index = 0; index < blockCount; index += 1) {
    const block = blocks[index];
    if (componentOf.has(index) || block.normMax <= 0.3 || block.sumScore <= 1e-4) continue;
    const componentId = components.length;
    const stack = [index];
    componentOf.set(index, componentId);
    const members = [];
    while (stack.length) {
      const memberIndex = stack.pop();
      members.push(blocks[memberIndex]);
      const bx = memberIndex % BREAKER_BLOCKS;
      const bz = Math.floor(memberIndex / BREAKER_BLOCKS);
      for (const [dx, dz] of neighborOffsets) {
        const nx = (bx + dx + BREAKER_BLOCKS) % BREAKER_BLOCKS;
        const nz = (bz + dz + BREAKER_BLOCKS) % BREAKER_BLOCKS;
        const neighborIndex = nz * BREAKER_BLOCKS + nx;
        if (componentOf.has(neighborIndex)) continue;
        const neighbor = blocks[neighborIndex];
        if (neighbor.normMax > 0.3 && neighbor.sumScore > 1e-4) {
          componentOf.set(neighborIndex, componentId);
          stack.push(neighborIndex);
        }
      }
    }

    // Aggregate: weight-centred position, spread, principal axis, momentum.
    let totalWeight = 0;
    let pivot = members[0];
    for (const member of members) {
      totalWeight += member.sumScore;
      if (member.sumScore > pivot.sumScore) pivot = member;
    }
    let sumX = 0;
    let sumZ = 0;
    for (const member of members) {
      sumX += member.sumScore * wrapDelta(member.meanX, pivot.meanX);
      sumZ += member.sumScore * wrapDelta(member.meanZ, pivot.meanZ);
    }
    const centerX = wrapIntoDomain(pivot.meanX + sumX / totalWeight);
    const centerZ = wrapIntoDomain(pivot.meanZ + sumZ / totalWeight);

    let cxx = 0;
    let czz = 0;
    let cxz = 0;
    let momX = 0;
    let momZ = 0;
    let peakNorm = 0;
    for (const member of members) {
      const weight = member.sumScore / totalWeight;
      const dx = wrapDelta(member.meanX, centerX);
      const dz = wrapDelta(member.meanZ, centerZ);
      cxx += weight * (member.cxx + dx * dx);
      czz += weight * (member.czz + dz * dz);
      cxz += weight * (member.cxz + dx * dz);
      momX += member.momX;
      momZ += member.momZ;
      peakNorm = Math.max(peakNorm, member.normMax);
    }

    // Principal axis of the breaking region = the crest line. The eigenvector
    // sign is ambiguous, so it is flipped to match the flow momentum: the wave
    // throws the way it travels.
    const gap = Math.hypot(cxx - czz, 2 * cxz);
    const lambdaMax = 0.5 * (cxx + czz + gap);
    let theta = 0.5 * Math.atan2(2 * cxz, cxx - czz);
    if (Math.sin(theta) * momX - Math.cos(theta) * momZ < 0) {
      theta += Math.PI;
    }
    const extent = Math.min(42, Math.max(9, 2.5 * Math.sqrt(Math.max(lambdaMax, 0.4))));
    components.push({
      centerX,
      centerZ,
      dirX: Math.cos(theta),
      dirZ: Math.sin(theta),
      extent,
      strength: totalWeight * peakNorm,
    });
  }
  components.sort((a, b) => b.strength - a.strength);
  return components;
}

function updateBreakerAnchors(summary) {
  breakerSummary = summary;
  const components = extractBreakerComponents(summary);
  // Filter out breakers too close to the camera
  const MIN_BREAKER_DIST_FROM_CAMERA = 45.0; // world units - increased from 30
  const cameraX = cameraWorldPos[0];
  const cameraZ = cameraWorldPos[2];
  const filteredComponents = components.filter((c) => {
    const dx = c.centerX - cameraX;
    const dz = c.centerZ - cameraZ;
    return dx * dx + dz * dz >= MIN_BREAKER_DIST_FROM_CAMERA * MIN_BREAKER_DIST_FROM_CAMERA;
  });
  if (FORCE_BREAKER || DEV.hero || DEV.test) {
    // Synthesize a breaking component directly ahead of the camera so a breaker
    // is guaranteed visible (forcebreaker: verify profile; hero: deterministic capture).
    filteredComponents.push({
      centerX: cameraWorldPos[0],
      centerZ: cameraWorldPos[2] + 55,
      dirX: 1, dirZ: 0, extent: 22, strength: 1,
    });
  }
  for (const anchor of breakerAnchors) {
    anchor.claimed = false;
    anchor.component = null;
  }
  for (const component of filteredComponents) {
    let target = null;
    let bestDistance = BREAKER_MATCH_RADIUS;
    for (const anchor of breakerAnchors) {
      if (anchor.claimed) continue;
      const distance = wrapDistance(anchor.centerX, anchor.centerZ, component.centerX, component.centerZ);
      if (distance < bestDistance) {
        target = anchor;
        bestDistance = distance;
      }
    }
    if (!target) {
      target = breakerAnchors.find((anchor) => !anchor.claimed && (!anchor.active || anchor.targetEnvelope < 0.05));
    }
    if (!target) break;
    target.claimed = true;
    target.active = true;
    target.component = component;
    if (target.envelope < 0.02) {
      // Fresh spawn: snap to the candidate. The envelope then grows the wave
      // out of the sea over ~2 s, so it rears up instead of popping in.
      target.centerX = component.centerX;
      target.centerZ = component.centerZ;
      target.dirX = component.dirX;
      target.dirZ = component.dirZ;
      target.extent = component.extent;
      target.phaseOffset = Math.random() * Math.PI * 2;
    }
  }

  const dominantStrength = components.length ? components[0].strength : 1;
  for (const anchor of breakerAnchors) {
    if (!anchor.component) {
      anchor.targetEnvelope = 0;
      continue;
    }
    const component = anchor.component;
    const relative = Math.min(1, component.strength / dominantStrength);
    anchor.targetEnvelope = 1;
    anchor.targetCenterX = component.centerX;
    anchor.targetCenterZ = component.centerZ;
    anchor.targetDirX = component.dirX;
    anchor.targetDirZ = component.dirZ;
    anchor.targetExtent = component.extent;
    // Model parameters derived from how dominant this breaker is: the great
    // wave rears over a short crest stretch with a deep spiral, lesser waves
    // break faster and shallower.
    // Add seeded variation per component for less homogeneity.
    const seed = component.centerX * 12.9898 + component.centerZ * 78.233 + component.extent * 45.164;
    const rand = mulberry32(Math.abs(seed >>> 0));
    const varScale = 0.15 + 0.1 * relative; // more variation for dominant breakers
    
    anchor.targetRadius = Math.min(8.5, (1.6 + 6.0 * relative) * (TEST.active ? 1.0 : (1.0 + (rand() - 0.5) * varScale)));
    anchor.targetHeightGain = Math.min(1.2, (0.42 + 0.58 * relative) * (TEST.active ? 1.0 : (1.0 + (rand() - 0.5) * varScale)));
    anchor.curlRate = (0.61 - 0.19 * relative) * (TEST.active ? 1.0 : (1.0 + (rand() - 0.5) * 0.1));
    anchor.curlWaves = Math.min(8, Math.max(2.5, anchor.extent / 9)) * (TEST.active ? 1.0 : (1.0 + (rand() - 0.5) * 0.08));
    anchor.targetCrestPeak = (0.55 - 0.25 * relative) * (TEST.active ? 1.0 : (1.0 + (rand() - 0.5) * 0.12));
    anchor.targetCrestWidth = (0.58 - 0.06 * relative) * (TEST.active ? 1.0 : (1.0 + (rand() - 0.5) * 0.15));
    anchor.targetThetaSpan = (3.0 + 2.2 * relative) * (TEST.active ? 1.0 : (1.0 + (rand() - 0.5) * 0.1));
    anchor.taper = 0.4 * (TEST.active ? 1.0 : (1.0 + (rand() - 0.5) * 0.2));
    anchor.targetThrowGain = (0.55 + 0.55 * relative) * (TEST.active ? 1.0 : (1.0 + (rand() - 0.5) * varScale));
    anchor.targetDetailGain = Math.min(1.5, (0.7 + 0.6 * relative) * (TEST.active ? 1.0 : (1.0 + (rand() - 0.5) * 0.12)));
  }

  // Test-mode live overrides: sliders (?test) replace simulation-derived targets so
  // each art value can be tuned by eye. Null fields keep their default.
  if (TEST.active) {
    const a = breakerAnchors.find((x) => x.component) || breakerAnchors[0];
    if (a) {
      // Bypass envelope growth so the test wave appears instantly and stays stable.
      a.envelope = 1;
      a.targetEnvelope = 1;
      if (TEST.heightGain != null) a.targetHeightGain = TEST.heightGain;
      if (TEST.radius != null) a.targetRadius = TEST.radius;
      if (TEST.crestPeak != null) a.targetCrestPeak = TEST.crestPeak;
      if (TEST.crestWidth != null) a.targetCrestWidth = TEST.crestWidth;
      if (TEST.curlWaves != null) a.curlWaves = TEST.curlWaves;
      if (TEST.taper != null) a.taper = TEST.taper;
      if (TEST.thetaSpan != null) a.targetThetaSpan = TEST.thetaSpan;
      if (TEST.throwGain != null) a.targetThrowGain = TEST.throwGain;
      if (TEST.detailGain != null) a.targetDetailGain = TEST.detailGain;
      // hook/tongue scales feed the shader via debugMode.y/z (see waveProfile)
      DEV._hookScale = TEST.hookScale;
      DEV._tongueScale = TEST.tongueScale;
    }
  }
}

function smoothBreakerAnchors(deltaSeconds) {
  const positionRate = 1 - Math.exp(-deltaSeconds * 1.6);
  const shapeRate = 1 - Math.exp(-deltaSeconds * 1.1);
  const envelopeStep = deltaSeconds / BREAKER_ENVELOPE_SECONDS;
  let activity = 0;
  for (const anchor of breakerAnchors) {
    if (anchor.claimed) {
      anchor.centerX += wrapDelta(anchor.targetCenterX, anchor.centerX) * positionRate;
      anchor.centerZ += wrapDelta(anchor.targetCenterZ, anchor.centerZ) * positionRate;
      anchor.centerX = wrapIntoDomain(anchor.centerX);
      anchor.centerZ = wrapIntoDomain(anchor.centerZ);
      // Shortest-path angle blend.
      let angleFrom = Math.atan2(anchor.dirZ, anchor.dirX);
      let angleTo = Math.atan2(anchor.targetDirZ, anchor.targetDirX);
      let angleDelta = angleTo - angleFrom;
      angleDelta -= Math.round(angleDelta / (Math.PI * 2)) * Math.PI * 2;
      angleFrom += angleDelta * positionRate;
      anchor.dirX = Math.cos(angleFrom);
      anchor.dirZ = Math.sin(angleFrom);
      anchor.extent += (anchor.targetExtent - anchor.extent) * shapeRate;
      anchor.radius += (anchor.targetRadius - anchor.radius) * shapeRate;
      anchor.heightGain += (anchor.targetHeightGain - anchor.heightGain) * shapeRate;
      anchor.crestPeak += (anchor.targetCrestPeak - anchor.crestPeak) * shapeRate;
      anchor.crestWidth += (anchor.targetCrestWidth - anchor.crestWidth) * shapeRate;
      anchor.thetaSpan += (anchor.targetThetaSpan - anchor.thetaSpan) * shapeRate;
      anchor.throwGain += (anchor.targetThrowGain - anchor.throwGain) * shapeRate;
      anchor.detailGain += (anchor.targetDetailGain - anchor.detailGain) * shapeRate;
    } else {
      // Dying — envelope heads to 0 and active clears when fully gone.
    }
    anchor.envelope = Math.max(0, Math.min(1, anchor.envelope + (anchor.targetEnvelope - anchor.envelope > 0 ? envelopeStep : -envelopeStep)));
    if (anchor.envelope <= 0.0) {
      anchor.active = false;
    }
    activity += anchor.envelope;
  }
  breakerActivity = activity / breakerAnchors.length;
}


// #8 FoamFinger — generate 2-3 fingers per active breaker and pack into buffer.
const MAX_FINGERS = 16;
function writeFoamFingers() {
  const data = new Float32Array(MAX_FINGERS * 16);
  let count = 0;
  breakerAnchors.forEach((anchor) => {
    if (anchor.envelope < 0.01 || count >= MAX_FINGERS - 2) return;
    const baseSeed = anchor.phaseOffset || 0;
    const r = anchor.radius;
    // Primary finger at hook/tongue transition
    let off = count * 16;
    data[off+0] = 0.72; data[off+1] = 0.78; data[off+2] = -1; data[off+3] = 0;
    data[off+4] = 0.3; data[off+5] = 0.4; data[off+6] = r*0.6; data[off+7] = 0.012;
    data[off+8] = 0.5; data[off+9] = 0.7; data[off+10] = 0.1; data[off+11] = baseSeed;
    data[off+12] = baseSeed; data[off+13] = 1.0; data[off+14] = 0; data[off+15] = 0;
    count++;
    if (count >= MAX_FINGERS) return;
    // Secondary finger
    off = count * 16;
    data[off+0] = 0.75; data[off+1] = 0.82; data[off+2] = 0; data[off+3] = 1;
    data[off+4] = 0.5; data[off+5] = 0.6; data[off+6] = r*0.35; data[off+7] = 0.008;
    data[off+8] = 0.7; data[off+9] = 0.5; data[off+10] = 0.2; data[off+11] = baseSeed+0.3;
    data[off+12] = baseSeed+0.3; data[off+13] = 0.8; data[off+14] = 0; data[off+15] = 0;
    count++;
  });
  device.queue.writeBuffer(foamFingerBuffer, 0, data);
  return count;
}

function writeBreakerParams() {
  const data = new Float32Array(breakerAnchors.length * 40);
  breakerAnchors.forEach((anchor, index) => {
    const offset = index * 40;
    const halfSpan = anchor.extent * 0.5;
    const envelope = anchor.envelope;
    data[offset + 0] = anchor.centerX - anchor.dirX * halfSpan;
    data[offset + 1] = 0;
    data[offset + 2] = anchor.centerZ - anchor.dirZ * halfSpan;
    data[offset + 3] = 0;
    data[offset + 4] = anchor.centerX + anchor.dirX * halfSpan;
    data[offset + 5] = 0;
    data[offset + 6] = anchor.centerZ + anchor.dirZ * halfSpan;
    data[offset + 7] = anchor.radius * envelope;
    data[offset + 8] = anchor.heightGain * envelope;
    data[offset + 9] = anchor.curlRate;
    data[offset + 10] = anchor.curlWaves;
    data[offset + 11] = anchor.phaseOffset;
    data[offset + 12] = anchor.crestPeak;
    data[offset + 13] = anchor.crestWidth;
    data[offset + 14] = anchor.thetaSpan;
    data[offset + 15] = anchor.taper;
    data[offset + 16] = anchor.throwGain;
    data[offset + 17] = anchor.detailGain;
    data[offset + 18] = envelope > 0.01 ? 1 : 0;
    data[offset + 19] = 0;
    // CrestCurve: pack cubic Bezier control points so waveSample can bow the crest
    // toward the camera. p0..p3 define the 3D centreline; shape.y = forwardBow.
    if (anchor.component) {
      const ox = data[offset + 0], oz = data[offset + 2];
      const dx = data[offset + 4] - ox, dz = data[offset + 6] - oz;
      const perpX = -dz, perpZ = dx;
      const perpLen = Math.hypot(perpX, perpZ);
      const camX = cameraWorldPos[0], camZ = cameraWorldPos[2];
      const cx = ox + dx * 0.5, cz = oz + dz * 0.5;
      const toCamProj = ((camX - cx) * perpX + (camZ - cz) * perpZ) / (perpLen || 1);
      const bowAmt = Math.max(0, toCamProj) * 0.3;
      // p0 = originA, p3 = originB
      data[offset + 20] = ox; data[offset + 21] = 0; data[offset + 22] = oz; data[offset + 23] = 0;
      data[offset + 32] = ox + dx; data[offset + 33] = 0; data[offset + 34] = oz + dz; data[offset + 35] = 0;
      // p1, p2 bow toward camera
      data[offset + 24] = ox + dx * 0.33 + perpX * bowAmt; data[offset + 25] = 0; data[offset + 26] = oz + dz * 0.33 + perpZ * bowAmt; data[offset + 27] = 0;
      data[offset + 28] = ox + dx * 0.66 + perpX * bowAmt; data[offset + 29] = 0; data[offset + 30] = oz + dz * 0.66 + perpZ * bowAmt; data[offset + 31] = 0;
      // shape: peakU=0.5, forwardBow=bowAmt, bank=0, seed=phaseOffset
      data[offset + 36] = 0.5; data[offset + 37] = bowAmt; data[offset + 38] = 0; data[offset + 39] = anchor.phaseOffset || 0;
    }
  });
  device.queue.writeBuffer(breakerParamsBuffer, 0, data);
}

// Packs active breaker crest lines into the format the spray compute shader reads.
function writeSprayCrest() {
  const data = new Float32Array(WAVE_INSTANCES * 16);
  breakerAnchors.forEach((anchor, index) => {
    const offset = index * 16;
    const halfSpan = anchor.extent * 0.5;
    data[offset + 0] = anchor.centerX - anchor.dirX * halfSpan;  // startX
    data[offset + 1] = 0;                                        // startY
    data[offset + 2] = anchor.centerZ - anchor.dirZ * halfSpan;  // startZ
    data[offset + 3] = anchor.centerX + anchor.dirX * halfSpan;  // endX
    data[offset + 4] = 0;                                        // endY
    data[offset + 5] = anchor.centerZ + anchor.dirZ * halfSpan;  // endZ
    data[offset + 6] = anchor.radius * anchor.envelope;          // radius
    data[offset + 7] = anchor.heightGain * anchor.envelope;      // heightGain
    data[offset + 8] = anchor.curlRate;
    data[offset + 9] = anchor.curlWaves;
    data[offset + 10] = anchor.phaseOffset;
    data[offset + 11] = anchor.crestPeak;
    data[offset + 12] = anchor.crestWidth;
    data[offset + 13] = anchor.thetaSpan;
    data[offset + 14] = anchor.taper;
    data[offset + 15] = anchor.envelope;                         // envelope
  });
  device.queue.writeBuffer(sprayCrestBuffer, 0, data);
}

function scheduleBreakerReadback(encoder) {
  if (breakerReadbackPending) return;
  breakerReadbackPending = true;
  encoder.copyBufferToBuffer(breakerSummaryBuffer, 0, breakerStagingBuffer, 0, SUMMARY_FLOATS * 4);
}

function completeBreakerReadback() {
  breakerStagingBuffer.mapAsync(GPUMapMode.READ).then(() => {
    const copy = new Float32Array(SUMMARY_FLOATS);
    copy.set(new Float32Array(breakerStagingBuffer.getMappedRange()));
    breakerStagingBuffer.unmap();
    breakerReadbackPending = false;
    updateBreakerAnchors(copy);
  }).catch(() => {
    breakerReadbackPending = false;
  });
}

// Debug/verification hook: lets a headless run inspect what the detection
// pass found and where the anchors stand.
window.__OCEAN_BREAKERS__ = {
  summary: () => (breakerSummary ? Array.from(breakerSummary) : null),
  anchors: () => breakerAnchors.map((anchor) => ({
    active: anchor.active,
    centerX: Number(anchor.centerX.toFixed(2)),
    centerZ: Number(anchor.centerZ.toFixed(2)),
    extent: Number(anchor.extent.toFixed(2)),
    envelope: Number(anchor.envelope.toFixed(3)),
  })),
};

function mulberry32(seed) {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussianPair(random) {
  const radius = Math.sqrt(-2 * Math.log(Math.max(1e-7, random())));
  const angle = Math.PI * 2 * random();
  return [radius * Math.cos(angle), radius * Math.sin(angle)];
}

function spectrumDensity(kx, ky, settings) {
  const k = Math.hypot(kx, ky);
  if (k < 1e-5) return 0;
  const waveX = kx / k;
  const waveY = ky / k;
  const alignment = waveX * settings.windDirection[0] + waveY * settings.windDirection[1];
  const directional = alignment >= 0 ? Math.abs(alignment) ** 2.15 : 0.16 * Math.abs(alignment) ** 1.7;
  const largestWave = settings.windSpeed ** 2 / 9.81;
  const dampingLength = largestWave * 0.0015;
  const phillips = settings.amplitude
    * Math.exp(-1 / ((k * largestWave) ** 2))
    * directional
    * Math.exp(-((k * dampingLength) ** 2))
    / (k ** 4);
  const omega = Math.sqrt(9.81 * k);
  const omegaPeak = Math.PI * 2 / settings.peakPeriod;
  const sigma = omega <= omegaPeak ? 0.07 : 0.09;
  const peakShape = Math.exp(-((omega - omegaPeak) ** 2) / (2 * sigma ** 2 * omegaPeak ** 2));
  return Math.max(0, phillips * (3.3 ** peakShape));
}

function createInitialSpectrum() {
  const spectrum = new Float32Array(SIMULATION_CELLS * 4);
  const random = mulberry32(0x74c2a91d);
  const cascades = [
    {
      length: 84,
      amplitude: 3.1e-7,
      windSpeed: 14.2,
      windDirection: normalize3([0.68, 0.74, 0]).slice(0, 2),
      peakPeriod: 8.4,
    },
    {
      length: 15,
      amplitude: 5.8e-7,
      windSpeed: 8.8,
      windDirection: normalize3([-0.34, 0.94, 0]).slice(0, 2),
      peakPeriod: 3.55,
    },
  ];

  for (let y = 0; y < SIMULATION_SIZE; y += 1) {
    const frequencyY = y <= SIMULATION_SIZE / 2 ? y : y - SIMULATION_SIZE;
    for (let x = 0; x < SIMULATION_SIZE; x += 1) {
      const frequencyX = x <= SIMULATION_SIZE / 2 ? x : x - SIMULATION_SIZE;
      const target = (y * SIMULATION_SIZE + x) * 4;
      for (let cascade = 0; cascade < cascades.length; cascade += 1) {
        const settings = cascades[cascade];
        const scale = Math.PI * 2 / settings.length;
        const density = spectrumDensity(frequencyX * scale, frequencyY * scale, settings);
        const gaussian = gaussianPair(random);
        const standardDeviation = Math.sqrt(density * 0.5) * SIMULATION_CELLS;
        spectrum[target + cascade * 2] = gaussian[0] * standardDeviation;
        spectrum[target + cascade * 2 + 1] = gaussian[1] * standardDeviation;
      }
    }
  }
  return spectrum;
}

function createInitialDynamicState() {
  const state = new Float32Array(SIMULATION_CELLS * 4);
  const domainSize = 84;
  const gravity = 9.81;
  const restDepth = 1.85;
  const normalize2 = ([x, y]) => {
    const inverseLength = 1 / Math.max(1e-6, Math.hypot(x, y));
    return [x * inverseLength, y * inverseLength];
  };
  const waves = [
    // Main breaker: placed at the Hokusai composition focus so it naturally
    // breaks within the frame. Higher amplitude and tighter width give a
    // plunging crest that the breaker sheets can ride.
    { direction: normalize2([0.20, -0.98]), center: [-10.5, 14.0], amplitude: 4.20, modulation: 0.10, phase: 0.6 },
    { direction: normalize2([-0.55, 0.83]), center: [7, 3], amplitude: 1.80, modulation: 0.12, phase: 2.0 },
    { direction: normalize2([0.90, -0.44]), center: [-18, 27], amplitude: 1.20, modulation: 0.08, phase: -1.1 },
  ];

  for (let y = 0; y < SIMULATION_SIZE; y += 1) {
    for (let x = 0; x < SIMULATION_SIZE; x += 1) {
      const worldX = (x / SIMULATION_SIZE - 0.5) * domainSize;
      const worldY = (y / SIMULATION_SIZE - 0.5) * domainSize;
      let elevation = 0;
      let momentumX = 0;
      let momentumZ = 0;
      let foamSeed = 0;
      for (const wave of waves) {
        const tangent = [-wave.direction[1], wave.direction[0]];
        const relativeX = worldX - wave.center[0];
        const relativeZ = worldY - wave.center[1];
        const normalDistance = relativeX * wave.direction[0] + relativeZ * wave.direction[1];
        const alongDistance = relativeX * tangent[0] + relativeZ * tangent[1];
        const inverseWidth = Math.sqrt(3 * wave.amplitude / (4 * restDepth ** 3));
        const normalizedDistance = normalDistance * inverseWidth;
        const sech = 1 / Math.cosh(Math.min(12, Math.abs(normalizedDistance)));
        const lateralAmplitude = 0.78 + 0.22 * Math.cos(alongDistance * wave.modulation + wave.phase);
        const profile = wave.amplitude * sech * sech * lateralAmplitude;
        const phaseSpeed = Math.sqrt(gravity * (restDepth + wave.amplitude));
        const slope = 2 * profile * inverseWidth * Math.abs(Math.tanh(normalizedDistance));
        const breakingAmount = Math.max(0, Math.min(1, (slope - 0.22) / 0.42));
        const smoothBreaking = breakingAmount * breakingAmount * (3 - 2 * breakingAmount);
        elevation += profile;
        momentumX += wave.direction[0] * profile * phaseSpeed;
        momentumZ += wave.direction[1] * profile * phaseSpeed;
        foamSeed = Math.max(foamSeed, smoothBreaking * 0.78);
      }
      const index = (y * SIMULATION_SIZE + x) * 4;
      state[index] = elevation;
      state[index + 1] = momentumX;
      state[index + 2] = momentumZ;
      state[index + 3] = foamSeed;
    }
  }
  return state;
}

function createComputeParamsBuffer(mode, stage, direction) {
  const data = new ArrayBuffer(COMPUTE_PARAMS_SIZE);
  const view = new DataView(data);
  view.setUint32(8, mode, true);
  view.setUint32(12, stage, true);
  view.setUint32(16, direction, true);
  const buffer = device.createBuffer({
    size: COMPUTE_PARAMS_SIZE,
    usage: GPUBufferUsage.UNIFORM,
    mappedAtCreation: true,
  });
  new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data));
  buffer.unmap();
  return buffer;
}

function writeEvolveParams(time, delta) {
  const data = new ArrayBuffer(COMPUTE_PARAMS_SIZE);
  const view = new DataView(data);
  view.setFloat32(0, time, true);
  view.setFloat32(4, delta, true);
  view.setUint32(20, frameIndex, true);
  device.queue.writeBuffer(evolveParamsBuffer, 0, data);
}

function writeDynamicParams(time, delta, substep) {
  const data = new ArrayBuffer(COMPUTE_PARAMS_SIZE);
  const view = new DataView(data);
  view.setFloat32(0, time, true);
  view.setFloat32(4, delta, true);
  view.setUint32(8, frameIndex, true);
  view.setUint32(12, substep, true);
  view.setFloat32(16, smoothPointer[0], true);
  view.setFloat32(20, smoothPointer[1], true);
  view.setFloat32(24, interactionEnergy, true);
  view.setFloat32(28, moving ? 1 : 0, true);
  device.queue.writeBuffer(dynamicParamsBuffer, 0, data);
}

async function checkedModule(label, code) {
  const module = device.createShaderModule({ label, code });
  const compilation = await module.getCompilationInfo();
  const errors = compilation.messages.filter((message) => message.type === 'error');
  if (errors.length) {
    throw new Error(`${label}\n${errors.map((error) => `line ${error.lineNum}: ${error.message}`).join('\n')}`);
  }
  return module;
}

function announce(message) {
  status.textContent = message;
}

function setMoving(next) {
  if (moving === next) return;
  moving = next;
  if (!moving) pausedElapsed = performance.now() - startTime;
  else startTime = performance.now() - pausedElapsed;
  motionToggle.setAttribute('aria-pressed', String(!moving));
  motionToggle.textContent = moving ? 'Pause ocean motion' : 'Resume ocean motion';
  announce(moving ? 'Ocean motion resumed' : 'Ocean motion paused');
}

function createFrameTextures() {
  sceneTexture?.destroy();
  multisampleTexture?.destroy();
  depthTexture?.destroy();
  sceneTexture = device.createTexture({
    label: 'HDR scene resolve',
    size: [canvas.width, canvas.height],
    format: HDR_FORMAT,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });
  multisampleTexture = device.createTexture({
    label: 'HDR multisample target',
    size: [canvas.width, canvas.height],
    sampleCount: SAMPLE_COUNT,
    format: HDR_FORMAT,
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });
  depthTexture = device.createTexture({
    label: 'Multisample depth',
    size: [canvas.width, canvas.height],
    sampleCount: SAMPLE_COUNT,
    format: 'depth24plus',
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });
  postBindGroup = device.createBindGroup({
    layout: postPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: sceneSampler },
      { binding: 2, resource: sceneTexture.createView() },
    ],
  });
}

function resize() {
  if (!device || !context || !postPipeline) return;
  const deviceRatio = Math.min(window.devicePixelRatio || 1, window.innerWidth < 720 ? 1.16 : 1.42);
  const pixelRatio = deviceRatio * renderScale;
  const width = Math.max(1, Math.floor(canvas.clientWidth * pixelRatio));
  const height = Math.max(1, Math.floor(canvas.clientHeight * pixelRatio));
  if (canvas.width === width && canvas.height === height) return;
  canvas.width = width;
  canvas.height = height;
  context.configure({ device, format, alphaMode: 'opaque' });
  createFrameTextures();
}

function trackPointer(event) {
  const bounds = canvas.getBoundingClientRect();
  pointer = [
    Math.min(1, Math.max(0, (event.clientX - bounds.left) / bounds.width)),
    Math.min(1, Math.max(0, (event.clientY - bounds.top) / bounds.height)),
  ];
  interactionEnergy = Math.min(1, interactionEnergy + 0.2);
}

function adaptQuality(delta) {
  averageFrameTime = averageFrameTime * 0.96 + delta * 0.04;
  adaptationElapsed += delta;
  if (adaptationElapsed < 2800) return;
  adaptationElapsed = 0;
  // Only step DOWN when the frame budget is blown. We start at the target scale
  // (0.96 desktop / 0.84 mobile), so ramping UP would only reconfigure the
  // swapchain and read as a resize jump on load — there is no benefit.
  if (averageFrameTime > 30 && renderScale > 0.56) {
    renderScale = Math.max(0.56, renderScale - 0.07);
    resize();
  }
}

function createSpectralPasses(layout) {
  const passes = [];
  let sourceIndex = 0;
  const appendPass = (mode, stage, direction) => {
    const destinationIndex = 1 - sourceIndex;
    const paramsBuffer = createComputeParamsBuffer(mode, stage, direction);
    const bindGroup = device.createBindGroup({
      layout,
      entries: [
        { binding: 0, resource: { buffer: paramsBuffer } },
        { binding: 1, resource: { buffer: initialSpectrumBuffer } },
        { binding: 2, resource: { buffer: fftBuffers[sourceIndex] } },
        { binding: 3, resource: { buffer: fftBuffers[destinationIndex] } },
      ],
    });
    passes.push({ bindGroup, paramsBuffer });
    sourceIndex = destinationIndex;
  };

  appendPass(0, 0, 0);
  for (let stage = 0; stage < FFT_STAGES; stage += 1) appendPass(1, stage, 0);
  appendPass(0, 0, 1);
  for (let stage = 0; stage < FFT_STAGES; stage += 1) appendPass(1, stage, 1);
  return { passes, finalBufferIndex: sourceIndex };
}

async function initialize() {
  if (!navigator.gpu) throw new Error('WebGPU is unavailable');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('A WebGPU adapter is unavailable');
  device = await adapter.requestDevice();
  device.addEventListener('uncapturederror', (event) => {
    console.error(`WebGPU validation error: ${event.error.message}`);
  });
  device.lost.then(() => {
    notice.hidden = false;
    notice.querySelector('h1').textContent = 'The graphics device was disconnected.';
  });

  context = canvas.getContext('webgpu');
  format = navigator.gpu.getPreferredCanvasFormat();

  createCoreBuffers();
  createBreakerBuffers();
  createGrids();

  // Compile every shader module up front; checkedModule throws on WGSL errors.
  const [spectralModule, dynamicModule, resolveModule, breakersModule, sceneModule, postModule, sprayModule] = await Promise.all([
    checkedModule('Spectral ocean compute', spectralShader),
    checkedModule('Dynamic water compute', dynamicShader),
    checkedModule('Ocean field resolve', resolveShader),
    checkedModule('Breaker detection', breakersShader),
    checkedModule('Ocean scene', `${sceneShader}\n${waveGeometryShader}`),
    checkedModule('HDR post process', postShader),
    checkedModule('Spray particles', sprayShader),
  ]);

  createComputeAndScenePipelines({
    spectralModule, dynamicModule, resolveModule, breakersModule, sceneModule, postModule,
  });
  createSpraySystem(sprayModule);

  resize();
  experience.classList.add('is-ready');
  // #10: parse deterministic hero/capture URL flags before the render loop so
  // the composition is reproducible from the first frame.
  parseDevParams();
  // Reset timers right before starting render loop so first frame has elapsed ≈ 0
  startTime = performance.now();
  previousFrame = performance.now();
  if (DEV.test) buildTestPanel();
  frameRequest = requestAnimationFrame(draw);
}

// Core simulation buffers: uniforms + the double-buffered spectral / dynamic /
// ocean state used by the compute chain.
function createCoreBuffers() {
  uniformBuffer = device.createBuffer({
    size: UNIFORM_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  evolveParamsBuffer = device.createBuffer({
    size: COMPUTE_PARAMS_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  dynamicParamsBuffer = device.createBuffer({
    size: COMPUTE_PARAMS_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  resolveParamsBuffer = device.createBuffer({
    size: RESOLVE_PARAMS_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const fftBufferSize = SIMULATION_CELLS * 4 * Float32Array.BYTES_PER_ELEMENT;
  initialSpectrumBuffer = createGpuBuffer(createInitialSpectrum(), GPUBufferUsage.STORAGE);
  fftBuffers = [
    createZeroedBuffer(fftBufferSize, GPUBufferUsage.STORAGE),
    createZeroedBuffer(fftBufferSize, GPUBufferUsage.STORAGE),
  ];
  const initialDynamicState = createInitialDynamicState();
  dynamicBuffers = [
    createGpuBuffer(initialDynamicState, GPUBufferUsage.STORAGE),
    createGpuBuffer(initialDynamicState, GPUBufferUsage.STORAGE),
  ];
  const oceanBufferSize = SIMULATION_CELLS * 8 * Float32Array.BYTES_PER_ELEMENT;
  oceanBuffers = [
    createZeroedBuffer(oceanBufferSize, GPUBufferUsage.STORAGE),
    createZeroedBuffer(oceanBufferSize, GPUBufferUsage.STORAGE),
  ];
}

// Breaker scoring + CPU anchor readback buffers.
function createBreakerBuffers() {
  breakerConfigBuffer = device.createBuffer({
    size: 48,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  breakerScoreBuffer = createZeroedBuffer(SIMULATION_CELLS * 4, GPUBufferUsage.STORAGE);
  breakerSummaryBuffer = createZeroedBuffer(SUMMARY_FLOATS * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  breakerStagingBuffer = device.createBuffer({
    size: SUMMARY_FLOATS * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  breakerParamsBuffer = createZeroedBuffer(breakerAnchors.length * 40 * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  foamFingerBuffer = createZeroedBuffer(16 * 16 * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST); // 16 fingers max, 16 floats each
}

function createGrids() {
  oceanGrid = createGrid(OCEAN_COLUMNS, OCEAN_ROWS);
  waveGrid = createGrid(WAVE_COLUMNS, WAVE_ROWS);
  clawGrid = createGrid(CLAW_COLUMNS, CLAW_ROWS);
}

function dispatchCompute(encoder, pipeline, bindGroup, x, y = 1) {
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(x, y);
  pass.end();
}

function setGrid(pass, grid) {
  pass.setVertexBuffer(0, grid.vertexBuffer);
  pass.setIndexBuffer(grid.indexBuffer, 'uint32');
}

// Builds every GPU pipeline the frame loop needs: the compute simulation chain
// (spectral → fft → dynamic → resolve → breaker score/reduce) and the scene
// render passes (background, ocean, wave sheets, claws, post).
function createComputeAndScenePipelines(modules) {
  const { spectralModule, dynamicModule, resolveModule, breakersModule, sceneModule, postModule } = modules;

  // ---- Compute: spectral ocean + FFT ----
  const spectralLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  });
  const spectralPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [spectralLayout] });
  spectralPipeline = device.createComputePipeline({
    layout: spectralPipelineLayout,
    compute: { module: spectralModule, entryPoint: 'evolveSpectrum' },
  });
  fftPipeline = device.createComputePipeline({
    layout: spectralPipelineLayout,
    compute: { module: spectralModule, entryPoint: 'fftPass' },
  });
  evolveBindGroup = device.createBindGroup({
    layout: spectralLayout,
    entries: [
      { binding: 0, resource: { buffer: evolveParamsBuffer } },
      { binding: 1, resource: { buffer: initialSpectrumBuffer } },
      { binding: 2, resource: { buffer: fftBuffers[1] } },
      { binding: 3, resource: { buffer: fftBuffers[0] } },
    ],
  });
  const spectralPassResult = createSpectralPasses(spectralLayout);
  fftPasses = spectralPassResult.passes;

  // ---- Compute: dynamic shallow-water ----
  const dynamicLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  });
  dynamicPipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [dynamicLayout] }),
    compute: { module: dynamicModule, entryPoint: 'evolveDynamicWater' },
  });
  dynamicBindGroups = [0, 1].map((sourceIndex) => device.createBindGroup({
    layout: dynamicLayout,
    entries: [
      { binding: 0, resource: { buffer: dynamicParamsBuffer } },
      { binding: 1, resource: { buffer: dynamicBuffers[sourceIndex] } },
      { binding: 2, resource: { buffer: dynamicBuffers[1 - sourceIndex] } },
    ],
  }));

  // ---- Compute: ocean resolve ----
  const resolveLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    ],
  });
  resolvePipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [resolveLayout] }),
    compute: { module: resolveModule, entryPoint: 'resolveOcean' },
  });
  resolveBindGroups = [0, 1].map((previousIndex) => [0, 1].map((dynamicIndex) => device.createBindGroup({
      layout: resolveLayout,
      entries: [
        { binding: 0, resource: { buffer: resolveParamsBuffer } },
        { binding: 1, resource: { buffer: fftBuffers[spectralPassResult.finalBufferIndex] } },
        { binding: 2, resource: { buffer: oceanBuffers[previousIndex] } },
        { binding: 3, resource: { buffer: oceanBuffers[1 - previousIndex] } },
        { binding: 4, resource: { buffer: dynamicBuffers[dynamicIndex] } },
      ],
    })));

  // ---- Compute: breaker detection ----
  const breakerLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  });
  const breakerPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [breakerLayout] });
  scoreBreakersPipeline = device.createComputePipeline({
    layout: breakerPipelineLayout,
    compute: { module: breakersModule, entryPoint: 'scoreBreakers' },
  });
  reduceBreakersPipeline = device.createComputePipeline({
    layout: breakerPipelineLayout,
    compute: { module: breakersModule, entryPoint: 'reduceBreakers' },
  });
  breakerBindGroups = [0, 1].map((oceanIndex) => [0, 1].map((dynamicIndex) => device.createBindGroup({
    layout: breakerLayout,
    entries: [
      { binding: 0, resource: { buffer: breakerConfigBuffer } },
      { binding: 1, resource: { buffer: oceanBuffers[oceanIndex] } },
      { binding: 2, resource: { buffer: dynamicBuffers[dynamicIndex] } },
      { binding: 3, resource: { buffer: breakerScoreBuffer } },
      { binding: 4, resource: { buffer: breakerSummaryBuffer } },
    ],
  })));

  // ---- Scene render: surface bind groups ----
  const surfaceLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
    ],
  });
  const scenePipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [surfaceLayout] });
  surfaceBindGroups = oceanBuffers.map((buffer) => device.createBindGroup({
    layout: surfaceLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer } },
    ],
  }));
  // The wave sheets additionally read their placement data, so their bind
  // groups carry one more entry than the sea and background.
  const breakerSurfaceLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
    ],
  });
  waveBindGroups = oceanBuffers.map((buffer) => device.createBindGroup({
    layout: breakerSurfaceLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer } },
      { binding: 2, resource: { buffer: breakerParamsBuffer } },
    ],
  }));

  const colorTarget = { format: HDR_FORMAT };
  const depthStencil = {
    format: 'depth24plus',
    depthWriteEnabled: true,
    depthCompare: 'less',
  };
  backgroundPipeline = device.createRenderPipeline({
    layout: scenePipelineLayout,
    vertex: { module: sceneModule, entryPoint: 'backgroundVertex' },
    fragment: { module: sceneModule, entryPoint: 'backgroundFragment', targets: [colorTarget] },
    primitive: { topology: 'triangle-list' },
    multisample: { count: SAMPLE_COUNT },
    depthStencil: { ...depthStencil, depthWriteEnabled: false, depthCompare: 'always' },
  });
  const surfaceVertex = {
    module: sceneModule,
    buffers: [{
      arrayStride: 8,
      attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x2' }],
    }],
  };
  const surfaceFragment = { module: sceneModule, entryPoint: 'surfaceFragment', targets: [colorTarget] };
  const surfaceBase = {
    layout: scenePipelineLayout,
    fragment: surfaceFragment,
    primitive: { topology: 'triangle-list', cullMode: 'none' },
    multisample: { count: SAMPLE_COUNT },
    depthStencil,
  };
  oceanPipeline = device.createRenderPipeline({
    ...surfaceBase,
    vertex: { ...surfaceVertex, entryPoint: 'oceanVertex' },
  });
  const breakerSurfaceBase = {
    ...surfaceBase,
    layout: device.createPipelineLayout({ bindGroupLayouts: [breakerSurfaceLayout] }),
  };
  wavePipeline = device.createRenderPipeline({
    ...breakerSurfaceBase,
    vertex: { ...surfaceVertex, entryPoint: 'waveVertex' },
  });
  foamPipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: sceneModule, entryPoint: 'foamVertex', buffers: [] },
    fragment: { module: sceneModule, entryPoint: 'surfaceFragment', targets: [{ format, blend: { color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' } } }] },
    primitive: { topology: 'triangle-strip', cullMode: 'none' },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
  });

  clawPipeline = device.createRenderPipeline({
    ...breakerSurfaceBase,
    vertex: { ...surfaceVertex, entryPoint: 'clawVertex' },
  });

  postPipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: postModule, entryPoint: 'postVertex' },
    fragment: { module: postModule, entryPoint: 'postFragment', targets: [{ format }] },
    primitive: { topology: 'triangle-list' },
  });
  sceneSampler = device.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  });
}

// Spray particle system: compute spawn/update pipelines + point-sprite render.
// Render WGSL is inline (SPRAY_VERTEX_WGSL / SPRAY_FRAGMENT_WGSL) because it
// shares the Particle struct layout with shaders/spray.wgsl.
function createSpraySystem(sprayModule) {
  const sprayLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  });
  const sprayPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [sprayLayout] });
  spraySpawnPipeline = device.createComputePipeline({
    layout: sprayPipelineLayout,
    compute: { module: sprayModule, entryPoint: 'spawnSpray' },
  });
  sprayUpdatePipeline = device.createComputePipeline({
    layout: sprayPipelineLayout,
    compute: { module: sprayModule, entryPoint: 'updateSpray' },
  });

  // Spray parameters (elapsed, delta, spawnRate, instanceCount) + padding.
  sprayParamsBuffer = device.createBuffer({
    size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  // Crest data: one BreakerCrest (16 floats) per wave instance.
  sprayCrestBuffer = device.createBuffer({
    size: WAVE_INSTANCES * 16 * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  // Particle storage: MAX_PARTICLES * 64 bytes (4 x vec4f).
  sprayBuffer = device.createBuffer({
    size: MAX_PARTICLES * 64,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  sprayBindGroup = device.createBindGroup({
    layout: sprayLayout,
    entries: [
      { binding: 0, resource: { buffer: sprayParamsBuffer } },
      { binding: 1, resource: { buffer: sprayCrestBuffer } },
      { binding: 2, resource: { buffer: sprayBuffer } },
    ],
  });

  // Spray render pipeline: instanced billboard quads, 4x MSAA to match scene pass.
  const sprayVertexModule = device.createShaderModule({ code: SPRAY_VERTEX_WGSL });
  const sprayFragmentModule = device.createShaderModule({ code: SPRAY_FRAGMENT_WGSL });
  sprayRenderPipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: sprayVertexModule, entryPoint: 'main' },
    fragment: {
      module: sprayFragmentModule,
      entryPoint: 'main',
      targets: [{ format: HDR_FORMAT, blend: {
        color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
        alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
      }}],
    },
    primitive: { topology: 'triangle-list' },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'less' },
    multisample: { count: SAMPLE_COUNT },
  });
  sprayRenderBindGroup = device.createBindGroup({
    layout: sprayRenderPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: sprayBuffer } },
    ],
  });
}

function draw(now) {
  const deltaMilliseconds = Math.min(50, now - previousFrame);
  previousFrame = now;
  adaptQuality(deltaMilliseconds);

  if (moving) {
    const smoothing = 1 - Math.pow(0.86, deltaMilliseconds / 16.7);
    smoothPointer[0] += (pointer[0] - smoothPointer[0]) * smoothing;
    smoothPointer[1] += (pointer[1] - smoothPointer[1]) * smoothing;
    interactionEnergy *= Math.pow(0.967, deltaMilliseconds / 16.7);
  }

  const elapsed = (moving ? now - startTime : pausedElapsed) / 1000;
  // #10: resolve the hero phase. Explicit ?phase= pins it for reproducible
  // captures; otherwise it advances with the cycle so the composition lives.
  const presetCycle = (HERO_PRESETS[DEV.preset] || {}).cycleSeconds || 18;
  heroPhase = DEV.phase != null
    ? DEV.phase
    : (elapsed / presetCycle) % 1;
  const motionSpeed = prefersReducedMotion ? 0.16 : 1;
  const deltaSeconds = moving ? deltaMilliseconds / 1000 : 0;
  const dynamicDelta = Math.min(deltaSeconds, 0.034) * motionSpeed;
  dynamicTime += dynamicDelta;
  // Hokusai frames the wave from down in the trough, close enough that the crest
  // takes the top of the plate and the sky survives only in the upper right.
  const camera = [
    3.0 + (smoothPointer[0] - 0.5) * 0.30,
    2.0 + (0.5 - smoothPointer[1]) * 0.22 + Math.sin(elapsed * 0.09 * motionSpeed) * 0.075,
    -32.0,
  ];
  cameraWorldPos[0] = camera[0];
  cameraWorldPos[1] = camera[1];
  cameraWorldPos[2] = camera[2];
  const target = [
    -6.0 + (smoothPointer[0] - 0.5) * 0.18,
    8.5 + (0.5 - smoothPointer[1]) * 0.14,
    12.0,
  ];
  const aspect = canvas.width / canvas.height;
  const fovRadians = 50 * Math.PI / 180;
  const projection = perspectiveLeftHanded(fovRadians, aspect, 0.35, 260);
  const view = lookAtLeftHanded(camera, target);
  const viewProjection = multiplyMatrices(projection, view);
  // Camera basis for the background rays. Same construction lookAtLeftHanded
  // uses, kept out here because the sky needs the vectors themselves.
  const forward = normalize3(subtract3(target, camera));
  const camRight = normalize3(cross3([0, 1, 0], forward));
  const camUp = cross3(forward, camRight);
  const tanHalfY = Math.tan(fovRadians / 2);

  // Update Web Audio listener position (camera)
  setListenerPosition(
    camera[0], camera[1], camera[2],
    forward[0], forward[1], forward[2],
    camUp[0], camUp[1], camUp[2]
  );

  // Update spatial sound to follow the main active breaker
  const activeAnchor = breakerAnchors.find(a => a.active && a.envelope > 0.1);
  if (activeAnchor) {
    // Breaker position: center of the crest line, height from wave profile
    const breakerX = activeAnchor.centerX;
    const breakerY = activeAnchor.heightGain * activeAnchor.radius * 0.5; // approximate crest height
    const breakerZ = activeAnchor.centerZ;
    updateBreakerPosition(breakerX, breakerY, breakerZ, activeAnchor.dirX, activeAnchor.dirZ);
  }
  const pixelRatio = canvas.clientWidth ? canvas.width / canvas.clientWidth : 1;
  const sunDirection = normalize3([0.31, 0.19, 0.93]);
  const uniforms = new Float32Array(UNIFORM_FLOATS);
  uniforms.set(viewProjection, 0);
  uniforms.set([camera[0], camera[1], camera[2], elapsed], 16);
  uniforms.set([canvas.width, canvas.height, motionSpeed, pixelRatio], 20);
  uniforms.set([smoothPointer[0], smoothPointer[1], interactionEnergy, renderScale], 24);
  uniforms.set([deltaSeconds, 1.08, renderScale, frameIndex], 28);
  uniforms.set([sunDirection[0], sunDirection[1], sunDirection[2], 0], 32);
  uniforms.set([camRight[0], camRight[1], camRight[2], tanHalfY * aspect], 36);
  uniforms.set([camUp[0], camUp[1], camUp[2], tanHalfY], 40);
  uniforms.set([forward[0], forward[1], forward[2], 0], 44);
  // debugMode.x: 0=off, 1=regions tint. y/z: live hook/tongue scale (test sliders).
  uniforms.set([DEV.debug, TEST.hookScale, TEST.tongueScale, 0], 48);
  // Mountain removed: background is only sky + clouds now.
  device.queue.writeBuffer(uniformBuffer, 0, uniforms);
  writeEvolveParams(elapsed * motionSpeed, deltaSeconds);
  writeDynamicParams(dynamicTime, dynamicDelta / DYNAMIC_SUBSTEPS, 0);
  device.queue.writeBuffer(resolveParamsBuffer, 0, new Float32Array([dynamicTime, dynamicDelta, 0.45, frameIndex]));
  // Breaker detection configuration. The focus is the composition prior: the
  // print wants its breaker mid-frame, left of the mountain, and the prior is
  // where that preference lives — explicit, not disguised as physics.
  const breakerConfig = new Float32Array([
    2.0, 16.0, // focus (world XZ) - moved further right, further forward
    16.0, 14.0, // ellipse radii - shrunk from 22,18 to cut left side more
    camera[0], camera[2],
    6.0,        // minimum distance from camera
    0.02,       // score floor
    84.0,       // domainSize
    elapsed,    // time
    0.0, 0.0,   // pad
  ]);
  device.queue.writeBuffer(breakerConfigBuffer, 0, breakerConfig);

  const encoder = device.createCommandEncoder();
  const groups = Math.ceil(SIMULATION_SIZE / 8);
  dispatchCompute(encoder, spectralPipeline, evolveBindGroup, groups, groups);
  for (const spectralPass of fftPasses) {
    dispatchCompute(encoder, fftPipeline, spectralPass.bindGroup, groups, groups);
  }
  for (let substep = 0; substep < DYNAMIC_SUBSTEPS; substep += 1) {
    dispatchCompute(encoder, dynamicPipeline, dynamicBindGroups[dynamicBufferIndex], groups, groups);
    dynamicBufferIndex = 1 - dynamicBufferIndex;
  }
  dispatchCompute(encoder, resolvePipeline, resolveBindGroups[oceanBufferIndex][dynamicBufferIndex], groups, groups);
  oceanBufferIndex = 1 - oceanBufferIndex;

  // Score the field for breaking, collapse onto the block lattice, and every
  // so often copy the lattice back for the CPU anchor manager.
  dispatchCompute(encoder, scoreBreakersPipeline, breakerBindGroups[oceanBufferIndex][dynamicBufferIndex], groups, groups);
  dispatchCompute(encoder, reduceBreakersPipeline, breakerBindGroups[oceanBufferIndex][dynamicBufferIndex], BREAKER_BLOCKS, BREAKER_BLOCKS);
  smoothBreakerAnchors(deltaSeconds);
  writeBreakerParams();
  breakerFrameCounter = (breakerFrameCounter + 1) % READBACK_INTERVAL;
  if (breakerFrameCounter === 0 && moving) {
    scheduleBreakerReadback(encoder);
  }

  // ===== Spray particle simulation =====
  writeSprayCrest();
  writeFoamFingers();
  device.queue.writeBuffer(sprayParamsBuffer, 0, new Float32Array([
    elapsed,
    deltaSeconds,
    moving ? 1.0 : 0.0,    // spawnRate
    WAVE_INSTANCES,
    0, 0, 0, 0,
  ]));
  dispatchCompute(encoder, spraySpawnPipeline, sprayBindGroup, Math.ceil(WAVE_INSTANCES / SPRAY_WORKGROUP_SIZE));
  dispatchCompute(encoder, sprayUpdatePipeline, sprayBindGroup, Math.ceil(MAX_PARTICLES / SPRAY_WORKGROUP_SIZE));

  const scenePass = encoder.beginRenderPass({
    colorAttachments: [{
      view: multisampleTexture.createView(),
      resolveTarget: sceneTexture.createView(),
      clearValue: { r: 0.001, g: 0.008, b: 0.032, a: 1 },
      loadOp: 'clear',
      storeOp: 'discard',
    }],
    depthStencilAttachment: {
      view: depthTexture.createView(),
      depthClearValue: 1,
      depthLoadOp: 'clear',
      depthStoreOp: 'discard',
    },
  });
  scenePass.setBindGroup(0, surfaceBindGroups[oceanBufferIndex]);
  scenePass.setPipeline(backgroundPipeline);
  scenePass.draw(3);
  setGrid(scenePass, oceanGrid);
  scenePass.setPipeline(oceanPipeline);
  scenePass.drawIndexed(oceanGrid.indexCount);
  scenePass.setBindGroup(0, waveBindGroups[oceanBufferIndex]);
  setGrid(scenePass, waveGrid);
  scenePass.setPipeline(wavePipeline);
  const waveInstanceCount = Math.min(WAVE_INSTANCES, breakerAnchors.length);
  if (waveInstanceCount > 0) {
    scenePass.drawIndexed(waveGrid.indexCount, waveInstanceCount);
    setGrid(scenePass, clawGrid);
    scenePass.setPipeline(clawPipeline);
    scenePass.drawIndexed(clawGrid.indexCount, waveInstanceCount);
  }
  // Spray particles drawn on top of the ocean, depth-tested against it.
  scenePass.setBindGroup(0, sprayRenderBindGroup);
  scenePass.setPipeline(sprayRenderPipeline);
  scenePass.draw(6, MAX_PARTICLES);
  scenePass.end();

  const postPass = encoder.beginRenderPass({
    colorAttachments: [{
      view: context.getCurrentTexture().createView(),
      clearValue: { r: 0, g: 0, b: 0, a: 1 },
      loadOp: 'clear',
      storeOp: 'store',
    }],
  });
  postPass.setPipeline(postPipeline);
  postPass.setBindGroup(0, postBindGroup);
  postPass.draw(3);
  postPass.end();

  device.queue.submit([encoder.finish()]);
  if (breakerReadbackPending) {
    completeBreakerReadback();
  }
  frameIndex = (frameIndex + 1) % 10_000_000;
  frameRequest = requestAnimationFrame(draw);
}

// pointermove removed: no passive following
canvas.addEventListener('pointerdown', (event) => {
  trackPointer(event);
  interactionEnergy = 1;
  canvas.focus({ preventScroll: true });
}, { passive: true });
canvas.addEventListener('pointerleave', () => {
  pointer = [0.5, 0.5];
});
motionToggle.addEventListener('click', () => setMoving(!moving));
soundToggle.addEventListener('click', toggleOceanSound);
window.addEventListener('keydown', (event) => {
  if (event.code !== 'Space' || event.repeat) return;
  event.preventDefault();
  setMoving(!moving);
});
window.addEventListener('resize', resize);
document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    cancelAnimationFrame(frameRequest);
    frameRequest = null;
    return;
  }
  if (frameRequest !== null) return;
  previousFrame = performance.now();
  frameRequest = requestAnimationFrame(draw);
});

initialize().catch((error) => {
  console.error(error);
  notice.hidden = false;
  notice.querySelector('p').textContent = 'This browser could not start the live ocean simulation.';
});
