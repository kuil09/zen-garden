import spectralShader from './shaders/spectral.wgsl?raw';
import dynamicShader from './shaders/dynamic.wgsl?raw';
import resolveShader from './shaders/resolve.wgsl?raw';
import breakersShader from './shaders/breakers.wgsl?raw';
import sceneShader from './shaders/scene.wgsl?raw';
import waveGeometryShader from './shaders/wave-geometry.wgsl?raw';
import postShader from './shaders/post.wgsl?raw';
import { toggleOceanSound } from './ocean-sound.js';

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
const UNIFORM_FLOATS = 48;
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

let device;
let context;
let format;
let uniformBuffer;
let evolveParamsBuffer;
let dynamicParamsBuffer;
let resolveParamsBuffer;
let initialSpectrumBuffer;
let fftBuffers;
let dynamicBuffers;
let oceanBuffers;
let spectralPipeline;
let fftPipeline;
let dynamicPipeline;
let resolvePipeline;
let scoreBreakersPipeline;
let reduceBreakersPipeline;
let backgroundPipeline;
let oceanPipeline;
let postPipeline;
let evolveBindGroup;
let fftPasses;
let dynamicBindGroups;
let resolveBindGroups;
let breakerConfigBuffer;
let breakerScoreBuffer;
let breakerSummaryBuffer;
let breakerStagingBuffer;
let breakerParamsBuffer;
let breakerBindGroups;
let surfaceBindGroups;
let waveBindGroups;
let postBindGroup;
let oceanGrid;
// Camera world position for breaker spawn filtering
let cameraWorldPos = [0.0, 0.0, 0.0];
let waveGrid;
let wavePipeline;
let clawGrid;
let clawPipeline;
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
let renderScale = window.innerWidth < 720 ? 0.66 : 0.84;
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
  const MIN_BREAKER_DIST_FROM_CAMERA = 30.0; // world units - increased from 18
  const cameraX = cameraWorldPos[0];
  const cameraZ = cameraWorldPos[2];
  const filteredComponents = components.filter((c) => {
    const dx = c.centerX - cameraX;
    const dz = c.centerZ - cameraZ;
    return dx * dx + dz * dz >= MIN_BREAKER_DIST_FROM_CAMERA * MIN_BREAKER_DIST_FROM_CAMERA;
  });
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
    
    anchor.targetRadius = Math.min(8.5, (1.6 + 6.0 * relative) * (1.0 + (rand() - 0.5) * varScale));
    anchor.targetHeightGain = Math.min(1.2, (0.42 + 0.58 * relative) * (1.0 + (rand() - 0.5) * varScale));
    anchor.curlRate = (0.61 - 0.19 * relative) * (1.0 + (rand() - 0.5) * 0.1);
    anchor.curlWaves = Math.min(8, Math.max(2.5, anchor.extent / 9)) * (1.0 + (rand() - 0.5) * 0.08);
    anchor.targetCrestPeak = (0.55 - 0.25 * relative) * (1.0 + (rand() - 0.5) * 0.12);
    anchor.targetCrestWidth = (0.58 - 0.06 * relative) * (1.0 + (rand() - 0.5) * 0.15);
    anchor.targetThetaSpan = (3.0 + 2.2 * relative) * (1.0 + (rand() - 0.5) * 0.1);
    anchor.taper = 0.4 * (1.0 + (rand() - 0.5) * 0.2);
    anchor.targetThrowGain = (0.55 + 0.55 * relative) * (1.0 + (rand() - 0.5) * varScale);
    anchor.targetDetailGain = Math.min(1.5, (0.7 + 0.6 * relative) * (1.0 + (rand() - 0.5) * 0.12));
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

function writeBreakerParams() {
  const data = new Float32Array(breakerAnchors.length * 20);
  breakerAnchors.forEach((anchor, index) => {
    const offset = index * 20;
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
  });
  device.queue.writeBuffer(breakerParamsBuffer, 0, data);
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
  if (averageFrameTime > 30 && renderScale > 0.56) {
    renderScale = Math.max(0.56, renderScale - 0.07);
    resize();
  } else if (averageFrameTime < 18.2 && renderScale < 0.96) {
    renderScale = Math.min(0.96, renderScale + 0.035);
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

  const initialSpectrum = createInitialSpectrum();
  initialSpectrumBuffer = createGpuBuffer(initialSpectrum, GPUBufferUsage.STORAGE);
  const fftBufferSize = SIMULATION_CELLS * 4 * Float32Array.BYTES_PER_ELEMENT;
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
  breakerParamsBuffer = createZeroedBuffer(breakerAnchors.length * 20 * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);

  oceanGrid = createGrid(OCEAN_COLUMNS, OCEAN_ROWS);
  waveGrid = createGrid(WAVE_COLUMNS, WAVE_ROWS);
  clawGrid = createGrid(CLAW_COLUMNS, CLAW_ROWS);

  const [spectralModule, dynamicModule, resolveModule, breakersModule, sceneModule, postModule] = await Promise.all([
    checkedModule('Spectral ocean compute', spectralShader),
    checkedModule('Dynamic water compute', dynamicShader),
    checkedModule('Ocean field resolve', resolveShader),
    checkedModule('Breaker detection', breakersShader),
    checkedModule('Ocean scene', `${sceneShader}\n${waveGeometryShader}`),
        checkedModule('HDR post process', postShader),
  ]);

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

  // Breaker detection reads the resolved ocean + dynamic field and writes the
  // block summary the CPU reads back.
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

  resize();
  experience.classList.add('is-ready');
  frameRequest = requestAnimationFrame(draw);
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

function draw(now) {
  resize();
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
  // Mountain removed: background is only sky + clouds now.
  device.queue.writeBuffer(uniformBuffer, 0, uniforms);
  writeEvolveParams(elapsed * motionSpeed, deltaSeconds);
  writeDynamicParams(dynamicTime, dynamicDelta / DYNAMIC_SUBSTEPS, 0);
  device.queue.writeBuffer(resolveParamsBuffer, 0, new Float32Array([dynamicTime, dynamicDelta, 0.45, frameIndex]));
  // Breaker detection configuration. The focus is the composition prior: the
  // print wants its breaker mid-frame, left of the mountain, and the prior is
  // where that preference lives — explicit, not disguised as physics.
  const breakerConfig = new Float32Array([
    -3.0, 14.0, // focus (world XZ) - moved right from -6 to -3
    22.0, 18.0, // ellipse radii - shrunk from 30,22 to cut left side
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
