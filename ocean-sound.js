// Procedural Ocean Sound (Web Audio API, royalty-free, no external files)
// Generates pink-ish noise with slow amplitude modulation for wave-like rhythm
// Includes spatial audio via PannerNode that follows the main breaker position

let audioContext = null;
let soundGain = null;
let soundPanner = null;  // PannerNode for spatial audio
let soundEnabled = false;
let soundSource = null;
let soundInterval = null;
let soundNoiseBuffer = null;

export async function initOceanSound() {
  if (audioContext) return;
  audioContext = new (window.AudioContext || window.webkitAudioContext)();
  
  // Master gain
  soundGain = audioContext.createGain();
  soundGain.gain.value = 0;
  
  // Spatial panner - follows main breaker position
  soundPanner = audioContext.createPanner();
  soundPanner.panningModel = 'HRTF';  // High-quality spatialization
  soundPanner.distanceModel = 'inverse';
  soundPanner.refDistance = 8.0;      // Reference distance (world units)
  soundPanner.maxDistance = 120.0;    // Max distance for attenuation
  soundPanner.rolloffFactor = 1.2;    // Roll-off rate
  soundPanner.coneInnerAngle = 120;   // Inner cone (full volume)
  soundPanner.coneOuterAngle = 180;   // Outer cone
  soundPanner.coneOuterGain = 0.3;    // Gain outside cone
  // Orientation: facing negative Z (toward camera). Use setOrientation for initial setup.
  soundPanner.setOrientation(0, 0, -1);
  
  // Chain: source -> gain -> panner -> destination
  soundGain.connect(soundPanner);
  soundPanner.connect(audioContext.destination);

  // Generate band-limited noise buffer (10 seconds, loopable)
  const sampleRate = audioContext.sampleRate;
  const duration = 10; // seconds
  const length = sampleRate * duration;
  soundNoiseBuffer = audioContext.createBuffer(1, length, sampleRate);
  const channelData = soundNoiseBuffer.getChannelData(0);

  // Pink-ish noise (Paul Kellet's approximation, filtered for wave-like spectrum)
  let b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0;
  for (let i = 0; i < length; i++) {
    const white = Math.random() * 2 - 1;
    b0 = 0.99886 * b0 + white * 0.0555179;
    b1 = 0.99332 * b1 + white * 0.0750759;
    b2 = 0.96900 * b2 + white * 0.1538520;
    b3 = 0.86650 * b3 + white * 0.3104851;
    b4 = 0.55000 * b4 + white * 0.5329522;
    b5 = -0.7616 * b5 - white * 0.0168980;
    channelData[i] = b0 + b1 + b2 + b3 + b4 + b5 + b6 + white * 0.5362;
    channelData[i] *= 0.11; // normalize
    b6 = white * 0.115926;
  }
}

export async function startOceanSound() {
  if (!audioContext) await initOceanSound();
  
  // CRITICAL: resume must happen in user gesture context
  if (audioContext.state === 'suspended') {
    console.log('[OceanSound] Resuming AudioContext...');
    await audioContext.resume();
    console.log('[OceanSound] AudioContext state:', audioContext.state);
  }

  stopOceanSound(); // clean up any existing

  // Create a looping noise source with slow amplitude modulation
  soundSource = audioContext.createBufferSource();
  soundSource.buffer = soundNoiseBuffer;
  soundSource.loop = true;
  soundSource.connect(soundGain);
  soundSource.start(0);
  console.log('[OceanSound] Source started, gain:', soundGain.gain.value);

  // Slow amplitude modulation (wave rhythm ~8-12 second periods)
  let phase = 0;
  soundInterval = setInterval(() => {
    if (!soundEnabled) return;
    phase += 0.05;
    // Multiple slow oscillators for natural wave rhythm
    const env = 0.35 + 0.25 * Math.sin(phase * 0.7)
      + 0.15 * Math.sin(phase * 1.3 + 1.2)
      + 0.10 * Math.sin(phase * 0.42 + 2.5);
    const targetGain = Math.max(0.02, env * 0.28);
    soundGain.gain.linearRampToValueAtTime(targetGain, audioContext.currentTime + 2.0);
  }, 200);
}

export function stopOceanSound() {
  if (soundInterval) {
    clearInterval(soundInterval);
    soundInterval = null;
  }
  if (soundSource) {
    soundSource.stop();
    soundSource.disconnect();
    soundSource = null;
  }
  if (soundGain && audioContext) {
    soundGain.gain.setValueAtTime(0, audioContext.currentTime);
  }
}

export function toggleOceanSound() {
  soundEnabled = !soundEnabled;
  const btn = document.getElementById('soundToggle');
  if (btn) btn.setAttribute('aria-pressed', soundEnabled);
  console.log('[OceanSound] Toggle:', soundEnabled ? 'ON' : 'OFF');
  if (soundEnabled) {
    startOceanSound().catch(err => console.error('[OceanSound] Start failed:', err));
  } else {
    stopOceanSound();
  }
}

export function isSoundEnabled() {
  return soundEnabled;
}

/**
 * Update spatial panner position to follow the main breaker
 * @param {number} x - World X position
 * @param {number} y - World Y position (height)
 * @param {number} z - World Z position
 * @param {number} dirX - Breaker direction X (optional, for orientation)
 * @param {number} dirZ - Breaker direction Z (optional, for orientation)
 */
export function updateBreakerPosition(x, y, z, dirX = 0, dirZ = -1) {
  if (!soundPanner) return;
  
  // Position the panner at the breaker location
  soundPanner.positionX.setValueAtTime(x, audioContext.currentTime);
  soundPanner.positionY.setValueAtTime(y, audioContext.currentTime);
  soundPanner.positionZ.setValueAtTime(z, audioContext.currentTime);
  
  // Orient panner to face the breaker's direction
  // This makes the sound directional along the wave crest
  soundPanner.orientationX.setValueAtTime(dirX, audioContext.currentTime);
  soundPanner.orientationY.setValueAtTime(0, audioContext.currentTime);
  soundPanner.orientationZ.setValueAtTime(dirZ, audioContext.currentTime);
}

export function setListenerPosition(x, y, z, forwardX, forwardY, forwardZ, upX, upY, upZ) {
  if (!audioContext || !audioContext.listener) return;
  
  // Update listener position (camera)
  audioContext.listener.positionX.setValueAtTime(x, audioContext.currentTime);
  audioContext.listener.positionY.setValueAtTime(y, audioContext.currentTime);
  audioContext.listener.positionZ.setValueAtTime(z, audioContext.currentTime);
  
  // Update listener orientation
  audioContext.listener.forwardX.setValueAtTime(forwardX, audioContext.currentTime);
  audioContext.listener.forwardY.setValueAtTime(forwardY, audioContext.currentTime);
  audioContext.listener.forwardZ.setValueAtTime(forwardZ, audioContext.currentTime);
  audioContext.listener.upX.setValueAtTime(upX, audioContext.currentTime);
  audioContext.listener.upY.setValueAtTime(upY, audioContext.currentTime);
  audioContext.listener.upZ.setValueAtTime(upZ, audioContext.currentTime);
}