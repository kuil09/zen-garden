// Procedural Ocean Sound (Web Audio API, royalty-free, no external files)
// Generates pink-ish noise with slow amplitude modulation for wave-like rhythm

let audioContext = null;
let soundGain = null;
let soundEnabled = false;
let soundSource = null;
let soundInterval = null;
let soundNoiseBuffer = null;

export async function initOceanSound() {
  if (audioContext) return;
  audioContext = new (window.AudioContext || window.webkitAudioContext)();
  soundGain = audioContext.createGain();
  soundGain.gain.value = 0;
  soundGain.connect(audioContext.destination);

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