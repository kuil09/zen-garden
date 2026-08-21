# Tidal Atelier

Tidal Atelier is a full-screen procedural WebGPU ocean meditation. The visible water is generated live in the browser without image textures, video, or external visual assets.

The simulation combines a dual-cascade spectral ocean with a nonlinear shallow-water field. An HLL approximate Riemann solver advances colliding bores, while limiting steepness, Froude number, and flow convergence drive transported foam. A separate GPU compute pass emits ballistic spray from active breaking events. The scene is rendered through a multisampled HDR pipeline with Fresnel reflection, GGX highlights, Beer-Lambert absorption, crest scattering, bloom, and filmic tone mapping.

## Run locally

```bash
npm install
npm run dev
```

Open `http://localhost:4173` in a WebGPU-capable browser.

## Interaction

- Move or drag the pointer across the ocean to disturb the water.
- Press `Space` to pause or resume the simulation.
- Reduced-motion preferences automatically lower animation intensity.

## Browser support

Current Chromium browsers and recent Safari releases provide the best WebGPU support. Browsers without WebGPU receive a compatibility notice.
