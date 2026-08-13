# Tidal Atelier

Tidal Atelier is a full-screen WebGPU breaking wave drawn in the manner of Hokusai's *Under the Wave off Kanagawa*. Everything on screen is generated live in the browser; there is no image texture, video, or external visual asset anywhere in the pipeline.

The breaker is a parametric swept sheet rather than a height field. A height field is single-valued in `(x, z)` and so cannot describe an overturning wave, where the lip hangs forward over its own face. `shaders/wave-geometry.wgsl` sweeps a cross-section around a shrinking spiral instead: up the steep face, over the crest, forward through the lip, and down the plunging tongue. The foam claws are separate geometry rooted on the crest, cut into fingers by a branching field in the fragment stage, with the water thrown past their tips carved into discrete drops.

A dual-cascade spectral ocean and a nonlinear shallow-water field (HLL approximate Riemann solver) still run every frame. They no longer decide the wave's shape; they carry the open sea and modulate detail across the breaker's surface.

Rendering is deliberately not photographic. Surfaces resolve to four flat plates of Prussian blue, the frame is edge-detected into carved ink outlines, and the result is printed onto paper with fibre grain and per-channel plate misregistration.

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
