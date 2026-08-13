# Zen Garden

A calm, interactive WebGPU study for a short pause. The fragment shader creates slow, evolving ripples that respond to pointer movement. It is an original contemplative visual, not an ocean simulation or a recreation of another project.

## Run locally

Serve the directory with any static server, then open it in a WebGPU-capable browser.

```bash
python3 -m http.server 4173
```

Visit `http://localhost:4173`.

## Interaction

- Move a pointer or touch the canvas to leave a subtle ripple.
- Use the intensity slider to change the wave energy.
- Pause and resume the scene with the control button.
- Hide the interface with the quiet-mode control or the `H` key; press `Escape` to restore it.

## Browser support

The page uses WebGPU and gracefully shows a support message in browsers without it. Current Chromium browsers and recent Safari versions offer the best experience.
