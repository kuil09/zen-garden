# Design QA

## Target

Hokusai, *Under the Wave off Kanagawa* — the woodblock print, not a photographic barrel.
Limited Prussian-blue palette, flat colour plates, carved ink outlines, claw-shaped
foam, a low viewpoint, and paper texture.

## What changed

The previous build could not reach that target for a structural reason, not a
tuning one: the water was a camera-relative height-field fan sampling a wrapping
84 m simulation domain. An overturning wave is multi-valued in `(x, z)`, so no
choice of amplitude, choppiness, or wind speed could have produced one. The
rendered result was a calm open sea seen from eye level, identical at t = 12 s,
30 s and 60 s.

- The breaker is now a parametric swept sheet (`shaders/wave-geometry.wgsl`).
  The cross-section spirals past vertical, which is what produces the overhang.
- Foam claws are real geometry rooted on the crest and cut into fingers in the
  fragment stage, so they stand against the sky in silhouette. Drawing them as a
  surface mask alone left them trapped inside the wave.
- Shading resolves to four flat plates; a Sobel over the frame finds the plate
  boundaries and inks them. The photographic path (GGX, glitter, full Fresnel,
  bloom, ACES) is gone.
- The spray particle system was removed. It was driven by the old height field
  and emitted into the wrong places; drops are now carved from the claw strip.

## Verification

Captured headless at 1280x720 (`--enable-unsafe-webgpu --use-angle=metal`) at
t = 6 s, 12 s, 30 s and 60 s. Note that headless virtual time does not advance CSS
transitions, so captures need `?nofade` or they catch the canvas mid fade-in.

- The wave silhouette holds across all four capture times, and the break phase
  travels along the crest between them.
- No WGSL compilation or WebGPU validation errors; the WebGPU fallback notice
  stays hidden, so initialisation succeeds.
- Pause, reduced motion, adaptive resolution, and pointer parallax are unchanged.

## Known gaps

- Fuji is a background SDF and is partly occluded by the mid-distance swell.
- The great wave's near flank reads as a straight edge; the print's sweeps up
  more gradually.
- There are no boats.
