# Wave Research Experiment Checklist

## Scope

This checklist is for the implementation phase after external research is approved. It makes the research decision-complete. No experiment in this file has been run. Runtime, visual, and performance claims are NOT_RUN.

## Evidence protocol

- [ ] Keep two fixed seeds and three random seeds.
- [ ] Capture every condition for at least 30 seconds.
- [ ] Record browser; GPU; canvas size; device pixel ratio; render scale; and WebGPU adapter.
- [ ] Keep camera; palette; and post-process settings constant for each paired comparison.
- [ ] Change one mechanism at a time or record the confound.
- [ ] Save captures at 0; 6; 12; 20; 30; and 60 seconds when the condition survives 30 seconds.
- [ ] Mark observations as DIRECT_RUNTIME; IMAGE_INSPECTION; STATIC_ONLY; or NOT_RUN.

## Hypothesis ledger

| ID | Hypothesis | Falsification observation | Status | Decision |
| --- | --- | --- | --- | --- |
| H1 | Low/high bands preserve the large wave while sharpening the crest. | Detail repeats as tiles or drifts independently. | NOT_RUN | Defer to spectral capture. |
| H2 | Negative divergence plus minimum Hessian eigenvalue is better than a height threshold. | Foam flashes; covers calm water; or misses a breaking crest. | NOT_RUN | Defer to paired foam captures. |
| H3 | Surface-adhered foam lasts longer than ballistic-only particles. | Foam floats; tunnels; or costs more than the visual gain. | NOT_RUN | Defer to lifecycle comparison. |
| H4 | Gravity and drag with impact-dependent birth create readable spray. | Synchronized bursts or one dominant size appear. | NOT_RUN | Defer to spray histogram. |
| H5 | Curvature-aware contours improve crest readability over Sobel alone. | Lines become noisy or detach from form. | NOT_RUN | Defer to paired line captures. |
| H6 | A bounded anisotropic lobe creates broken highlights without glossy rendering. | Highlights overpower flat plates or flicker. | NOT_RUN | Defer to shading comparison. |
| H7 | Off-center crest and stable far mountain increase tension. | Crest clips or mobile hierarchy weakens. | NOT_RUN | Defer to aspect-ratio captures. |
| H8 | Irregular wind pulses reduce repetition over 30 seconds. | A dominant period or global brightness jump remains. | NOT_RUN | Defer to temporal review. |

## Experiment set

### E01: spectral band separation

- Control: existing spectral cascade and wave geometry.
- Intervention: one bounded high-frequency band or changed band gain.
- Measure: crest width; dominant spatial frequency; temporal correlation; visible repetition.
- Acceptance: sharper crest with coherent main-wave phase.
- Failure: tiled repetition; aliasing; or detached high-frequency motion.
- Expected cost: low to medium; NOT_RUN.
- Priority: A.
- Future contact: main.js; shaders/spectral.wgsl; shaders/dynamic.wgsl.

### E02: curvature and convergence foam source

- Control: current foam source and decay.
- Intervention: require minimum Hessian eigenvalue threshold and negative divergence together.
- Measure: foam area; foam birth rate; one-frame maximum increment; crest overlap.
- Acceptance: foam grows continuously at compressed crests and decays after the event.
- Failure: white flash; calm-water foam; or suppressed crest foam.
- Expected cost: one local stencil plus scalar operations; NOT_RUN.
- Priority: A.
- Future contact: shaders/resolve.wgsl; shaders/dynamic.wgsl.

### E03: foam advection and reattachment

- Control: current backtrace and exponential decay.
- Intervention: surface-adhesion and rejoin heuristic.
- Measure: foam centroid distance from surface; lifetime; reattachment count.
- Acceptance: foam follows the sheet and rejoins after short separation without popping.
- Failure: floating foam; tunneling; or excessive lifetime.
- Expected cost: low to medium; NOT_RUN.
- Priority: B.
- Future contact: shaders/resolve.wgsl; shaders/spray.wgsl.

### E04: spray birth and ballistic decay

- Control: current spray threshold; gravity; drag; and life range.
- Intervention: birth probability from breaking intensity; local flow; normal throw; gravity; and drag.
- Measure: birth count; life histogram; speed histogram; impact delay; source distance.
- Acceptance: continuous spray arc with varied sizes and no synchronized flash.
- Failure: uniform size; one-frame birth wall; or floating particles.
- Expected cost: existing particle pass plus parameters; NOT_RUN.
- Priority: B.
- Future contact: shaders/spray.wgsl; shaders/scene.wgsl.

### E05: whitewater class separation

- Control: one foam and spray visual treatment.
- Intervention: persistent surface foam versus transient ballistic spray.
- Measure: class lifetime; surface distance; opacity contribution; active count.
- Acceptance: anchored white crest and thrown droplets are visually distinct.
- Failure: classes merge or secondary pass dominates.
- Expected cost: low if existing fields are reused; NOT_RUN.
- Priority: B.

### E06: ink contour comparison

- Control: current post-process Sobel edge.
- Intervention: curvature-aware or crest-aware line weighting.
- Measure: crest line precision; distant line density; silhouette continuity.
- Acceptance: crest contour strengthens while distant ranks remain sparse.
- Failure: noisy outlines; double edges; or detached lines.
- Expected cost: low fragment cost; NOT_RUN.
- Priority: A.
- Future contact: shaders/post.wgsl; shaders/scene.wgsl.

### E07: anisotropic highlight cluster

- Control: current flat plates and jade crest highlight.
- Intervention: bounded Ward-like lobe aligned to crest tangent.
- Measure: highlight orientation; cluster count; temporal coherence; plate contrast.
- Acceptance: directional glints without a photographic surface.
- Failure: glossy glare; flicker; or flow-independent highlights.
- Expected cost: low fragment arithmetic; NOT_RUN.
- Priority: B.
- Future contact: shaders/scene.wgsl.

### E08: irregular wind pulse

- Control: current periodic pulse.
- Intervention: incommensurate low-frequency pulses with bounded amplitude and fixed seed.
- Measure: temporal autocorrelation; repeated crest interval; global luminance jump.
- Acceptance: no dominant short period over 30 seconds.
- Failure: obvious loop; synchronized flashes; or unstable contrast.
- Expected cost: negligible CPU arithmetic; NOT_RUN.
- Priority: B.
- Future contact: main.js shared parameters.

### E09: Hokusai composition preset

- Control: current camera and focal placement.
- Intervention: low camera; lateral offset; slight tilt; stable far reference locus.
- Measure: crest occupancy; focal centroid; negative space; mobile crop safety.
- Acceptance: crest dominates without clipping and far reference remains legible.
- Failure: clipping; flat centered composition; or hidden background locus.
- Expected cost: negligible; NOT_RUN.
- Priority: A.
- Future contact: main.js camera block; shaders/scene.wgsl background.

### E10: distance LOD and active spray budget

- Control: current fixed particle dispatch and draw behavior.
- Intervention: distance-based emission and draw reduction.
- Measure: active particle count; frame time; crest readability at desktop and mobile scales.
- Acceptance: desktop near 60 fps and mobile near 30 fps are measured rather than assumed.
- Failure: far spray consumes budget or near crest has visible holes.
- Expected cost: lower than control at distance; NOT_RUN.
- Priority: B.

## Quantitative review sheet

| Metric | Definition | Collection method | Acceptance target | Status |
| --- | --- | --- | --- | --- |
| Foam area | Fraction of pixels classified as foam | Image segmentation on fixed captures | No one-frame spike beyond tolerance | NOT_RUN |
| Foam increment | area(t) minus area(t-1) | Frame sequence analysis | Continuous change rather than flash | NOT_RUN |
| Crest repeat period | First strong temporal autocorrelation peak | Crest mask sequence | No dominant short period in first 30 seconds | NOT_RUN |
| Tile frequency | Repeated spatial peaks in 2D FFT of crest mask | Frame and temporal tile analysis | No stable grid-aligned peaks | NOT_RUN |
| Spray life | Particle lifetime distribution | GPU debug readback or instrumented capture | Broad distribution with no birth wall | NOT_RUN |
| Contour density | Ink pixels per image region | Capture segmentation | High near crest; sparse at distance | NOT_RUN |
| Frame time | Median and 95th percentile duration | Browser profiler | Desktop near 60 fps; mobile near 30 fps | NOT_RUN |
| Visual stability | Device loss; shader errors; validation errors | Runtime logs | No errors during 30-second run | NOT_RUN |

## Seed matrix

| Run | Seed | Device class | Duration | Captures | Status |
| --- | --- | --- | ---: | --- | --- |
| F1 | fixed-1 | Desktop | 30 s | 0; 6; 12; 20; 30 s | NOT_RUN |
| F2 | fixed-2 | Desktop | 30 s | 0; 6; 12; 20; 30 s | NOT_RUN |
| R1 | random-1 | Desktop | 30 s | 0; 6; 12; 20; 30 s | NOT_RUN |
| R2 | random-2 | Desktop | 30 s | 0; 6; 12; 20; 30 s | NOT_RUN |
| R3 | random-3 | Mobile | 30 s | 0; 6; 12; 20; 30 s | NOT_RUN |

## Decision rules

- Promote C to B only when the source mechanism is clear and its failure condition is observable.
- Promote B to A only when a paired capture improves the target visual constraint without violating preservation constraints.
- Do not call a paper method implemented from static inspection.
- Do not call the 30-second repetition constraint satisfied from a single screenshot.
- If a method improves local appearance but displaces stability; GPU cost; or mobile readability; record partial progress rather than a solved constraint.

## Final implementation gate

- [ ] Eight or more final sources reviewed at full-text or authoritative abstract level.
- [ ] Selected mechanism has a source-derived equation or clearly marked qualitative model.
- [ ] Control condition and falsification condition documented.
- [ ] Desktop and mobile budgets defined before code changes.
- [ ] Art-reference claims are not presented as physics claims.
- [ ] Runtime and performance fields remain NOT_RUN until measured.
