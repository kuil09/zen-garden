# Wave Research Brief

## Purpose and boundary

This is an external research artifact for the WebGPU Hokusai-wave project. It does not modify main.js, WGSL shaders, render settings, tests, or deployment configuration. Existing code paths are named only as future implementation touchpoints.

The research target is an evidence-backed shortlist for five questions:

1. How can spectral and shallow-water fields produce a more legible plunging crest?
2. How can foam and spray be tied to curvature, compression, convergence, and decay?
3. Which secondary-particle behaviors are credible at WebGPU scale?
4. Which stylized shading mechanisms strengthen a woodblock reading without restoring photographic rendering?
5. Which Hokusai composition observations can become explicit visual hypotheses?

## Evidence convention

| Label | Meaning |
| --- | --- |
| PRIMARY_FULLTEXT | Full paper or technical document was directly readable from the linked source. |
| PRIMARY_LINK | The author, publisher, or institutional link is authoritative; full extraction may require a later download. |
| PRIMARY_ABSTRACT | The source page or abstract was readable; detailed equations require full-text review. |
| DIRECT_OFFICIAL | Official museum or institutional collection or essay source. |
| SECONDARY_SURVEY | Survey or index used to locate methods, not as sole evidence for numerical claims. |
| INFERENCE | Implementation translation proposed by this brief, not a claim made by the source. |
| NOT_RUN | No WebGPU runtime, visual capture, or performance claim has been established. |

The matrix separates source claims from implementation interpretations. A source can support a mechanism while leaving its WebGPU cost, artistic fit, and parameter scale unresolved.

## Constraint model

| Constraint | Role | Status | Evidence condition |
| --- | --- | --- | --- |
| Eight or more final sources have stable links | Target / hard | Given | Every final row links to a primary paper, official technical document, or official museum source. |
| No code or shader modification during research | Preservation / hard | Given | Only the three research artifacts are added. |
| Physics and art interpretation remain distinct | Preservation / hard | Given | Source-derived statements and implementation inferences are labeled separately. |
| WebGPU feasibility is not claimed without a runtime experiment | Boundary / hard | Given | Cost and feasibility entries remain estimates or NOT_RUN until measured. |
| Artistic impact is prioritized over minimum power | Priority / soft | Given | Ranking favors crest readability and foam continuity before lowest cost. |
| Mobile viability remains a future constraint | Resource / soft | Provisional | LOD and particle reductions are experiments, not verified results. |

## 30 to 15 to 10 screening

The full candidate set is in wave-paper-matrix.csv. The first pass contains 30 rows across ocean core, whitewater, stylized shading, and art reference. The shortlist retains 15 rows with direct relevance to the current architecture. The final ten are the smallest set that jointly covers spectrum, stability, breaking, whitewater, contour, anisotropic shading, and visual composition.

### Final ten candidates

| Rank | ID | Priority | Decision | Why it survives |
| ---: | --- | --- | --- | --- |
| 1 | A01 | A | Reproduce spectral equations and parameter semantics first. | Direct basis for FFT and dual-scale ocean paths. |
| 2 | A03 | A | Use breaking-wave profile and spray attachment as behavior reference. | Addresses the shape problem without a new full solver. |
| 3 | A05 | A | Test minimum-Hessian-eigenvalue plus decay against foam continuity. | Makes foam a measurable surface event. |
| 4 | C02 | A | Compare screen contours with curvature-aware contour intent. | Directly addresses ink-like outline behavior. |
| 5 | A02 | B | Use stability and dissipation as a control condition. | Defines timestep behavior without replacing HLL. |
| 6 | B02 | B | Separate spray, foam, mist, and bubble behavior conceptually. | Production-proven lifecycle model at a practical level. |
| 7 | B03 | B | Test surface adhesion and phase-specific lifetimes. | Addresses floating foam and abrupt disappearance. |
| 8 | C03 | B | Test a bounded anisotropic lobe along crest tangent. | Supports directional highlights without glossy rendering. |
| 9 | B01 | B | Use droplet-size and impact timing distributions as calibration targets. | Adds variation without reproducing enormous CFD grids. |
| 10 | C01 | B | Compare warm/cool tone mapping against flat ink ramps. | Low-cost volume readability experiment. |

D01 through D04 remain official art references and are not counted as physics candidates.

## Axis A: ocean core

### Tessendorf: spectral synthesis

Tessendorf's notes provide the direct basis for a two-scale spectral ocean. The frequency-domain height field evolves with gravity-wave dispersion and is reconstructed with an inverse FFT.

    h(k,t) = h0(k) exp(i omega(k)t) + conjugate(h0(-k)) exp(-i omega(k)t)
    omega(k) = sqrt(g * |k|)
    h(x,t) = sum_k h(k,t) exp(i k dot x)

The Phillips spectrum controls directional energy and damping. The source supports the method and variables; the choice of two cascades, scale ratio, and artistic wind pulse remains an implementation decision.

Source: [Simulating Ocean Water](https://jtessen.people.clemson.edu/reports/papers_files/coursenotes2004.pdf).

### Stam: stability boundary

Stam describes force addition, backtraced advection, implicit diffusion, and projection. The source explicitly distinguishes visual plausibility from engineering accuracy and notes numerical dissipation as a tradeoff.

    u(n+1)(x) = u(n)(x - u dt)
    div(u) = 0 after projection

This is not a recommendation to replace the current HLL shallow-water path. It is a control condition for timestep behavior and dissipation.

Source: [Stable Fluids](https://graphics.stanford.edu/courses/cs448-01-spring/papers/stam.pdf).

### Wavelet turbulence: scale separation

Kim et al. separate large-scale flow from high-frequency detail and emphasize temporal coherence. This aligns with a rounded body plus sharp crest, but the original method targets larger fluid-detail synthesis. The implementation translation is limited to bounded bands and a coherence test.

Source: [Wavelet Turbulence for Fluid Simulation](https://www.cs.cornell.edu/~tedkim/WTURB/wavelet_turbulence.pdf).

## Axis B: breaking foam and spray

### Peachey: wave breaking and spray attachment

Peachey's paper describes a phase function, a profile affected by steepness and depth, and particles for spray from breaking waves. It supports separating the main wave profile from secondary spray. Exact parameter mapping to this project remains INFERENCE until full-text review.

Source: [Modeling waves and surf](https://doi.org/10.1145/15886.15893).

### Whitecap phenomenology: minimum eigenvalue and fraction

The Tessendorf, Reinhardt, and Gao report describes calibrating a minimum-eigenvalue threshold against an ensemble whitecap fraction. For a local height Hessian:

    lambda_min = 0.5 * (Hxx + Hzz - sqrt((Hxx - Hzz)^2 + 4 Hxz^2))
    W = P(lambda_min < T)

The first equation is the 2D Hessian eigenvalue translation. The report's contribution is threshold calibration and decay framing. Requiring both a curvature event and negative velocity divergence is an implementation hypothesis, not a complete physical breaking law.

Source: [Whitecap Phenomenology for Ocean Surface Simulation](https://jtessen.people.clemson.edu/gilligan/html/whitecap_fraction.pdf).

### Droplet distributions

Wang, Yang, and Stern use extremely high-resolution breaking-wave simulations to study bubbles and droplets including size distributions. For this project the value is calibration: test whether one particle size dominates, whether post-breaking birth is delayed, and whether spray intensity changes with breaking severity.

Source: [High-fidelity simulations of bubble droplet and spray formation in breaking waves](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/highfidelity-simulations-of-bubble-droplet-and-spray-formation-in-breaking-waves/8B26A9EA5D63F43BDC6BA15B2AEA7AC1).

### Whitewater lifecycle references

The Pixar memo describes layered foam, spray, mist, and bubbles in production. Wretborn, Flynn, and Stomakhin describe surface-constrained wet foam and phase-specific transport. Together they motivate these implementation hypotheses:

- foam remains near and advects with the surface;
- spray receives ballistic gravity and drag;
- phase changes are controlled by depth or surface contact;
- emission is continuous enough to avoid a one-frame flash.

Sources: [Simulating Whitewater Rapids in Ratatouille](https://graphics.pixar.com/library/Whitewater/paper.pdf) and [Guided Bubbles and Wet Foam for Realistic Whitewater Simulation](https://alexey.stomakhin.com/research/siggraph2022_whitewater.pdf).

## Axis C: stylized shading and line structure

### Gooch tone mapping

Gooch et al. provide a non-photorealistic lighting model that maps lighting into warm and cool tones for technical illustration. The research use is a bounded ramp comparison, not an attempt to reproduce Hokusai's exact ink chemistry.

Source: [A non-photorealistic lighting model for automatic technical illustration](https://doi.org/10.1145/280814.280950).

### Suggestive contours

DeCarlo et al. define suggestive contours using visible-surface curvature behavior and radial-curvature zero crossings. The research question is whether a curvature-aware line criterion improves the current screen-space Sobel result around the crest and far sea ranks.

Source: [Suggestive Contours for Conveying Shape](https://ics.uci.edu/~majumder/vispercep/suggestive_contous.pdf).

### Ward anisotropic reflection

Ward's anisotropic reflection model provides a direction-dependent lobe. The implementation interpretation is to align tangent direction with crest flow and keep the lobe weak enough that the surface still reads as a print.

Source: [Measuring and Modeling Anisotropic Reflection](https://eta-publications.lbl.gov/sites/default/files/lbl-31698.pdf).

## Axis D: Hokusai composition and material reference

The Met source supports research into Prussian blue, saturation, contrast, and woodblock production. LACMA describes the circular wave and triangular mountain as a geometric composition that combines movement and stability. The Tokyo Fuji Museum emphasizes movement versus stillness and near/far juxtaposition. The NGV course provides additional print and key-line context.

These sources establish art-historical observations, not physical or rendering equations. Future visual experiments should state an artistic hypothesis explicitly, for example: an off-center crest and stable far mountain increase perceived tension. They should not state that Hokusai requires a specific camera offset.

Sources: [The Great Wave: Anatomy of an Icon](https://www.metmuseum.org/ja/essays/hokusai-great-wave), [LACMA collection notes](https://collections.lacma.org/object/88781), [Tokyo Fuji Art Museum collection](https://www.fujibi.or.jp/en/collection/artwork/03628/), and [NGV online course](https://courses.ngv.vic.gov.au/topic/the-great-wave/).

## Open questions

1. Does curvature plus convergence improve foam continuity without making the surface white?
2. Does a low/high frequency split sharpen crests without visible spectral tiling over 30 seconds?
3. Does surface adhesion improve foam persistence more than it increases particle cost?
4. Does a Ward-like directional lobe survive plate quantization while remaining ink-like?
5. Does an off-center composition improve the focal read across desktop and mobile aspect ratios?

All five questions remain NOT_RUN until the implementation phase produces controlled captures and frame-time evidence.

