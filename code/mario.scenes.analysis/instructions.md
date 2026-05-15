# Instructions for Claude Code (Planning Mode)

You are working inside my `mario.scenes` repository.

Goal:
Create a reproducible pipeline that:
1. computes/updates the UMAP of scene annotations;
2. uses that UMAP to generate a slideshow-compatible sequence of short Manim animations;
3. each animation should cover one concept only;
4. together, the clips should tell the story of scene-space generalization and hippocampal ripples.

Important repo constraints:
- The repository root is `mario.scenes`.
- Existing/deprecated UMAP code is in:
  `mario.scenes/code/deprecated`
- New analysis code, animation code, configs, and outputs should go under:
  `mario.scenes/code/mario.scenes.analysis`
- Do not overwrite deprecated code.
- Inspect deprecated code and reuse logic only if useful.
- Create clean, modern, maintainable code in `code/mario.scenes.analysis`.

Main conceptual story:
- Each point is a Mario scene.
- Scene position comes from UMAP of design-pattern annotation vectors.
- Practiced/pre-sleep scenes define source points A.
- Learning spreads to nearby scenes through a generalization kernel.
- Hippocampal ripples may affect how performance generalizes after sleep.

Core kernel:
k(s,s') = exp(-d(s,s') / lambda)

Main model:
Delta y_{i,s} =
(gamma + delta R_i)
sum_{s' in A_i}
ell_{i,s'} exp(-d(s,s') / lambda)
+ epsilon_{i,s}

Hypotheses:

H1 — Global gain:
Delta y_{i,s} = alpha + eta R_i + epsilon_{i,s}

H2 — Structured gain:
Delta y_{i,s} =
(gamma + delta R_i)
sum_{s' in A_i}
ell_{i,s'} exp(-d(s,s') / lambda)
+ epsilon_{i,s}

H3 — Kernel widening:
Delta y_{i,s} =
gamma
sum_{s' in A_i}
ell_{i,s'} exp(-d(s,s') / (lambda_0 + lambda_1 R_i))
+ epsilon_{i,s}

Critical visual distinctions:
- H1: uniform brightness everywhere
- H2: stronger brightness near sources
- H3: wider radius of influence
- H2+H3: both

Deliverables:
1. UMAP pipeline outputs (CSV, distance matrices, metadata)
2. Manim animations module
3. Config file
4. Render scripts
5. README documentation

Animations to implement:
- SceneCloudIntro
- HighlightPracticedScenes
- KernelLambdaDemo
- KernelIntroSingleSource
- KernelIntroMultipleSources
- PrePostPerformanceConcept
- SleepRippleInterlude
- H1GlobalGain
- H2StructuredGain
- H3KernelWidening
- H2H3Combined
- ModelComparisonGrid
- EquationBuild
- DistanceCaveat
- TransferPrediction
- LevelToSceneAggregation
- RippleSliderDemo
- FinalPunchline

Implementation requirements:
- Use Manim Community Edition
- 16:9 output
- Short clips (5–15s)
- Clean visuals, minimal text
- Deterministic seeds
- Modular helper functions

Technical notes:
- Prefer Jaccard distance for modeling
- Use UMAP space for visualization if needed
- Configurable options for kernel source
- Simulate data if needed

Output naming:
01_scene_cloud_intro.mp4
...
18_final_punchline.mp4

Planning mode tasks:
1. Inspect repo
2. Identify deprecated UMAP code
3. Define dependencies
4. Propose structure
5. List steps
6. Identify risks
7. Explain animation generation
8. Explain rendering + slides usage

Do NOT start coding until plan is approved.
