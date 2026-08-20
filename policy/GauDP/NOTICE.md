# Vendored source notice

This directory is self-contained at runtime. The repositories below were used
only as source material; neither is imported, linked, or placed on `PYTHONPATH`.

- Policy-Lightning (`https://github.com/Ziyeeee/Policy-Lightning`), commit
  `c944b4989a89c99c69d2572ea870f6a04680f5e7`.
  The diffusion U-Net, vision utilities, normalization utilities, and
  `GaussianConvEncoder` under `gaudp/core/` were copied and their imports were
  made package-relative. Upstream license: MIT; a copy is in
  `LICENSE-POLICY-LIGHTNING`.
- NoPoSplat, commit `a097a78c5bdd0486493f74abb1165614d86ae952`
  (`https://github.com/cvg/NoPoSplat`). Encoder, renderer, camera geometry and
  supporting model code are under `gaudp/third_party/noposplat/`. Its MIT
  license is preserved as `gaudp/third_party/noposplat/LICENSE`.
- The vendored CroCo/cuRoPE files retain their original copyright headers and
  CC BY-NC-SA 4.0 notice. Review that non-commercial license before using this
  implementation outside research.

All local integration code in this directory follows the containing MHBench
repository's license.
