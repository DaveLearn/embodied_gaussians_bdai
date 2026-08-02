# frame-seg-init provenance

The `src/frame_seg_init` implementation and its focused tests were integrated
from the `frame-seg-init` repository:

- Upstream: `git@github.com:DaveLearn/frame-seg-init.git`
- Source branch: `integration/python312-library-api`
- Source commit: `287c0a9779efc6030f0a22c80521ec7d75d9fd48`
- Original author: David Pershouse

The DEG-specific command-line adapter was deliberately omitted. Embodied
Gaussians uses the package-owned library types directly through
`embodied_gaussians.segmentation.frame_seg`.

The integrated version targets Python 3.12, CUDA 13 PyTorch wheels, Open3D
0.19, an explicit device/cache/checkpoint runtime configuration, and no
automatic model downloads.
