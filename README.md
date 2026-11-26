# Evaluation artifact: split-genome GPU kernels for evolutionary algorithms

Artifact of the paper *Split-Genome Layouts for Evolutionary Algorithms on GPUs: Bottlenecks
and Trade-offs at Million-Variable Scale*: the CUDA kernels, the EvoX-based baselines, the
raw timing and fitness data, and the scripts that regenerate every figure and table.

- `code/` CUDA C++ kernels and their tests
- `python/` the EvoX-based baselines
- `scripts/` experiment drivers and analysis scripts
- `configs/` the problem shapes of the experiments
- `results/` the raw data of the paper

## Requirements

- To regenerate the figures and tables: Python 3 (`pip install -r requirements-analysis.txt`). No GPU.
- To run the kernels: an NVIDIA GPU, the CUDA Toolkit 12.9 or newer (13.2 for Blackwell) and
  [NVIDIA MathDx 25.12](https://developer.nvidia.com/mathdx). MathDx is not redistributed:
  unpack it into `code/external/mathdx/25.12/` or set `MATHDX_DIR` to its `include/` directory.
- To run the baselines: PyTorch 2.14.0 with CUDA and [EvoX 1.4.0](https://github.com/EMI-Group/evox)
  (`pip install -r requirements-baselines.txt`), and a C compiler for `torch.compile`.

## Quick start

Regenerate all figures, tables and numbers from the shipped data (no GPU, a few minutes):

    ./reproduce_figures.sh

Build and test the kernels and time one configuration (GPU, a few minutes):

    CUDA_ARCH=120 MATHDX_DIR=/path/to/mathdx/25.12/include ./smoke_test.sh

Measure six speedups over the baselines and compare them with the paper's (GPU,
about 10 minutes, after the smoke test has built the kernels):

    ./speedup_tests.sh

`CUDA_ARCH` is the compute capability of the GPU without the dot (120 for the RTX PRO 6000
Blackwell, 86 for the RTX A6000). `REPRODUCE.md` maps each figure and table to its script,
gives the commands that re-run every experiment, and describes the data.

## License and citation

Code under the MIT license (`LICENSE`), data in `results/` under CC BY 4.0. Please cite the
paper (`CITATION.cff`).
