## <div align="center">Lab 10 – CUDA: timing, errors, 2D and 3D grids (Google Colab)</div>

SE3082 – Parallel Computing, Semester 2, 2026 · Lab sheet: [LabSheet10-CUDA.pdf](LabSheet10-CUDA.pdf)

### How to run
1. Upload `SE3082_Lab10_IT24100532.ipynb` to [Colab](https://colab.research.google.com) (**File → Upload notebook**).
2. **Runtime → Change runtime type → T4 GPU → Save**, then **Runtime → Run all**.
3. Copy the times into [answers.md](answers.md), then download the notebook with its outputs and add it here.

### Files
| File | Exercise |
|---|---|
| `src/ex1_timed.cu`, `src/ex1_nosync.cu` | Exercise 1(a): timed program, and the experiment without `cudaDeviceSynchronize()` |
| `src/ex3_blocks_timed.cu`, `src/ex4_threads_timed.cu`, `src/ex5_vecmul_timed.cu`, `src/ex6_matrix_timed.cu` | Exercise 1(b): Lab 9 programs with the timer |
| `src/ex5_vecmul_100M_timed.cu` | Exercise 1(c): 100 million elements, kernel time vs whole-sequence time |
| `src/ex2_check.cu` | Exercise 2: `CHECK()` around every CUDA call (all TODOs done) |
| `src/ex2_check_expA.cu`, `src/ex2_check_expB.cu`, `src/ex1_expC.cu` | Exercise 2: experiments A, B and C |
| `src/ex3_picture.cu` | Exercise 3: 800 × 600 picture with a 2D grid of 16 × 16 blocks (all TODOs done) |
| `src/ex4_volume.cu` | Exercise 4: 500 × 500 × 500 volume with a 3D grid of 8 × 8 × 8 blocks (all TODOs done) |

The answers to the THINK / PREDICT / EXPERIMENT questions are text cells in the notebook.

### Checked before submitting
These programs were not run on a GPU yet. They were checked with a CPU emulation of the CUDA calls
on Linux (same `rand()` as Colab), and every value matches the lab sheet's self-check tables:

| Program | Result |
|---|---|
| ex1 / ex2 | `83 + 77 = 160` ... `93 + 24 = 117`, `0 mismatches out of 512` |
| Experiment A | `code:9, reason: invalid configuration argument` |
| Experiment B | `code:2, reason: out of memory` |
| Experiment C | no error message, `512 mismatches out of 512` |
| ex3_picture | `Grid: 50 x 38 blocks of 16 x 16 threads`, row 0: `103 198 105 115 81 255 74 236 41 205` → `206 255 210 230 162 255 148 255 82 255`, `0 mismatches out of 480000`, `240121` pixels capped |
| ex4_volume | `Grid: 63 x 63 x 63 blocks`, `C[499][498][0] = 76 x 15 = 1140`, `C[499][499][499] = 47 x 68 = 3196`, `0 mismatches out of 125000000` |

The times (Exercise 1 table, 1(c) and Exercise 4) can only be measured on the GPU.
