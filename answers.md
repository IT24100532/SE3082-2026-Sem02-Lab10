# Lab 10 – timings to fill in after running on Colab

The GPU times can only be measured on the Colab T4, so fill these in from the notebook output.

## Exercise 1(b) – Lab 9 programs with the timer

| Lab 9 program | Launch configuration | Run 1 (s) | Run 2 (s) | Run 3 (s) |
|---|---|---|---|---|
| Exercise 3 | 512 blocks × 1 thread | | | |
| Exercise 4 | 1 block × 512 threads | | | |
| Exercise 5 | 10,000,000 elements, 19,532 blocks × 512 threads | | | |
| Exercise 6 | 100,000,000 elements, 313 × 625 blocks of 32 × 16 | | | |

**Predict 1:** Exercise 4 (1 block × 512 threads) should be faster than Exercise 3 (512 blocks × 1 thread).
Exercise 4 fills 16 complete warps, while each block in Exercise 3 uses only 1 of the 32 lanes in its warp,
and the GPU has to schedule 512 separate blocks. (Both are so small that the difference may be close to
the timer's resolution.)

**Predict 2:** Exercise 6 has 10× as many elements as Exercise 5, so it should take roughly 10× as long:
both kernels do one multiplication per element and are limited by memory speed. Compare your table and
explain any difference (for example, the very first kernel launch in a program is slower).

## Exercise 1(c) – 100 million elements

| Measurement | Time (s) |
|---|---|
| Kernel only | |
| Copy in + kernel + copy out | |

Expect the whole sequence to be many times larger than the kernel alone: the copies move 1.2 GB over PCIe,
while the kernel does one multiplication per element.

## Exercise 4 – 3D kernel vs Lab 9 Exercise 6

| Program | Elements | Time taken (s) |
|---|---|---|
| Lab 9 Exercise 6 (2D) | 100,000,000 | |
| Lab 10 Exercise 4 (3D) | 125,000,000 | |
