// Lab 10 Exercise 1(c): timed vector multiply with 100 million elements.
// Prints the kernel-only time and the whole copy-in -> compute -> copy-out time.
#include <cstdio>
#include <cstdlib>
#include <sys/time.h>

#define N 100000000             // 100 million elements (about 400 MB per array)
#define THREADS_PER_BLOCK 512

__global__ void vecMul(int *a, int *b, int *c, int n) {
    // TODO 1: global index of this thread
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // TODO 2: only compute if i is inside the array. The last block has 384 extra
    // threads (19,532 x 512 = 10,000,384), and without this check they would read and
    // write past the end of the arrays.
    if (i < n)
        c[i] = a[i] * b[i];
}

void random_ints(int *x, int size) {
    for (int i = 0; i < size; i++)
        x[i] = rand() % 100;
}

// Track CPU (wall-clock) time, in seconds
double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)tp.tv_sec + (double)tp.tv_usec * 1.e-6;
}

int main(void) {
    double dStart, dElaps, wStart, wElaps;
    int *a, *b, *c;             // host copies of a, b, c
    int *d_a, *d_b, *d_c;       // device copies of a, b, c
    size_t size = (size_t)N * sizeof(int);

    // Allocate device memory
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Allocate host memory and generate the input values
    a = (int *)malloc(size); random_ints(a, N);
    b = (int *)malloc(size); random_ints(b, N);
    c = (int *)malloc(size);

    // Whole sequence starts before the first host-to-device copy
    wStart = cpuSecond();

    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // TODO 3: number of blocks needed to cover N elements (round UP)
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    printf("Launching %d blocks x %d threads\n", blocks, THREADS_PER_BLOCK);

    // TODO 4: launch vecMul with <<<blocks, THREADS_PER_BLOCK>>>
    dStart = cpuSecond();
    vecMul<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();    // wait for the GPU to finish, THEN stop the clock
    dElaps = cpuSecond() - dStart;

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));

    // TODO 5: copy the result d_c back into c on the host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    wElaps = cpuSecond() - wStart;   // ...and stops after the last device-to-host copy

    // print only the last 5 results (100 million is too many to print)
    for (int i = N - 5; i < N; i++)
        printf("%d) %d x %d = %d\n", i, a[i], b[i], c[i]);

    // TODO 7: verify all N results on the CPU and print the number of mismatches
    int errors = 0;
    for (int i = 0; i < N; i++)
        if (c[i] != a[i] * b[i]) errors++;
    printf("Verification: %d mismatches out of %d\n", errors, N);

    printf("Kernel only        - %.7f s\n", dElaps);
    printf("Copy in + kernel + copy out - %.7f s\n", wElaps);

    // Cleanup
    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}
