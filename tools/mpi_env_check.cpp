#include <mpi.h>

#include <cstdio>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int sum = 0;
    MPI_Allreduce(&rank, &sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    const int expected = size * (size - 1) / 2;
    if (rank == 0) std::printf("MPI environment: ranks=%d allreduce=%s\n", size, sum == expected ? "PASS" : "FAIL");
    MPI_Finalize();
    return sum == expected ? 0 : 1;
}
