#include <mpi.h>

#include <cstdio>
#include <vector>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size < 2) {
        if (rank == 0) std::fprintf(stderr, "mpi_comm_benchmark requires at least two ranks\n");
        MPI_Finalize();
        return 1;
    }
    const std::vector<int> sizes{1, 64, 1024, 65536, 1048576};
    std::vector<unsigned char> buffer(static_cast<std::size_t>(sizes.back()), static_cast<unsigned char>(rank));
    if (rank == 0) std::printf("bytes latency_us bandwidth_MiB_s\n");
    for (int bytes : sizes) {
        MPI_Barrier(MPI_COMM_WORLD);
        const double begin = MPI_Wtime();
        constexpr int repeats = 100;
        if (rank == 0) {
            for (int i = 0; i < repeats; ++i) {
                MPI_Send(buffer.data(), bytes, MPI_BYTE, 1, 10, MPI_COMM_WORLD);
                MPI_Recv(buffer.data(), bytes, MPI_BYTE, 1, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        } else if (rank == 1) {
            for (int i = 0; i < repeats; ++i) {
                MPI_Recv(buffer.data(), bytes, MPI_BYTE, 0, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(buffer.data(), bytes, MPI_BYTE, 0, 11, MPI_COMM_WORLD);
            }
        }
        const double elapsed = MPI_Wtime() - begin;
        if (rank == 0) {
            const double one_way = elapsed / (2.0 * repeats);
            std::printf("%d %.3f %.2f\n", bytes, one_way * 1e6,
                        static_cast<double>(bytes) / one_way / (1024.0 * 1024.0));
        }
    }
    MPI_Finalize();
    return 0;
}
