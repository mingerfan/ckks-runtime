/* MPI 通信带宽/延迟基准测试。
 *
 * 在 rank 0 与 rank 1 之间进行 ping-pong，对一组由小到大的消息长度采样：
 *   - 延迟 latency：单程时间（往返时间的一半），单位 µs
 *   - 带宽 bandwidth：单向吞吐 = len / 单程时间，单位 MB/s
 * 每个长度先做 warmup 再重复若干次取平均，时间用 MPI_Wtime()。
 *
 * 运行：mpirun -np 2 ./build/mpi_test_comm
 * 需要至少 2 个进程，其余 rank 闲置并随 Barrier 同步。
 * 退出码 0 表示正常完成。
 */
#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define WARMUP_ITERS 10     /* 预热次数，不计入计时 */
#define REPEAT_ITERS 100    /* 正式计时重复次数 */

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0)
            fprintf(stderr, "需要至少 2 个进程，请用 mpirun -np 2 (或更多) 运行\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("MPI 通信基准 (procs=%d, ping-pong 0<->1)\n", size);
        printf("%-10s %14s %16s %10s\n",
               "bytes", "latency(us)", "bandwidth(MB/s)", "iters");
        fflush(stdout);
    }

    /* 消息长度序列（2 的幂，便于观察 MPI 切换内部协议的阈值）*/
    static const size_t msg_sizes[] = {
        0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
        1024, 2048, 4096, 8192, 16384, 32768, 65536,
        131072, 262144, 524288, 1048576, 4194304, 8388608
    };
    const int n_sizes = (int)(sizeof(msg_sizes) / sizeof(msg_sizes[0]));

    /* 取最大长度分配一次缓冲区 */
    size_t max_len = 0;
    for (int i = 0; i < n_sizes; i++)
        if (msg_sizes[i] > max_len) max_len = msg_sizes[i];

    char *buf = (char *)malloc(max_len > 0 ? max_len : 1);
    if (!buf) {
        fprintf(stderr, "[rank %d] malloc 失败\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    /* 填充非零字节，避免全零缓冲被实现短路/压缩 */
    memset(buf, (rank == 0) ? 0x55 : 0xAA, max_len);

    for (int i = 0; i < n_sizes; i++) {
        const size_t len = msg_sizes[i];
        const int count = (int)len;            /* MPI_BYTE 个数即字节数 */

        if (rank == 0) {
            /* warmup */
            for (int it = 0; it < WARMUP_ITERS; it++) {
                MPI_Send(buf, count, MPI_BYTE, 1, 100, MPI_COMM_WORLD);
                MPI_Recv(buf, count, MPI_BYTE, 1, 101, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
            }
            /* 正式计时：一次 ping-pong = send + recv */
            const double t0 = MPI_Wtime();
            for (int it = 0; it < REPEAT_ITERS; it++) {
                MPI_Send(buf, count, MPI_BYTE, 1, 100, MPI_COMM_WORLD);
                MPI_Recv(buf, count, MPI_BYTE, 1, 101, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
            }
            const double dt = MPI_Wtime() - t0;

            const double per_round_s = dt / REPEAT_ITERS;
            const double lat_s = per_round_s / 2.0;          /* 单程 */
            const double lat_us = lat_s * 1e6;
            const double bw_mbs = (len > 0)
                ? (len / lat_s) / (1024.0 * 1024.0)
                : 0.0;

            printf("%-10zu %14.3f %16.2f %10d\n",
                   len, lat_us, bw_mbs, REPEAT_ITERS);
            fflush(stdout);
        } else if (rank == 1) {
            /* 与 rank 0 对称：先收后发 */
            for (int it = 0; it < WARMUP_ITERS; it++) {
                MPI_Recv(buf, count, MPI_BYTE, 0, 100, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
                MPI_Send(buf, count, MPI_BYTE, 0, 101, MPI_COMM_WORLD);
            }
            for (int it = 0; it < REPEAT_ITERS; it++) {
                MPI_Recv(buf, count, MPI_BYTE, 0, 100, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
                MPI_Send(buf, count, MPI_BYTE, 0, 101, MPI_COMM_WORLD);
            }
        }
        /* 其余 rank 在本轮 ping-pong 中闲置，靠 Barrier 同步进入下一长度 */
        MPI_Barrier(MPI_COMM_WORLD);
    }

    free(buf);
    MPI_Finalize();
    return 0;
}
