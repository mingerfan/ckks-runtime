/* MPI 可用性自检程序。
 * 覆盖三类基本通信：
 *   1) 启动/收尾 + rank/size 查询
 *   2) 集合通信：MPI_Allreduce
 *   3) 点对点通信：MPI_Sendrecv（环状传递）
 * 另在 rank 0 做一次 VecOps（operations.hpp）最小自检，确保 C++ 工具类可用。
 * 退出码 0 表示全部子检查通过。
 */
#include <mpi.h>

#include <cstddef>
#include <cstdio>

#include "operations.hpp"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char proc_name[MPI_MAX_PROCESSOR_NAME];
    int name_len = 0;
    MPI_Get_processor_name(proc_name, &name_len);

    printf("[rank %d/%d] hello on host '%s'\n", rank, size, proc_name);
    fflush(stdout);

    int all_ok = 1;

    /* --- 0) VecOps(CKKS 抽象) 自检（仅 rank 0，release 安全，不参与 OVERALL 聚合）--- */
    if (rank == 0) {
        constexpr std::size_t KN = 4096;          /* 自检用较小环维，快速 */
        VecOps<double, KN> ops;
        VecData<double, KN> pt;  pt.fill(2.0);
        VecData<double, KN> ct;  ct.fill(3.0);
        VecData<double, KN> s = ops.add_cp(ct, pt);      /* 期望逐元素 5.0 */
        VecData<double, KN> p = ops.mul_cp(ct, pt);      /* 期望逐元素 6.0 */
        VecData<double, KN> rs = ops.rescale(p);         /* 模拟后端 no-op copy */
        VecData<double, KN> ms = ops.modswitch(p);       /* 模拟后端 no-op copy */
        VecData<double, KN> bt = ops.boot(p);            /* 模拟后端 no-op copy */

        VecData<double, KN> seq;
        seq[0] = 10.0;
        seq[1] = 20.0;
        VecData<double, KN> rot = ops.rotate(seq, 1);
        bool ok = (s[0] == 5.0) && (p[0] == 6.0) &&
                  (rs[0] == 6.0) && (ms[0] == 6.0) && (bt[0] == 6.0) &&
                  (s[KN - 1] == 5.0) && (p[KN - 1] == 6.0) &&
                  (rot[0] == 20.0) && (rot[KN - 1] == 10.0);
        if (!ok) {
            fprintf(stderr, "[rank 0] VecOps self-check FAILED\n");
            MPI_Abort(MPI_COMM_WORLD, 3);
        }
        printf("[rank 0] VecOps self-check OK\n");
        fflush(stdout);
    }

    /* --- 1) 集合通信：所有 rank 的序号求和 ----------------------------- */
    int local = rank;
    int sum = -1;
    MPI_Allreduce(&local, &sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    int expected_sum = (size * (size - 1)) / 2;
    int sum_ok = (sum == expected_sum);
    if (rank == 0) {
        printf("[rank 0] Allreduce SUM = %d (expected %d) -> %s\n",
               sum, expected_sum, sum_ok ? "OK" : "FAIL");
        fflush(stdout);
    }
    all_ok &= sum_ok;

    /* --- 2) 点对点：环状传递（i -> i+1, 收自 i-1）--------------------- */
    int next = (rank + 1) % size;
    int prev = (rank - 1 + size) % size;
    int token = rank * 100;                 /* 发给 next */
    MPI_Sendrecv_replace(&token, 1, MPI_INT,
                         next, 42,           /* 发往 next，tag=42 */
                         prev, 42,           /* 接收自 prev，tag=42 */
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    int expected_token = prev * 100;         /* 来自 prev 的值 */
    int ring_ok = (token == expected_token);
    printf("[rank %d] ring received %d (expected %d) -> %s\n",
           rank, token, expected_token, ring_ok ? "OK" : "FAIL");
    fflush(stdout);
    all_ok &= ring_ok;

    /* --- 3) 汇总结果，由 rank 0 给出总判定 --------------------------- */
    int global_ok = 0;
    MPI_Allreduce(&all_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("[rank 0] OVERALL: %s\n", global_ok ? "ALL CHECKS PASSED" : "SOME CHECK FAILED");
        fflush(stdout);
    }

    MPI_Finalize();
    return global_ok ? 0 : 1;
}
