#pragma once
#include "experiments/binary_io_codec.hpp"
#include "runtime/binary_plan_parallel.hpp"
namespace fhegpu::binary_io_experiment {
inline RuntimePlan read_plan_parallel(std::string_view bytes, unsigned threads,
                                     double *scan = nullptr, double *decode = nullptr) {
    return binary_io::read_plan_parallel(bytes, threads, scan, decode, "PLNEXP01");
}
}
