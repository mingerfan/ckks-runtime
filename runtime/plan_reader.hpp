#pragma once
#include "runtime/json_plan_reader.hpp"

namespace fhegpu {
struct PlanReadOptions {
    // Zero uses CKKS_RUNTIME_PLAN_READ_THREADS, or four threads when unset.
    unsigned threads = 0;
};
struct PlanReadStats : JsonReadStats {
    bool binary = false;
    unsigned threads = 1;
    double scan_allocate_seconds = 0;
    double decode_seconds = 0;
};
class RuntimePlanReader {
public:
    // .bin requires CKKSPL01; .json uses the existing strict JSON reader.
    static LoadedRuntimePlan read_file(const std::string &path,
                                      PlanReadOptions options = {}, PlanReadStats *stats = nullptr);
};
} // namespace fhegpu
