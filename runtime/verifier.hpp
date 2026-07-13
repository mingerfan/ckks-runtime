#pragma once

#include "runtime/plan.hpp"

namespace fhegpu {

class PlanVerifier {
public:
    static void verify(const RuntimePlan &plan);
    static void verify_runtime_target(const RuntimePlan &plan, int rank,
                                      int world_size, int local_devices);
};

} // namespace fhegpu
