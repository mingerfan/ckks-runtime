#pragma once

#include "runtime/operator_spec.hpp"
#include "runtime/plan.hpp"

namespace fhegpu {

class PlanVerifier {
public:
    static PlanRequirements verify(const RuntimePlan &plan,
                                   const LoadedOperatorSpec &operator_spec,
                                   bool skip_artifact_digest_checks = false);
    static void verify_runtime_target(const RuntimePlan &plan, int rank,
                                      int world_size, int local_devices);
};

} // namespace fhegpu
