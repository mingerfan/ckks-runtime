#pragma once

#include "api/mock_api.hpp"
#include "runtime/runtime.hpp"

#include <unordered_map>

namespace fhegpu {

struct DiffPoint {
    std::size_t op_ordinal = 0;
    ValueId distributed_value = 0;
    ValueId reference_value = 0;
};
using DiffMap = std::vector<DiffPoint>;

struct BuiltPlan {
    RuntimePlan plan;
    DiffMap diff_map;
    ValueId reference_output = 0;
};

BuiltPlan make_fanout_plan(const std::vector<int> &device_counts);

class SequentialReferenceExecutor {
public:
    std::unordered_map<ValueId, VecValue> run(const std::vector<ComputeOp> &ops,
                                               const std::unordered_map<ValueId, VecValue> &inputs);
};

std::unordered_map<ValueId, VecValue> run_fanout_reference(const VecValue &cipher,
                                                            const VecValue &plain);
void compare_values(const VecValue &actual, const VecValue &expected,
                    double absolute_tolerance = 0.0, double relative_tolerance = 0.0);

struct HarnessResult {
    std::vector<RunArtifact<VecValue>> artifacts;
    std::vector<MockStats> stats;
};

HarnessResult run_mock_cluster(const RuntimePlan &plan,
                               const std::unordered_map<ValueId, VecValue> &rank0_inputs,
                               MockClusterConfig cluster_config,
                               VecExecConfig exec_config,
                               DiffMode diff_mode = DiffMode::FinalOnly);

} // namespace fhegpu
