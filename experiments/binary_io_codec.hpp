#pragma once
// Keep the historical experiment wire magic separate from production files.
#include "runtime/binary_plan_codec.hpp"
#include "runtime/binary_manifest_codec.hpp"
namespace fhegpu::binary_io_experiment {
using namespace binary_io;
inline void write_plan(std::ostream &out, RuntimePlan &p) { binary_io::write_plan(out, p, "PLNEXP01"); }
inline RuntimePlan read_plan(std::istream &in, std::uint64_t size) { return binary_io::read_plan(in, size, "PLNEXP01"); }
template <bool R> void manifest_records(Archive<R> &a, Manifest &m) { binary_io::manifest_records(a, m, "MNFEXP01"); }
}
