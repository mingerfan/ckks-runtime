#pragma once

#include <cstdint>
#include <iosfwd>
#include <string>
#include <variant>
#include <vector>

namespace fhegpu {

using ValueId = std::uint64_t;
using TransferId = std::uint64_t;

enum class ValueKind { Plaintext, Ciphertext };
enum class PlaceKind { Host, Device };

struct Place {
    PlaceKind kind = PlaceKind::Host;
    int rank = 0;
    int index = 0;
};

bool operator==(const Place &a, const Place &b);
bool operator!=(const Place &a, const Place &b);
bool operator<(const Place &a, const Place &b);
std::string to_string(const Place &place);

enum class ComputeKind {
    AddCC, AddCP, SubCC, SubCP, MulCC, MulCP,
    Negate, Rotate, Rescale, ModSwitch, Relinearize, Boot
};

enum class CommKind { Transfer, Replicate };
enum class CommHint { Auto, PointToPoint, Broadcast, Tree, Ring, HostStaged };
enum class Phase { Initialization, Execution, Finalization };
enum class RescaleMode { Eager, Lazy };
enum class BootMode { Native, DecryptReencrypt };
enum class BootImplementation { Native, DecryptReencrypt };
enum class RequiredCapability {
    Transfer, Replicate, HostCompute, BootNative, BootDecryptReencrypt
};
enum class KeyKind { Secret, Relin, Galois };

struct RotateAttrs { int steps = 0; };
struct RescaleAttrs { int target_level = 0; int target_scale_log2 = 0; };
struct ModSwitchAttrs { int target_level = 0; };
struct BootAttrs {
    int target_level = 0;
    int target_scale_log2 = 0;
    int target_components = 2;
    std::string operator_profile;
    BootImplementation implementation = BootImplementation::Native;
};
using ComputeAttrs = std::variant<std::monostate, RotateAttrs, RescaleAttrs,
                                  ModSwitchAttrs, BootAttrs>;

struct ComputeOp {
    ComputeKind kind = ComputeKind::AddCC;
    std::vector<ValueId> inputs;
    ValueId output = 0;
    Place place;
    ComputeAttrs attrs;
};

struct CommAction {
    TransferId id = 0;
    CommKind kind = CommKind::Transfer;
    CommHint hint = CommHint::Auto;
    std::vector<ValueId> inputs;
    std::vector<ValueId> outputs;
    std::vector<Place> sources;
    std::vector<Place> destinations;
    std::vector<ValueKind> output_types;
};

using InstructionBody = std::variant<ComputeOp, CommAction>;

struct Instruction {
    std::size_t ordinal = 0;
    InstructionBody body;
};

struct ValueDesc {
    ValueId id = 0;
    ValueKind kind = ValueKind::Ciphertext;
    Place place;
    std::string context;
    int level = 0;
    int scale_log2 = 0;
    bool ntt = true;
    int components = 2;
};

struct OperatorSpecRef {
    std::string id;
    int version = 0;
    std::string fingerprint;
};

struct KeyRequirement {
    KeyKind kind = KeyKind::Secret;
    Place place;
};

struct TargetConfig {
    std::string target_id;
    int world_size = 1;
    std::vector<int> device_counts;
    int capability_version = 0;
    OperatorSpecRef operator_spec;
    RescaleMode rescale_mode = RescaleMode::Eager;
    BootMode boot_mode = BootMode::Native;
    std::vector<RequiredCapability> required_capabilities;
};

struct RuntimePlan {
    std::uint32_t format_version = 1;
    std::uint64_t plan_id = 0;
    std::string fingerprint_sha256;
    TargetConfig target;
    std::vector<ValueDesc> values;
    std::vector<ValueId> external_inputs;
    std::vector<KeyRequirement> required_keys;
    std::vector<Instruction> initialization;
    std::vector<Instruction> execution;
    std::vector<Instruction> finalization;
    std::vector<ValueId> final_outputs;

    std::uint64_t fingerprint() const;
    void print(std::ostream &out) const;
};

std::string to_string(ValueKind kind);
std::string to_string(ComputeKind kind);
std::string to_string(CommKind kind);
std::string to_string(CommHint hint);
std::string to_string(Phase phase);
std::string to_string(RescaleMode mode);
std::string to_string(BootMode mode);
std::string to_string(BootImplementation implementation);
std::string to_string(RequiredCapability capability);
std::string to_string(KeyKind kind);

} // namespace fhegpu
