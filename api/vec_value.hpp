#pragma once

#include "runtime/plan.hpp"

#include <condition_variable>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace fhegpu {

struct VecMetadata {
    std::string context;
    std::uint64_t degree = 0;
    int level = 0;
    double scale = 1.0;
    bool ntt = true;
    int components = 1;
};

bool operator==(const VecMetadata &a, const VecMetadata &b);

struct VecPayload {
    ValueKind kind = ValueKind::Plaintext;
    std::vector<double> slots;
    VecMetadata metadata;
};

class VecValue {
public:
    VecValue() = default;
    static VecValue ready(VecPayload payload);
    static VecValue pending(ValueKind expected_kind);

    ValueKind kind() const;
    VecPayload materialize() const;
    VecValue deep_copy() const;
    void fulfill(VecPayload payload) const;
    void fail(std::exception_ptr error) const;
    const void *identity() const;

private:
    struct State {
        explicit State(ValueKind kind) : expected_kind(kind) {}
        ValueKind expected_kind;
        mutable std::mutex mutex;
        mutable std::condition_variable cv;
        bool ready = false;
        VecPayload payload;
        std::exception_ptr error;
    };
    explicit VecValue(std::shared_ptr<State> state) : state_(std::move(state)) {}
    std::shared_ptr<State> state_;
};

VecValue make_plain(std::vector<double> slots, std::string context = "ctx",
                    std::uint64_t degree = 8192, int level = 3,
                    double scale = 1.0, bool ntt = true);
VecValue make_cipher(std::vector<double> slots, std::string context = "ctx",
                     std::uint64_t degree = 8192, int level = 3,
                     double scale = 1.0, bool ntt = true, int components = 2);

} // namespace fhegpu
