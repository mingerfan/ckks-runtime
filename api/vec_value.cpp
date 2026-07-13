#include "api/vec_value.hpp"

#include <cmath>
#include <stdexcept>

namespace fhegpu {

bool operator==(const VecMetadata &a, const VecMetadata &b) {
    return a.context == b.context && a.degree == b.degree && a.level == b.level &&
           a.scale == b.scale && a.ntt == b.ntt && a.components == b.components;
}

VecValue VecValue::ready(VecPayload payload) {
    auto state = std::make_shared<State>(payload.kind);
    state->payload = std::move(payload);
    state->ready = true;
    return VecValue(std::move(state));
}

VecValue VecValue::pending(ValueKind kind) { return VecValue(std::make_shared<State>(kind)); }

ValueKind VecValue::kind() const {
    if (!state_) throw std::runtime_error("empty VecValue");
    return state_->expected_kind;
}

VecPayload VecValue::materialize() const {
    if (!state_) throw std::runtime_error("empty VecValue");
    std::unique_lock<std::mutex> lock(state_->mutex);
    state_->cv.wait(lock, [&] { return state_->ready || state_->error; });
    if (state_->error) std::rethrow_exception(state_->error);
    return state_->payload;
}

VecValue VecValue::deep_copy() const { return ready(materialize()); }

void VecValue::fulfill(VecPayload payload) const {
    if (!state_) throw std::runtime_error("empty VecValue");
    if (payload.kind != state_->expected_kind) throw std::runtime_error("VecValue fulfilled with wrong kind");
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        if (state_->ready || state_->error) throw std::runtime_error("VecValue completed twice");
        state_->payload = std::move(payload);
        state_->ready = true;
    }
    state_->cv.notify_all();
}

void VecValue::fail(std::exception_ptr error) const {
    if (!state_) throw std::runtime_error("empty VecValue");
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        if (state_->ready || state_->error) throw std::runtime_error("VecValue completed twice");
        state_->error = std::move(error);
    }
    state_->cv.notify_all();
}

const void *VecValue::identity() const { return state_.get(); }

static VecValue make_value(ValueKind kind, std::vector<double> slots, std::string context,
                           std::uint64_t degree, int level, double scale, bool ntt, int components) {
    if (slots.empty()) throw std::runtime_error("VecValue requires at least one slot");
    if (context.empty() || degree == 0 || level < 0 || !std::isfinite(scale) || scale <= 0.0)
        throw std::runtime_error("invalid VecValue metadata");
    if (kind == ValueKind::Ciphertext && components < 2) throw std::runtime_error("ciphertext requires at least two components");
    if (kind == ValueKind::Plaintext) components = 1;
    return VecValue::ready(VecPayload{kind, std::move(slots), VecMetadata{std::move(context), degree, level, scale, ntt, components}});
}

VecValue make_plain(std::vector<double> slots, std::string context, std::uint64_t degree,
                    int level, double scale, bool ntt) {
    return make_value(ValueKind::Plaintext, std::move(slots), std::move(context), degree, level, scale, ntt, 1);
}
VecValue make_cipher(std::vector<double> slots, std::string context, std::uint64_t degree,
                     int level, double scale, bool ntt, int components) {
    return make_value(ValueKind::Ciphertext, std::move(slots), std::move(context), degree, level, scale, ntt, components);
}

} // namespace fhegpu
