#include "api/vec_api.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>

namespace fhegpu {
namespace {

void require_compatible(const VecPayload &a, const VecPayload &b, bool check_scale) {
    if (a.slots.size() != b.slots.size()) throw std::runtime_error("slot count mismatch");
    if (a.metadata.context != b.metadata.context || a.metadata.degree != b.metadata.degree ||
        a.metadata.level != b.metadata.level || a.metadata.ntt != b.metadata.ntt)
        throw std::runtime_error("incompatible VecValue metadata");
    if (check_scale && a.metadata.scale_log2 != b.metadata.scale_log2) throw std::runtime_error("scale mismatch");
}

VecPayload unary_cipher(const std::vector<VecValue> &inputs) {
    if (inputs.size() != 1 || inputs[0].kind() != ValueKind::Ciphertext) throw std::runtime_error("unary cipher operation type mismatch");
    return inputs[0].materialize();
}

} // namespace

VecExecutor::VecExecutor(VecExecConfig config) : config_(config), random_(config.delay_seed) {
    if (config_.max_delay_ms < 0) throw std::runtime_error("negative Vec task delay");
}

VecExecutor::~VecExecutor() { stop(); }

int VecExecutor::next_delay_ms() {
    if (config_.max_delay_ms == 0) return 0;
    std::lock_guard<std::mutex> lock(random_mutex_);
    return std::uniform_int_distribution<int>(0, config_.max_delay_ms)(random_);
}

VecExecutor::Worker &VecExecutor::worker_for(const Place &place) {
    std::lock_guard<std::mutex> lock(workers_mutex_);
    if (stopped_) throw std::runtime_error("VecExecutor is stopped");
    auto &slot = workers_[place];
    if (!slot) {
        slot = std::make_unique<Worker>();
        Worker *worker = slot.get();
        worker->thread = std::thread([worker] {
            for (;;) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(worker->mutex);
                    worker->cv.wait(lock, [&] { return worker->stopping || !worker->tasks.empty(); });
                    if (worker->stopping && worker->tasks.empty()) return;
                    task = std::move(worker->tasks.front());
                    worker->tasks.pop_front();
                }
                task();
            }
        });
    }
    return *slot;
}

VecValue VecExecutor::compute(const ComputeOp &op, const std::vector<VecValue> &inputs) {
    if (config_.mode == VecExecMode::Sync) return VecValue::ready(compute_now(op, inputs));
    VecValue output = VecValue::pending(ValueKind::Ciphertext, compute_metadata(op, inputs));
    Worker &worker = worker_for(op.place);
    {
        std::lock_guard<std::mutex> lock(worker.mutex);
        worker.tasks.emplace_back([this, op, inputs, output] {
            try {
                const int delay = next_delay_ms();
                if (delay) std::this_thread::sleep_for(std::chrono::milliseconds(delay));
                output.fulfill(compute_now(op, inputs));
            } catch (...) { output.fail(std::current_exception()); }
        });
    }
    worker.cv.notify_one();
    return output;
}

VecMetadata VecExecutor::compute_metadata(const ComputeOp &op, const std::vector<VecValue> &inputs) const {
    const auto unary = [&] {
        if (inputs.size() != 1 || inputs[0].kind() != ValueKind::Ciphertext)
            throw std::runtime_error("unary cipher operation type mismatch");
        return inputs[0].metadata();
    };
    const auto binary = [&](ValueKind rhs_kind, bool multiply) {
        if (inputs.size() != 2 || inputs[0].kind() != ValueKind::Ciphertext || inputs[1].kind() != rhs_kind)
            throw std::runtime_error("binary operation type mismatch");
        VecMetadata a = inputs[0].metadata();
        const VecMetadata b = inputs[1].metadata();
        if (a.context != b.context || a.degree != b.degree || a.level != b.level || a.ntt != b.ntt)
            throw std::runtime_error("incompatible VecValue metadata");
        if (!multiply && a.scale_log2 != b.scale_log2) throw std::runtime_error("scale mismatch");
        if (multiply) {
            if (b.scale_log2 > std::numeric_limits<int>::max() - a.scale_log2)
                throw std::runtime_error("multiplication scale overflow");
            a.scale_log2 += b.scale_log2;
            if (rhs_kind == ValueKind::Ciphertext) a.components += b.components - 1;
        }
        return a;
    };
    switch (op.kind) {
    case ComputeKind::AddCC: case ComputeKind::SubCC: return binary(ValueKind::Ciphertext, false);
    case ComputeKind::AddCP: case ComputeKind::SubCP: return binary(ValueKind::Plaintext, false);
    case ComputeKind::MulCC: return binary(ValueKind::Ciphertext, true);
    case ComputeKind::MulCP: return binary(ValueKind::Plaintext, true);
    case ComputeKind::Negate: case ComputeKind::Rotate: return unary();
    case ComputeKind::Rescale: {
        auto metadata = unary();
        const auto attrs = std::get<RescaleAttrs>(op.attrs);
        if (attrs.target_level >= metadata.level) throw std::runtime_error("rescale must lower level");
        metadata.level = attrs.target_level;
        metadata.scale_log2 = attrs.target_scale_log2;
        return metadata;
    }
    case ComputeKind::ModSwitch: {
        auto metadata = unary();
        const int target = std::get<ModSwitchAttrs>(op.attrs).target_level;
        if (target >= metadata.level) throw std::runtime_error("modswitch must lower level");
        metadata.level = target;
        return metadata;
    }
    case ComputeKind::Relinearize: {
        auto metadata = unary();
        if (metadata.components <= 2) throw std::runtime_error("relinearize requires more than two components");
        metadata.components = 2;
        return metadata;
    }
    case ComputeKind::Boot: {
        auto metadata = unary();
        const auto attrs = std::get<BootAttrs>(op.attrs);
        metadata.level = attrs.target_level;
        metadata.scale_log2 = attrs.target_scale_log2;
        metadata.components = attrs.target_components;
        return metadata;
    }
    }
    throw std::runtime_error("unsupported compute operation");
}

VecPayload VecExecutor::compute_now(const ComputeOp &op, const std::vector<VecValue> &inputs) {
    const auto binary = [&](ValueKind rhs_kind, const auto &fn, bool multiply) {
        if (inputs.size() != 2 || inputs[0].kind() != ValueKind::Ciphertext || inputs[1].kind() != rhs_kind)
            throw std::runtime_error("binary operation type mismatch");
        VecPayload a = inputs[0].materialize();
        VecPayload b = inputs[1].materialize();
        require_compatible(a, b, !multiply);
        for (std::size_t i = 0; i < a.slots.size(); ++i) a.slots[i] = fn(a.slots[i], b.slots[i]);
        if (multiply) {
            if (b.metadata.scale_log2 > std::numeric_limits<int>::max() - a.metadata.scale_log2)
                throw std::runtime_error("multiplication scale overflow");
            a.metadata.scale_log2 += b.metadata.scale_log2;
            if (rhs_kind == ValueKind::Ciphertext) a.metadata.components += b.metadata.components - 1;
        }
        return a;
    };

    switch (op.kind) {
    case ComputeKind::AddCC: return binary(ValueKind::Ciphertext, [](double a, double b) { return a + b; }, false);
    case ComputeKind::AddCP: return binary(ValueKind::Plaintext, [](double a, double b) { return a + b; }, false);
    case ComputeKind::SubCC: return binary(ValueKind::Ciphertext, [](double a, double b) { return a - b; }, false);
    case ComputeKind::SubCP: return binary(ValueKind::Plaintext, [](double a, double b) { return a - b; }, false);
    case ComputeKind::MulCC: return binary(ValueKind::Ciphertext, [](double a, double b) { return a * b; }, true);
    case ComputeKind::MulCP: return binary(ValueKind::Plaintext, [](double a, double b) { return a * b; }, true);
    case ComputeKind::Negate: {
        auto a = unary_cipher(inputs); for (double &x : a.slots) x = -x; return a;
    }
    case ComputeKind::Rotate: {
        auto a = unary_cipher(inputs);
        const int steps = std::get<RotateAttrs>(op.attrs).steps;
        const long long n = static_cast<long long>(a.slots.size());
        long long shift = static_cast<long long>(steps) % n; if (shift < 0) shift += n;
        std::rotate(a.slots.begin(), a.slots.begin() + shift, a.slots.end()); return a;
    }
    case ComputeKind::Rescale: {
        auto a = unary_cipher(inputs); const auto attrs = std::get<RescaleAttrs>(op.attrs);
        if (attrs.target_level >= a.metadata.level) throw std::runtime_error("rescale must lower level");
        if (attrs.target_scale_log2 < 0) throw std::runtime_error("invalid rescaled scale");
        a.metadata.level = attrs.target_level;
        a.metadata.scale_log2 = attrs.target_scale_log2;
        return a;
    }
    case ComputeKind::ModSwitch: {
        auto a = unary_cipher(inputs); const int target = std::get<ModSwitchAttrs>(op.attrs).target_level;
        if (target >= a.metadata.level) throw std::runtime_error("modswitch must lower level");
        a.metadata.level = target; return a;
    }
    case ComputeKind::Relinearize: {
        auto a = unary_cipher(inputs); if (a.metadata.components <= 2) throw std::runtime_error("relinearize requires more than two components");
        a.metadata.components = 2; return a;
    }
    case ComputeKind::Boot: {
        auto a = unary_cipher(inputs); const auto attrs = std::get<BootAttrs>(op.attrs);
        a.metadata.level = attrs.target_level;
        a.metadata.scale_log2 = attrs.target_scale_log2;
        a.metadata.components = attrs.target_components;
        return a;
    }
    }
    throw std::runtime_error("unsupported compute operation");
}

void VecExecutor::stop() {
    std::map<Place, std::unique_ptr<Worker>> workers;
    {
        std::lock_guard<std::mutex> lock(workers_mutex_);
        if (stopped_) return;
        stopped_ = true;
        workers.swap(workers_);
    }
    for (auto &item : workers) { std::lock_guard<std::mutex> lock(item.second->mutex); item.second->stopping = true; item.second->cv.notify_all(); }
    for (auto &item : workers) if (item.second->thread.joinable()) item.second->thread.join();
}

} // namespace fhegpu
