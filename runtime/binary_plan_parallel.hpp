#pragma once

#include "runtime/binary_plan_codec.hpp"

#include <atomic>
#include <exception>
#include <mutex>
#include <thread>

namespace fhegpu::binary_io {

// Scan lengths without constructing operations; every field is validated by
// the existing decoder when the indexed block is materialized.
inline void skip_sequence(Archive<true> &a, std::size_t width) {
    std::uint64_t count = 0;
    a.number(count);
    a.check_count(count, width);
    a.skip(count * width);
}
inline void skip_instruction(Archive<true> &a) {
    a.skip(8); // ordinal
    std::uint8_t tag = 0;
    a.number(tag);
    switch (tag) {
    case 0: {
        a.skip(8); // output
        std::uint8_t payload = 0;
        a.number(payload);
        if (payload == 0) skip_sequence(a, 8);
        else if (payload == 1) a.skip(4);
        else throw std::runtime_error("unknown binary payload tag");
        break;
    }
    case 1: {
        a.skip(1); // compute kind
        skip_sequence(a, 8);
        a.skip(8 + 9); // output, place
        std::uint8_t attrs = 0;
        a.number(attrs);
        constexpr std::size_t widths[] = {0, 4, 8, 4, 17};
        if (attrs > 4) throw std::runtime_error("unknown binary compute attrs");
        a.skip(widths[attrs]);
        bool reuse = false;
        a.boolean(reuse);
        if (reuse) a.skip(8);
        break;
    }
    case 2:
        a.skip(8 + 1 + 1); // transfer id, kind, hint
        skip_sequence(a, 8);
        skip_sequence(a, 8);
        skip_sequence(a, 9);
        skip_sequence(a, 9);
        skip_sequence(a, 1);
        break;
    case 3: a.skip(8); break;
    case 4: break;
    default: throw std::runtime_error("unknown binary instruction tag");
    }
}

inline RuntimePlan read_plan_parallel(std::string_view bytes, unsigned threads,
                                      double *scan_seconds = nullptr, double *decode_seconds = nullptr,
                                      const char *magic = "CKKSPL01") {
    if (threads == 0 || threads > 64) throw std::runtime_error("threads must be between 1 and 64");
    const auto scan_start = std::chrono::steady_clock::now();
    Archive<true> a(bytes);
    a.magic(magic);
    std::string root;
    a.text(root);
    auto p = RuntimePlanJsonReader::read_text(root).plan;
    Strings strings;
    a.sequence(strings.values, 4, [&](std::string &s) { a.text(s); });
    struct Job {
        std::size_t begin, count;
        std::string_view bytes;
        std::vector<Instruction> *instructions; // nullptr denotes ValueDesc
    };
    std::vector<Job> jobs;
    constexpr std::size_t block_records = 16384;
    const auto count_array = [&](std::size_t width, std::size_t typed_size) {
        std::uint64_t count = 0;
        a.number(count);
        a.check_count(count, width);
        if (count > (8ULL << 30) / typed_size) throw std::runtime_error("invalid binary array length");
        return static_cast<std::size_t>(count);
    };
    const auto values_count = count_array(35, sizeof(ValueDesc));
    p.values.resize(values_count);
    for (std::size_t begin = 0; begin < values_count; begin += block_records) {
        const auto count = std::min(block_records, values_count - begin);
        const auto *start = a.cursor();
        a.skip(count * 35);
        jobs.push_back({begin, count, {start, count * 35}, nullptr});
    }
    a.sequence(p.external_inputs, 8, [&](ValueId &id) { a.number(id); });
    for (auto *phase : {&p.initialization, &p.execution, &p.finalization}) {
        const auto size = count_array(9, sizeof(Instruction));
        phase->resize(size);
        for (std::size_t begin = 0; begin < size; begin += block_records) {
            const auto count = std::min(block_records, size - begin);
            const auto *start = a.cursor();
            for (std::size_t i = 0; i < count; ++i) skip_instruction(a);
            jobs.push_back({begin, count, {start, static_cast<std::size_t>(a.cursor() - start)}, phase});
        }
    }
    a.sequence(p.final_outputs, 8, [&](ValueId &id) { a.number(id); });
    a.finish();
    const auto decode_start = std::chrono::steady_clock::now();
    if (scan_seconds) *scan_seconds = std::chrono::duration<double>(decode_start - scan_start).count();
    std::atomic<std::size_t> next{0};
    std::atomic<bool> failed{false};
    std::exception_ptr error;
    std::mutex error_mutex;
    const auto worker = [&] {
        try {
            while (!failed.load(std::memory_order_relaxed)) {
                const auto index = next.fetch_add(1, std::memory_order_relaxed);
                if (index >= jobs.size()) break;
                const auto &job = jobs[index];
                Archive<true> block(job.bytes, &strings);
                for (std::size_t i = job.begin; i < job.begin + job.count; ++i)
                    if (job.instructions) instruction(block, (*job.instructions)[i], p.format_version);
                    else value_record(block, p.values[i]);
                block.finish();
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(error_mutex);
            if (!error) error = std::current_exception();
            failed.store(true, std::memory_order_relaxed);
        }
    };
    std::vector<std::thread> workers;
    // Join successfully created workers even if thread creation fails.
    try {
        for (unsigned n = 1; n < threads; ++n) workers.emplace_back(worker);
    } catch (...) {
        failed.store(true, std::memory_order_relaxed);
        for (auto &thread : workers) thread.join();
        throw;
    }
    worker();
    for (auto &thread : workers) thread.join();
    if (error) std::rethrow_exception(error);
    if (decode_seconds) *decode_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - decode_start).count();
    return p;
}

} // namespace fhegpu::binary_io
