#include "runtime/thread_trace.hpp"

#include <nlohmann/json.hpp>

#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

} // namespace

int main() {
  try {
    const auto nonce =
        std::chrono::steady_clock::now().time_since_epoch().count();
    const auto output_path =
        std::filesystem::temp_directory_path() /
        ("ckks-runtime-thread-trace-" + std::to_string(nonce) + ".json");
    if (::setenv("POSEIDON_THREAD_TRACE", output_path.string().c_str(), 1) != 0)
      throw std::runtime_error("setenv failed");

    std::mutex mutex;
    std::condition_variable condition;
    bool ready = false;
    std::thread worker([&] {
      fhegpu::ThreadTrace::set_thread_name("trace-test-worker");
      fhegpu::ThreadTraceUniqueLock lock(mutex, "test.mutex");
      lock.wait(condition, [&] { return ready; }, "test.condition_wait");
      fhegpu::ThreadTrace::record_deferred_duration(
          "test.deferred", &mutex, 3, fhegpu::ThreadTrace::timestamp_ns(),
          fhegpu::ThreadTrace::timestamp_ns(), [] { return 1234; });
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    {
      fhegpu::ThreadTraceLockGuard lock(mutex, "test.mutex");
      ready = true;
    }
    condition.notify_all();
    worker.join();

    fhegpu::ThreadTrace::write_json(0);
    std::ifstream input(output_path);
    require(static_cast<bool>(input), "thread trace JSON was not written");
    const nlohmann::json trace = nlohmann::json::parse(input);
    require(trace.at("format_version") == 1, "wrong trace format version");

    bool found_worker = false;
    for (const auto &thread : trace.at("threads")) {
      if (thread.at("name") != "trace-test-worker")
        continue;
      found_worker = true;
      const auto &metrics = thread.at("metrics");
      require(metrics.at("test.condition_wait").at("count") == 1,
              "condition wait metric is missing");
      require(metrics.at("test.deferred").at("total_ns") == 1234,
              "deferred duration was not resolved");
      require(!thread.at("mutex_events").empty(),
              "worker mutex events are missing");
    }
    require(found_worker, "worker thread trace is missing");
    std::filesystem::remove(output_path);
    std::cout << "thread trace test passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "thread trace test failed: " << error.what() << '\n';
    return 1;
  }
}
