#include "runtime/thread_trace.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <thread>

namespace fhegpu {
namespace {

using Json = nlohmann::json;

struct MutexEvent {
  std::string name;
  std::uintptr_t instance = 0;
  std::uint64_t request_ns = 0;
  std::uint64_t acquired_ns = 0;
  std::uint64_t released_ns = 0;
};

struct DurationEvent {
  std::string name;
  std::uintptr_t instance = 0;
  std::uint64_t begin_ns = 0;
  std::uint64_t end_ns = 0;
};

struct DeferredDurationEvent {
  std::string name;
  std::uintptr_t instance = 0;
  int device = -1;
  std::uint64_t submit_begin_ns = 0;
  std::uint64_t submit_end_ns = 0;
  ThreadTrace::DeferredDurationResolver resolver;
};

struct ThreadBuffer {
  explicit ThreadBuffer(std::uint64_t id_value) : id(id_value) {
    mutex_events.reserve(4096);
    duration_events.reserve(1024);
    deferred_duration_events.reserve(1024);
  }

  std::uint64_t id = 0;
  std::string name;
  std::vector<MutexEvent> mutex_events;
  std::vector<DurationEvent> duration_events;
  std::vector<DeferredDurationEvent> deferred_duration_events;
};

struct DurationStats {
  std::uint64_t count = 0;
  std::uint64_t total_ns = 0;
  std::uint64_t min_ns = std::numeric_limits<std::uint64_t>::max();
  std::uint64_t max_ns = 0;

  void observe(std::uint64_t duration_ns) {
    ++count;
    total_ns += duration_ns;
    min_ns = std::min(min_ns, duration_ns);
    max_ns = std::max(max_ns, duration_ns);
  }
};

struct Collector {
  Collector()
      : enabled([] {
          const char *value = std::getenv("POSEIDON_THREAD_TRACE");
          return value != nullptr && value[0] != '\0' &&
                 std::string_view(value) != "0";
        }()),
        origin_ns(ThreadTrace::timestamp_ns()) {
    if (enabled)
      output_path = std::getenv("POSEIDON_THREAD_TRACE");
  }

  std::shared_ptr<ThreadBuffer> register_thread() {
    auto buffer = std::make_shared<ThreadBuffer>(static_cast<std::uint64_t>(
        std::hash<std::thread::id>{}(std::this_thread::get_id())));
    std::lock_guard<std::mutex> lock(registry_mutex);
    buffers.push_back(buffer);
    return buffer;
  }

  bool enabled = false;
  std::string output_path;
  std::uint64_t origin_ns = 0;
  std::mutex registry_mutex;
  std::vector<std::shared_ptr<ThreadBuffer>> buffers;
};

Collector &collector() {
  static Collector instance;
  return instance;
}

std::shared_ptr<ThreadBuffer> thread_buffer() {
  thread_local std::shared_ptr<ThreadBuffer> buffer =
      collector().register_thread();
  return buffer;
}

std::filesystem::path output_path_for_rank(std::string path, int rank) {
  if (path == "1")
    path = "/tmp/poseidon-thread-trace.%r.json";
  const std::string rank_text = std::to_string(rank);
  std::size_t offset = 0;
  bool replaced = false;
  while ((offset = path.find("%r", offset)) != std::string::npos) {
    path.replace(offset, 2, rank_text);
    offset += rank_text.size();
    replaced = true;
  }
  if (!replaced && rank != 0) {
    const std::filesystem::path original(path);
    const std::string ranked = original.stem().string() + ".rank" + rank_text +
                               original.extension().string();
    return original.parent_path() / ranked;
  }
  return path;
}

std::uint64_t relative_ns(std::uint64_t timestamp, std::uint64_t origin) {
  return timestamp >= origin ? timestamp - origin : 0;
}

Json stats_json(const DurationStats &stats) {
  return {
      {"count", stats.count},
      {"total_ns", stats.total_ns},
      {"average_ns", stats.count == 0 ? 0 : stats.total_ns / stats.count},
      {"min_ns", stats.count == 0 ? 0 : stats.min_ns},
      {"max_ns", stats.max_ns},
  };
}

} // namespace

bool ThreadTrace::enabled() { return collector().enabled; }

std::uint64_t ThreadTrace::timestamp_ns() noexcept {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

void ThreadTrace::set_thread_name(std::string name) {
  if (!enabled())
    return;
  thread_buffer()->name = std::move(name);
}

void ThreadTrace::record_mutex(const char *name, const void *instance,
                               std::uint64_t request_ns,
                               std::uint64_t acquired_ns,
                               std::uint64_t released_ns) {
  if (!enabled())
    return;
  thread_buffer()->mutex_events.push_back(
      {name, reinterpret_cast<std::uintptr_t>(instance), request_ns,
       acquired_ns, released_ns});
}

void ThreadTrace::record_duration(const char *name, const void *instance,
                                  std::uint64_t begin_ns,
                                  std::uint64_t end_ns) {
  if (!enabled())
    return;
  thread_buffer()->duration_events.push_back(
      {name, reinterpret_cast<std::uintptr_t>(instance), begin_ns, end_ns});
}

void ThreadTrace::record_deferred_duration(const char *name,
                                           const void *instance, int device,
                                           std::uint64_t submit_begin_ns,
                                           std::uint64_t submit_end_ns,
                                           DeferredDurationResolver resolver) {
  if (!enabled())
    return;
  thread_buffer()->deferred_duration_events.push_back(
      {name, reinterpret_cast<std::uintptr_t>(instance), device,
       submit_begin_ns, submit_end_ns, std::move(resolver)});
}

void ThreadTrace::write_json(int rank) {
  auto &state = collector();
  if (!state.enabled)
    return;

  std::vector<std::shared_ptr<ThreadBuffer>> buffers;
  {
    std::lock_guard<std::mutex> lock(state.registry_mutex);
    buffers = state.buffers;
  }

  Json threads = Json::array();
  for (const auto &buffer : buffers) {
    if (buffer->mutex_events.empty() && buffer->duration_events.empty() &&
        buffer->deferred_duration_events.empty())
      continue;

    std::map<std::string, DurationStats> metrics;
    Json mutex_events = Json::array();
    for (const auto &event : buffer->mutex_events) {
      const std::uint64_t wait_ns = event.acquired_ns - event.request_ns;
      const std::uint64_t hold_ns = event.released_ns - event.acquired_ns;
      metrics["mutex." + event.name + ".wait"].observe(wait_ns);
      metrics["mutex." + event.name + ".hold"].observe(hold_ns);
      mutex_events.push_back({
          {"name", event.name},
          {"instance", event.instance},
          {"request_ns", relative_ns(event.request_ns, state.origin_ns)},
          {"acquired_ns", relative_ns(event.acquired_ns, state.origin_ns)},
          {"released_ns", relative_ns(event.released_ns, state.origin_ns)},
          {"wait_ns", wait_ns},
          {"hold_ns", hold_ns},
      });
    }

    Json duration_events = Json::array();
    for (const auto &event : buffer->duration_events) {
      const std::uint64_t duration_ns = event.end_ns - event.begin_ns;
      metrics[event.name].observe(duration_ns);
      duration_events.push_back({
          {"name", event.name},
          {"instance", event.instance},
          {"begin_ns", relative_ns(event.begin_ns, state.origin_ns)},
          {"end_ns", relative_ns(event.end_ns, state.origin_ns)},
          {"duration_ns", duration_ns},
      });
    }

    Json deferred_events = Json::array();
    for (auto &event : buffer->deferred_duration_events) {
      const std::uint64_t duration_ns = event.resolver();
      metrics[event.name].observe(duration_ns);
      deferred_events.push_back({
          {"name", event.name},
          {"instance", event.instance},
          {"device", event.device},
          {"submit_begin_ns",
           relative_ns(event.submit_begin_ns, state.origin_ns)},
          {"submit_end_ns", relative_ns(event.submit_end_ns, state.origin_ns)},
          {"duration_ns", duration_ns},
      });
    }

    Json metric_json = Json::object();
    for (const auto &item : metrics)
      metric_json[item.first] = stats_json(item.second);

    threads.push_back({
        {"tid", buffer->id},
        {"name", buffer->name.empty() ? "thread-" + std::to_string(buffer->id)
                                      : buffer->name},
        {"metrics", std::move(metric_json)},
        {"mutex_events", std::move(mutex_events)},
        {"duration_events", std::move(duration_events)},
        {"deferred_duration_events", std::move(deferred_events)},
    });

    buffer->mutex_events.clear();
    buffer->duration_events.clear();
    buffer->deferred_duration_events.clear();
  }

  const std::filesystem::path path =
      output_path_for_rank(state.output_path, rank);
  if (!path.parent_path().empty())
    std::filesystem::create_directories(path.parent_path());
  std::ofstream output(path);
  if (!output)
    throw std::runtime_error("cannot write thread trace JSON: " +
                             path.string());
  output << Json{
                  {"format_version", 1},
                  {"clock", "steady_clock"},
                  {"rank", rank},
                  {"threads", std::move(threads)},
              }
                   .dump(2)
           << '\n';
  state.origin_ns = timestamp_ns();
}

} // namespace fhegpu
