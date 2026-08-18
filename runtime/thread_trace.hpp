#pragma once

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fhegpu {

class ThreadTrace {
public:
  using DeferredDurationResolver = std::function<std::uint64_t()>;

  static bool enabled();
  static std::uint64_t timestamp_ns() noexcept;
  static void set_thread_name(std::string name);

  static void record_mutex(const char *name, const void *instance,
                           std::uint64_t request_ns, std::uint64_t acquired_ns,
                           std::uint64_t released_ns);
  static void record_duration(const char *name, const void *instance,
                              std::uint64_t begin_ns, std::uint64_t end_ns);
  static void record_deferred_duration(const char *name, const void *instance,
                                       int device,
                                       std::uint64_t submit_begin_ns,
                                       std::uint64_t submit_end_ns,
                                       DeferredDurationResolver resolver);

  static void write_json(int rank);
};

class ThreadTraceLockGuard {
public:
  ThreadTraceLockGuard(std::mutex &mutex, const char *name)
      : mutex_(mutex), name_(name), traced_(ThreadTrace::enabled()) {
    if (traced_)
      request_ns_ = ThreadTrace::timestamp_ns();
    mutex_.lock();
    if (traced_)
      acquired_ns_ = ThreadTrace::timestamp_ns();
  }

  ~ThreadTraceLockGuard() {
    if (!traced_) {
      mutex_.unlock();
      return;
    }
    const std::uint64_t released_ns = ThreadTrace::timestamp_ns();
    mutex_.unlock();
    ThreadTrace::record_mutex(name_, &mutex_, request_ns_, acquired_ns_,
                              released_ns);
  }

  ThreadTraceLockGuard(const ThreadTraceLockGuard &) = delete;
  ThreadTraceLockGuard &operator=(const ThreadTraceLockGuard &) = delete;

private:
  std::mutex &mutex_;
  const char *name_;
  bool traced_ = false;
  std::uint64_t request_ns_ = 0;
  std::uint64_t acquired_ns_ = 0;
};

class ThreadTraceUniqueLock {
public:
  ThreadTraceUniqueLock(std::mutex &mutex, const char *name)
      : mutex_(&mutex), name_(name), traced_(ThreadTrace::enabled()) {
    lock();
  }

  ~ThreadTraceUniqueLock() {
    if (owns_lock_)
      unlock();
  }

  ThreadTraceUniqueLock(const ThreadTraceUniqueLock &) = delete;
  ThreadTraceUniqueLock &operator=(const ThreadTraceUniqueLock &) = delete;

  void lock() {
    if (owns_lock_)
      throw std::logic_error("ThreadTraceUniqueLock already owns its mutex");
    if (traced_)
      request_ns_ = ThreadTrace::timestamp_ns();
    mutex_->lock();
    owns_lock_ = true;
    if (traced_)
      acquired_ns_ = ThreadTrace::timestamp_ns();
  }

  void unlock() {
    if (!owns_lock_)
      throw std::logic_error("ThreadTraceUniqueLock does not own its mutex");
    const std::uint64_t released_ns = traced_ ? ThreadTrace::timestamp_ns() : 0;
    mutex_->unlock();
    owns_lock_ = false;
    if (traced_)
      ThreadTrace::record_mutex(name_, mutex_, request_ns_, acquired_ns_,
                                released_ns);
  }

  bool owns_lock() const noexcept { return owns_lock_; }

  template <class Predicate>
  void wait(std::condition_variable &condition, Predicate predicate,
            const char *wait_name) {
    if (!owns_lock_)
      throw std::logic_error(
          "ThreadTraceUniqueLock wait requires an owned mutex");
    if (!traced_) {
      std::unique_lock<std::mutex> native(*mutex_, std::adopt_lock);
      condition.wait(native, std::move(predicate));
      native.release();
      return;
    }

    const std::uint64_t released_ns = ThreadTrace::timestamp_ns();
    std::unique_lock<std::mutex> native(*mutex_, std::adopt_lock);
    const std::uint64_t wait_begin_ns = released_ns;
    try {
      condition.wait(native, std::move(predicate));
    } catch (...) {
      native.release();
      const std::uint64_t wait_end_ns = ThreadTrace::timestamp_ns();
      ThreadTrace::record_mutex(name_, mutex_, request_ns_, acquired_ns_,
                                released_ns);
      ThreadTrace::record_duration(wait_name, mutex_, wait_begin_ns,
                                   wait_end_ns);
      request_ns_ = wait_end_ns;
      acquired_ns_ = wait_end_ns;
      throw;
    }
    native.release();
    const std::uint64_t wait_end_ns = ThreadTrace::timestamp_ns();
    ThreadTrace::record_mutex(name_, mutex_, request_ns_, acquired_ns_,
                              released_ns);
    ThreadTrace::record_duration(wait_name, mutex_, wait_begin_ns, wait_end_ns);
    request_ns_ = wait_end_ns;
    acquired_ns_ = wait_end_ns;
  }

private:
  std::mutex *mutex_ = nullptr;
  const char *name_ = nullptr;
  bool traced_ = false;
  bool owns_lock_ = false;
  std::uint64_t request_ns_ = 0;
  std::uint64_t acquired_ns_ = 0;
};

} // namespace fhegpu
