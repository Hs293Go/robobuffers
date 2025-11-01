
#pragma once

#include <atomic>
#include <concepts>
#include <cstddef>
#include <functional>
#include <future>
#include <new>
#include <semaphore>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef __has_include
#if __has_include(<version>)
#include <version>
#endif
#endif

// moodycamel
#include "third_party/concurrentqueue.h"  // https://github.com/cameron314/concurrentqueue

namespace robo {

// Choose a default callable type (move-only when available).
namespace details {
#if defined(__cpp_lib_move_only_function) && \
    __cpp_lib_move_only_function >= 202110L
using fn_t = std::move_only_function<void()>;
#else
using fn_t = std::function<void()>;
#endif
}  // namespace details

/**
 * A minimal, high-throughput thread pool optimized for:
 *  - short-lived tasks (ZMQ message handlers, telemetry parsing)
 *  - high enqueue/dequeue rates
 *  - low wake-up latency
 *
 * Design:
 *  - Single global MPMC queue (moodycamel::ConcurrentQueue)
 *  - counting_semaphore for worker parking
 *  - in_flight counter with atomic wait/notify for wait_for_tasks()
 *  - jthread + stop_token for clean shutdown
 */
template <typename FunctionType = details::fn_t>
  requires std::invocable<FunctionType> &&
           std::is_same_v<void, std::invoke_result_t<FunctionType>>
class ThreadPoolMpmc {
 public:
  using init_fn =
      void (*)(std::size_t);  // optional per-thread init; can be nullptr

  explicit ThreadPoolMpmc(
      std::size_t thread_count = std::thread::hardware_concurrency(),
      init_fn on_thread_start = nullptr)
      : threads_(std::max<std::size_t>(1, thread_count)),
        sem_(0),
        stopping_(false),
        in_flight_(0) {
    for (std::size_t i = 0; i < threads_.size(); ++i) {
      threads_[i] =
          std::jthread([this, i, on_thread_start](std::stop_token st) {
            if (on_thread_start) {
              try {
                on_thread_start(i);
              } catch (...) {
                (void)0;
              }
            }
            worker_loop(st);
          });
    }
  }

  // non-copyable
  ThreadPoolMpmc(const ThreadPoolMpmc&) = delete;
  ThreadPoolMpmc& operator=(const ThreadPoolMpmc&) = delete;

  // movable (optional; easy to disable if you prefer)
  ThreadPoolMpmc(ThreadPoolMpmc&&) = delete;
  ThreadPoolMpmc& operator=(ThreadPoolMpmc&&) = delete;

  ~ThreadPoolMpmc() {
    // 1) finish queued work if you want: wait_for_tasks(); (optional)
    // Leave it to the user to call wait_for_tasks() if they care.
    // We will still drain until stop is requested below.

    // 2) request stop and wake all workers
    stopping_.store(true, std::memory_order_release);
    std::ranges::for_each(threads_, std::mem_fn(&std::jthread::request_stop));

    // Release enough permits so all blocked workers wake up.
    sem_.release(static_cast<int>(threads_.size()));

    // 3) join
    for (auto& t : threads_) {
      if (t.joinable()) {
        t.join();
      }
    }
  }

  [[nodiscard]] std::size_t size() const noexcept { return threads_.size(); }

  /**
   * Enqueue a task that returns a value; returns a std::future<T>.
   * The callable may be anything invocable with Args..., captured by move.
   */
  template <typename F, typename... Args>
    requires std::invocable<F, Args...>
  [[nodiscard]] auto enqueue(F&& f, Args&&... args) {
    using R = std::invoke_result_t<F, Args...>;

    // C++23: we can use move-only promise; pre-C++23: shared_ptr to promise
#if defined(__cpp_lib_move_only_function) && \
    __cpp_lib_move_only_function >= 202110L
    std::promise<R> prom;
    auto fut = prom.get_future();
    FunctionType task = [prom = std::move(prom), func = std::forward<F>(f),
                         ... capt = std::forward<Args>(args)]() mutable {
      try {
        if constexpr (std::is_void_v<R>) {
          std::invoke(func, capt...);
          prom.set_value();
        } else {
          prom.set_value(std::invoke(func, capt...));
        }
      } catch (...) {
        prom.set_exception(std::current_exception());
      }
    };
    push_task_(std::move(task));
    return fut;
#else
    auto sp = std::make_shared<std::promise<R>>();
    auto fut = sp->get_future();
    FunctionType task = [sp, func = std::forward<F>(f),
                         ... capt = std::forward<Args>(args)]() mutable {
      try {
        if constexpr (std::is_void_v<R>) {
          std::invoke(func, capt...);
          sp->set_value();
        } else {
          sp->set_value(std::invoke(func, capt...));
        }
      } catch (...) {
        sp->set_exception(std::current_exception());
      }
    };
    push_task(std::move(task));
    return fut;
#endif
  }

  /**
   * Enqueue a fire-and-forget task (return value ignored).
   */
  template <typename F, typename... Args>
    requires std::invocable<F, Args...>
  void enqueue_detach(F&& f, Args&&... args) {
    FunctionType task = [func = std::forward<F>(f),
                         ... capt = std::forward<Args>(args)]() mutable {
      try {
        std::invoke(func, capt...);
      } catch (...) {
        (void)0;
      }
    };
    push_task(std::move(task));
  }

  /**
   * Bulk enqueue (uses moodycamel bulk API when available).
   * The input must be a forward-iterable range of FunctionType (or
   * constructible).
   */
  template <typename Range>
    requires requires(Range& r) {
      std::begin(r);
      std::end(r);
    }
  void enqueue_bulk(Range&& tasks) {
    // count first (can also do single-pass with a small buffer)
    const std::size_t n = std::ranges::size(tasks);
    if (n == 0) {
      return;
    }

    in_flight_.fetch_add(static_cast<std::ptrdiff_t>(n),
                         std::memory_order_release);

    // Try bulk enqueue; if not contiguous, copy to a temporary buffer of
    // FunctionType. For generality, we allocate a temporary vector here
    // (usually small).
    std::unique_ptr<FunctionType[]> tmp = std::make_unique<FunctionType[]>(n);
    std::ranges::copy(tasks, tmp.begin());

    queue_.enqueue_bulk(tmp.begin(), n);

    sem_.release(static_cast<int>(n));
  }

  /**
   * Block until in-flight counter returns to zero (all tasks consumed +
   * completed). Note: this does not guarantee the global queue is strictly
   * empty at the instant you return, but the in_flight_ logic matches
   * enqueue/complete exactly, so it’s the right condition for "all submitted
   * work completed".
   */
  void wait_for_tasks() {
    for (;;) {
      auto cur = in_flight_.load(std::memory_order_acquire);
      if (cur == 0) {
        return;
      }
      in_flight_.wait(cur, std::memory_order_acquire);
    }
  }

  /**
   * Best-effort clearing of queued (not yet started) tasks.
   * Returns #tasks removed from the queue and decrements in_flight_
   * accordingly.
   */
  std::size_t clear_pending() {
    std::size_t removed = 0;
    FunctionType task;
    while (queue_.try_dequeue(task)) {
      ++removed;
      // balance in_flight_ as these tasks will never run
      in_flight_.fetch_sub(1, std::memory_order_release);
    }
    // No need to adjust the semaphore here; excess permits are fine for
    // workers, they will wake, fail dequeue, and loop (very low overhead). If
    // you want to tighten it, you could track an extra counter.
    return removed;
  }

 private:
  void worker_loop(const std::stop_token& st) {
    FunctionType task;

    while (true) {
      // Block until signalled (or stop requested).
      // Note: counting_semaphore::try_acquire_for could be used to bail out
      // faster under stop, but acquire() + checking stop periodically is fine
      // and faster.
      sem_.acquire();

      // Drain as much as possible on this wake-up to amortize wake cost.
      while (queue_.try_dequeue(task)) {
        // Run task
        try {
          std::invoke(std::move(task));
        } catch (...) { /* swallow */
          (void)0;
        }

        // This task is now complete
        const auto prev = in_flight_.fetch_sub(1, std::memory_order_release);
        if (prev == 1) {
          // we transitioned to zero: wake waiters
          in_flight_.notify_all();
        }

        if (st.stop_requested() && stopping_.load(std::memory_order_acquire)) {
          // Optional early exit after finishing what we took
          // Continue to consume the batch on this wake; next loop will exit
        }
      }

      if (st.stop_requested() && stopping_.load(std::memory_order_acquire)) {
        // We were asked to stop, and we’ve drained what we woke up for.
        // It’s safe to exit.
        break;
      }
    }
  }

  template <typename F>
  inline void push_task(F&& f) {
    in_flight_.fetch_add(1, std::memory_order_release);
    queue_.enqueue(std::forward<F>(f));
    sem_.release();
  }

 private:
  std::vector<std::jthread> threads_;
  moodycamel::ConcurrentQueue<FunctionType> queue_;
  std::counting_semaphore<> sem_;
  std::atomic<bool> stopping_;
  // Tracks "submitted but not yet fully executed" tasks.
  std::atomic<std::ptrdiff_t> in_flight_;
};

}  // namespace robo
