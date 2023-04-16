/* Copyright (c) 2023 Graphcore Ltd. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#pragma once
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <type_traits>

#include "tensorflow/core/platform/default/casts.h"

namespace xla {
namespace poplarplugin {

/**
 * @brief Atomic counter (int64).
 */
class AtomicCounter {
 public:
  using CountType = int64_t;
  /** Initialize the counter. */
  AtomicCounter(CountType init_value = 0) : m_counter{init_value} {}
  /** Atomic increment. */
  CountType increment() noexcept {
    // TODO: memory order to use?
    return m_counter.fetch_add(1, std::memory_order_relaxed);
  }

 private:
  std::atomic<CountType> m_counter;
};

/**
 * @brief Downcast a unique pointer.
 */
template <typename T, typename S>
std::unique_ptr<T> unique_down_cast(std::unique_ptr<S> ptr) noexcept {
  return std::unique_ptr<T>(tensorflow::down_cast<T*>(ptr.release()));
}

/**
 * @brief Very basic thread safe queue!
 */
template <class T>
class ThreadSafeQueue {
 public:
  // Make sure T objects can be moved in/out of the queue without copy.
  static_assert(std::is_nothrow_move_constructible_v<T>);
  static_assert(std::is_nothrow_move_assignable_v<T>);

  ThreadSafeQueue() : m_queue(), m_mutex(), m_cv() {}
  ~ThreadSafeQueue() {}

  /**
   * @brief Add an element to the queue.
   * Pass by value to take ownership.
   */
  void enqueue(T t) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_queue.push(std::move(t));
    m_cv.notify_one();
  }

  /**
   * @brief Get the front element. Blocking if the queue is empty.
   */
  T dequeue() {
    std::unique_lock<std::mutex> lock(m_mutex);
    while (m_queue.empty()) {
      // release lock as long as the wait and reaquire it afterwards.
      m_cv.wait(lock);
    }
    T val = std::move(m_queue.front());
    m_queue.pop();
    return val;
  }

 private:
  std::queue<T> m_queue;
  mutable std::mutex m_mutex;
  std::condition_variable m_cv;
};

}  // namespace poplarplugin
}  // namespace xla
