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

namespace xla {
namespace poplarplugin {

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

}  // namespace poplarplugin
}  // namespace xla
