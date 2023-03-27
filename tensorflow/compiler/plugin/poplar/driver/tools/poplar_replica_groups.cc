/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_replica_groups.h"

#include "tensorflow/compiler/plugin/poplar/driver/tools/hash.h"

namespace xla {
namespace poplarplugin {

/*static*/ PoplarReplicaGroups PoplarReplicaGroups::Consecutive(
    uint64 group_size) {
  // Treat 0 is a special case meaning all replicas in the same group.
  if (group_size == 0) {
    return PoplarReplicaGroups();
  }
  return PoplarReplicaGroups(group_size, Type::Consecutive);
}

/*static*/ PoplarReplicaGroups PoplarReplicaGroups::Orthogonal(
    uint64 group_size) {
  // Treat 0 is a special case meaning all replicas in the same group.
  if (group_size == 0) {
    return PoplarReplicaGroups();
  }
  return PoplarReplicaGroups(group_size, Type::Orthogonal);
}

uint64 PoplarReplicaGroups::GroupSizeOr(uint64 default_value) const {
  return group_size_.value_or(default_value);
}

uint64 PoplarReplicaGroups::GroupSizeOrDie() const {
  CHECK(group_size_.has_value());
  return *group_size_;
}

std::string PoplarReplicaGroups::ToString() const {
  if (!group_size_.has_value()) {
    return "single(group_size=all)";
  }
  switch (group_type_) {
    case Type::Consecutive:
      return "consecutive(group_size=" + std::to_string(*group_size_) + ")";
    case Type::Orthogonal:
      return "orthogonal(group_size=" + std::to_string(*group_size_) + ")";
  }
  LOG(FATAL) << "Unknown group type";
}

std::vector<xla::ReplicaGroup> PoplarReplicaGroups::ToXlaReplicaGroups() const {
  if (!group_size_.has_value()) {
    return {};
  }

  // We do not know the total number of replicas at this point, so we do not
  // know how many groups to generate. Therefore we generate only enough groups
  // to satisfy the HLO verifier. During lowering the correct number of groups
  // will be used based on the total number of replicas.
  const int64_t group_size = *group_size_;
  const int64_t num_groups = group_type_ == Type::Consecutive ? 1 : 2;

  std::vector<xla::ReplicaGroup> result(num_groups);
  for (int64_t i = 0; i < num_groups; ++i) {
    for (int64_t j = 0; j < group_size; ++j) {
      result[i].add_replica_ids(j * num_groups + i);
    }
  }
  return result;
}

/*static*/ xla::StatusOr<PoplarReplicaGroups>
PoplarReplicaGroups::FromXlaReplicaGroups(
    absl::Span<const xla::ReplicaGroup> groups) {
  if (groups.empty()) {
    return PoplarReplicaGroups();
  }
  // Readable replica groups visualization: helping error msgs.
  const std::string groups_repr = xla::poplarplugin::ToString(groups);
  const int64_t num_groups = groups.size();
  const int64_t group_size = groups[0].replica_ids_size();
  if (group_size == 0) {
    return xla::InvalidArgument("Unsupported empty replica group. Input replica groups: ", groups_repr);
  }

  TF_ASSIGN_OR_RETURN(const int64_t group_stride, GetReplicaGroupStride(groups[0]));
  // Check replica group topology is supported by IPU collectives.
  for (int64_t i = 0; i < num_groups; ++i) {
    const xla::ReplicaGroup& group = groups[i];
    // Check consistent replica group size.
    if (group.replica_ids_size() != group_size) {
      return xla::InvalidArgumentStrCat(
          "Inconsistent replica group size: Expected ", group_size, ", actual ",
          group.replica_ids_size(), "Input replica groups: ", groups_repr);
    }
    // Consistent replica group stride.
    TF_ASSIGN_OR_RETURN(const int64_t stride, GetReplicaGroupStride(groups[i]));
    if (stride != group_stride) {
      return xla::InvalidArgumentStrCat(
          "Inconsistent replica group stride: Expected ", group_stride, ", actual ",
          stride, "Input replica groups: ", groups_repr);
    }
  }

  // Only 3 topologies supported by GCL library: All, Consecutive and Orthogonal.
  if (num_groups == 1) {
    // All. TODO: pass zero group size?
    return PoplarReplicaGroups::Consecutive(group_size);
  }
  else if (num_groups == 2) {
    if (group_stride == 1) {
      // Consecutive groups.
      return PoplarReplicaGroups::Consecutive(group_size);
    }
    if (num_groups == group_stride) {
      // Orthogonal groups.
      return PoplarReplicaGroups::Orthogonal(group_size);
    }
  }
  // TODO: more general support than just 2 groups?
  return xla::InvalidArgumentStrCat("Unsupported number of replica groups: ",
                                    num_groups);
}

bool PoplarReplicaGroups::operator==(const PoplarReplicaGroups& other) const {
  return group_size_ == other.group_size_ && group_type_ == other.group_type_;
}

bool PoplarReplicaGroups::operator!=(const PoplarReplicaGroups& other) const {
  return !(*this == other);
}

size_t PoplarReplicaGroups::Hash() const {
  return hash_util::hash(group_size_, group_type_);
}

PoplarReplicaGroups::Type PoplarReplicaGroups::GroupType() const {
  return group_type_;
}

PoplarReplicaGroups::PoplarReplicaGroups(uint64 group_size, Type group_type)
    : group_size_(group_size), group_type_(group_type) {
  // 0 should use default constructor instead.
  CHECK_NE(group_size, 0);
}

std::ostream& operator<<(std::ostream& oss, const PoplarReplicaGroups& groups) {
  return oss << groups.ToString();
}

std::string ToString(const xla::ReplicaGroup& group)
{
  std::string repr = absl::StrCat("[", absl::StrJoin(group.replica_ids(), ", "), "]");
  return repr;
}
std::string ToString(absl::Span<const xla::ReplicaGroup> groups)
{
  std::vector<std::string> repr_groups;
  repr_groups.reserve(groups.size());
  for (const auto& group: groups) {
    repr_groups.push_back(xla::poplarplugin::ToString(group));
  }
  std::string repr = absl::StrCat("[", absl::StrJoin(repr_groups, ", "), "]");
  return repr;
}

StatusOr<int64_t> GetReplicaGroupStride(const xla::ReplicaGroup& group)
{
  const int64_t group_size = group.replica_ids_size();
  if (group_size <= 1) {
    // Single element group: stride 1 by default.
    return 1;
  }
  const int64_t group_stride = group.replica_ids(1) - group.replica_ids(0);
  for (int64_t idx = 0; idx < group_size - 1; ++idx) {
    // Only consistent stride supporting on IPUs.
    const int64_t stride = group.replica_ids(idx + 1) - group.replica_ids(idx);
    if (stride != group_stride) {
      return xla::InvalidArgumentStrCat("Replica group with inconsistent stride: ", ToString(group));
    }
  }
  return group_stride;
}

}  // namespace poplarplugin
}  // namespace xla
