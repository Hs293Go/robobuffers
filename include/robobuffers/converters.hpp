#ifndef ROBOBUFFERS_CONVERTERS_HPP_
#define ROBOBUFFERS_CONVERTERS_HPP_

#include <chrono>
#include <string_view>
#include <type_traits>

#include "robobuffers/accel_generated.h"
#include "robobuffers/accel_stamped_generated.h"
#include "robobuffers/expected.hpp"
#include "robobuffers/geom.hpp"
#include "robobuffers/header_generated.h"
#include "robobuffers/odometry_generated.h"
#include "robobuffers/pose_generated.h"
#include "robobuffers/pose_stamped_generated.h"
#include "robobuffers/quat_generated.h"
#include "robobuffers/twist_generated.h"
#include "robobuffers/twist_stamped_generated.h"
#include "robobuffers/vec3_generated.h"
#include "robobuffers/wrench_generated.h"
#include "robobuffers/wrench_stamped_generated.h"

namespace robo {

enum class ConvertError {
  kMissingHeader,
  kMissingFields,
  kInvalidSize,
};

template <typename T>
concept TimestampLike = std::is_convertible_v<T, uint64_t> || requires(T t) {
  typename T::clock;
  typename T::duration;
  { t.time_since_epoch() } -> std::same_as<typename T::duration>;
};

template <TimestampLike T>
uint64_t ToStampUs(T timestamp) {
  if constexpr (std::is_integral_v<T>) {
    return static_cast<uint64_t>(timestamp);
  } else {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               timestamp.time_since_epoch())
        .count();
  }
}

template <TimestampLike T>
flatbuffers::Offset<core_msgs::Header> ToHeader(
    flatbuffers::FlatBufferBuilder& fbb, T timestamp,
    std::string_view frame_id = "") {
  return core_msgs::CreateHeader(
      fbb, ToStampUs(timestamp),
      frame_id.empty() ? 0 : fbb.CreateString(frame_id));
}

template <typename T>
struct WithHeader {
  uint64_t stamp_us;
  std::string frame_id;
  T data;
};

template <typename T>
WithHeader<T> AddHeader(T&& data, const core_msgs::Header& header) {
  return {
      .stamp_us = header.stamp_us(),
      .frame_id = header.frame_id() ? header.frame_id()->str() : std::string(),
      .data = std::forward<T>(data)};
}

// Basic type converters
// =====================
//
// These converters are protected by concepts so that downstream composite
// converters are not burdened.

/**
 * @brief Converts an Eigen::Vector3f to a robobuffers Vec3 message.
 *
 * @param vec input Eigen vector-like quantity (can be Map or block)
 * @return geom_msgs::Vec3 corresponding robobuffers message
 */
template <typename Derived>
  requires(Derived::SizeAtCompileTime == 3 &&
           // Eigen "bool"-constants are long
           Derived::IsVectorAtCompileTime == 1L &&
           std::is_same_v<typename Derived::Scalar, float>)
geom_msgs::Vec3 ToMessage(const Eigen::MatrixBase<Derived>& vec) {
  return geom_msgs::Vec3(vec.x(), vec.y(), vec.z());
}

/**
 * @brief Converts a robobuffers Vec3 message to an Eigen::Vector3f.
 *
 * @param vec input robobuffers Vec3 message
 * @return Eigen::Vector3f corresponding Eigen vector
 */
inline Eigen::Vector3f FromMessage(const geom_msgs::Vec3& vec) {
  return Eigen::Vector3f(vec.x(), vec.y(), vec.z());
}

/**
 * @brief Converts an Eigen::Quaternionf to a robobuffers Quat message.
 *
 * @param quat input Eigen quaternion-like quantity (can be Map)
 * @return geom_msgs::Quat corresponding robobuffers message
 */
template <typename Derived>
  requires(std::is_same_v<typename Derived::Scalar, float>)
geom_msgs::Quat ToMessage(const Eigen::QuaternionBase<Derived>& quat) {
  return geom_msgs::Quat(quat.x(), quat.y(), quat.z(), quat.w());
}

/**
 * @brief Converts a robobuffers Quat message to an Eigen::Quaternionf.
 *
 * @param quat input robobuffers Quat message
 * @return Eigen::Quaternionf corresponding Eigen quaternion
 */
inline Eigen::Quaternionf FromMessage(const geom_msgs::Quat& quat) {
  return Eigen::Quaternionf(quat.w(), quat.x(), quat.y(), quat.z());
}

/**
 * @brief Converts an Eigen matrix covariance to a robobuffers vector message.
 *
 * @tparam Derived Eigen matrix-like type
 * @param fbb FlatBufferBuilder to use for allocation
 * @param cov input covariance matrix (must be square, float type)
 * @return flatbuffers::Offset<flatbuffers::Vector<float>> corresponding
 *         robobuffers message
 */
template <int Dim, typename Derived>
  requires(Derived::IsVectorAtCompileTime != 1L &&
           Derived::RowsAtCompileTime == Derived::ColsAtCompileTime &&
           Derived::RowsAtCompileTime == Dim &&
           std::is_same_v<typename Derived::Scalar, float>)
flatbuffers::Offset<flatbuffers::Vector<float>> CovarianceMatrixToMessage(
    flatbuffers::FlatBufferBuilder& fbb,
    const Eigen::MatrixBase<Derived>& cov) {
  const Eigen::Matrix<float, Derived::RowsAtCompileTime,
                      Derived::ColsAtCompileTime>
      tmp = cov;
  return fbb.CreateVector(tmp.data(), tmp.size());
}

template <int Dim, typename Derived>
  requires(Derived::IsVectorAtCompileTime != 1L &&
           Derived::RowsAtCompileTime == Derived::ColsAtCompileTime &&
           Derived::RowsAtCompileTime == Dim &&
           std::is_same_v<typename Derived::Scalar, float>)
flatbuffers::Offset<flatbuffers::Vector<float>> CovarianceMatrixToMessage(
    flatbuffers::FlatBufferBuilder& fbb,
    const Eigen::DiagonalBase<Derived>& cov) {
  return CovarianceMatrixToMessage<Dim>(fbb, cov.toDenseMatrix());
}

/**
 * @brief Try to convert a robobuffers covariance vector message to an Eigen
 * matrix.
 *
 * @tparam Dim dimension of the square covariance matrix
 * @param cov input robobuffers covariance vector message
 * @return Eigen::Matrix<float, Dim, Dim> corresponding Eigen covariance matrix
 * if sizes match
 * @ return std::nullopt otherwise
 */
template <int Dim>
expected<Eigen::Matrix<float, Dim, Dim>, ConvertError>
CovarianceMatrixFromMessage(const flatbuffers::Vector<float>& cov) {
  if (cov.size() != Dim * Dim) {
    return unexpected(ConvertError::kInvalidSize);
  }
  return Eigen::Matrix<float, Dim, Dim>::Map(cov.data());
}

// Composite type converters
// =========================
//
// Pose converters
// ---------------

/**
 * @brief Converts a transformation to the corresponding message.
 *
 * @param fbb FlatBufferBuilder to use for allocation
 * @param transform input robobuffers TransformBase
 * @return flatbuffers::Offset<geom_msgs::Pose> corresponding robobuffers
 * message
 */
template <typename Derived>
flatbuffers::Offset<geom_msgs::Pose> ToMessage(
    flatbuffers::FlatBufferBuilder& fbb,
    const TransformBase<Derived>& transform) {
  geom_msgs::Vec3 position = ToMessage(transform.translation());
  geom_msgs::Quat orientation = ToMessage(transform.rotation());

  return geom_msgs::CreatePose(fbb, &position, &orientation);
}

/**
 * @brief Converts a transformation and covariance to the corresponding message.
 *
 * @param fbb FlatBufferBuilder to use for allocation
 * @param transform input robobuffers TransformBase
 * @param covariance input covariance matrix (must be square, float type)
 * @return flatbuffers::Offset<geom_msgs::Pose> corresponding robobuffers
 * message
 */
template <typename Derived, typename CDerived>
flatbuffers::Offset<geom_msgs::Pose> ToMessage(
    flatbuffers::FlatBufferBuilder& fbb,
    const TransformBase<Derived>& transform,
    const Eigen::MatrixBase<CDerived>& covariance) {
  auto position = ToMessage(transform.translation());
  auto orientation = ToMessage(transform.rotation());
  auto cov = CovarianceMatrixToMessage<6>(fbb, covariance);
  return geom_msgs::CreatePose(fbb, &position, &orientation, cov);
}

struct TransformWithCovariance {
  TransformF32 transform;
  std::optional<Eigen::Matrix<float, 6, 6>> covariance = std::nullopt;
};

/**
 * @brief Converts a pose message to a transformation object with optional
 * covariance.
 *
 * @param pose input robobuffers Pose message
 * @return expected<TransformWithCovariance, ConvertError> A
 * TransformWithCovariance containing the transformation data and optional
 * covariance on success, or a ConvertError indicating the failure
 */
inline expected<TransformWithCovariance, ConvertError> FromMessage(
    const geom_msgs::Pose& pose) {
  if (pose.position() == nullptr || pose.orientation() == nullptr) {
    return unexpected(ConvertError::kMissingFields);
  }
  auto translation = FromMessage(*pose.position());
  auto rotation = FromMessage(*pose.orientation());
  auto transform = TransformF32(translation, rotation);

  if (pose.covariance()) {
    auto covariance = CovarianceMatrixFromMessage<6>(*pose.covariance());
    if (covariance) {
      return TransformWithCovariance{
          .transform = std::move(transform),
          .covariance = std::move(covariance).value()};
    }
    return unexpected(covariance.error());
  }
  return TransformWithCovariance{.transform = std::move(transform)};
}

/**
 * @brief Converts a transformation augmented with header info to the
 * corresponding message.
 *
 * @param fbb FlatBufferBuilder to use for allocation
 * @param timestamp timestamp
 * @param transform A transformation object
 * @param frame_id optional frame ID, defaults to empty string
 * @return flatbuffers::Offset<geom_msgs::PoseStamped> corresponding
 * robobuffers message
 */
template <typename Derived, TimestampLike T>
flatbuffers::Offset<geom_msgs::PoseStamped> ToMessage(
    flatbuffers::FlatBufferBuilder& fbb, T timestamp,
    const TransformBase<Derived>& transform, std::string_view frame_id = "") {
  auto header = ToHeader(fbb, timestamp, frame_id);
  auto pose = ToMessage(fbb, transform);
  return geom_msgs::CreatePoseStamped(fbb, header, pose);
}

/**
 * @brief Converts a transformation with covariance and header info to the
 * corresponding message.
 *
 * @param fbb FlatBufferBuilder to use for allocation
 * @param timestamp timestamp
 * @param transform A transformation object
 * @param covariance input covariance matrix (must be square, float type)
 * @param frame_id optional frame ID, defaults to empty string
 * @return flatbuffers::Offset<geom_msgs::PoseStamped> corresponding
 * robobuffers message
 */
template <typename Derived, TimestampLike T, typename CDerived>
flatbuffers::Offset<geom_msgs::PoseStamped> ToMessage(
    flatbuffers::FlatBufferBuilder& fbb, T timestamp,
    const TransformBase<Derived>& transform,
    const Eigen::MatrixBase<CDerived>& covariance,
    std::string_view frame_id = "") {
  auto header = ToHeader(fbb, timestamp, frame_id);
  auto pose = ToMessage(fbb, transform, covariance);
  return geom_msgs::CreatePoseStamped(fbb, header, pose);
}

/**
 * @brief Converts a pose message with header to a transformation object with
 * optional covariance.
 *
 * @param pose_stamped input robobuffers PoseStamped message
 * @return expected<WithHeader<TransformWithCovariance>, ConvertError> A
 * WithHeader<TransformWithCovariance> containing the header info and
 * transformation data on success, or a ConvertError indicating the failure
 */
inline expected<WithHeader<TransformWithCovariance>, ConvertError> FromMessage(
    const geom_msgs::PoseStamped& pose_stamped) {
  if (pose_stamped.header() == nullptr) {
    return unexpected(ConvertError::kMissingHeader);
  }
  if (pose_stamped.pose() == nullptr) {
    return unexpected(ConvertError::kMissingFields);
  }
  auto transform_result = FromMessage(*pose_stamped.pose());
  if (!transform_result) {
    return unexpected(transform_result.error());
  }
  return AddHeader(std::move(transform_result).value(), *pose_stamped.header());
}

// Twist converters
// ----------------

/**
 * @brief Converts a generalized velocity (twist) to the corresponding message.
 *
 * @param fbb FlatBufferBuilder to use for allocation
 * @param twist input robobuffers TwistBase
 * @return flatbuffers::Offset<geom_msgs::Twist> corresponding robobuffers
 * message
 */
template <typename Derived>
flatbuffers::Offset<geom_msgs::Twist> ToMessage(
    flatbuffers::FlatBufferBuilder& fbb, const TwistBase<Derived>& twist) {
  geom_msgs::Vec3 linear = ToMessage(twist.linear());
  geom_msgs::Vec3 angular = ToMessage(twist.angular());
  return geom_msgs::CreateTwist(fbb, &linear, &angular);
}

/**
 * @brief Converts a generalized velocity (twist) with covariance to the
 * corresponding message.
 * @param fbb FlatBufferBuilder to use for allocation
 * @param twist input robobuffers TwistBase
 * @param covariance input covariance matrix (must be square, float type)
 * @return flatbuffers::Offset<geom_msgs::Twist> corresponding robobuffers
 * message
 */
template <typename Derived, typename CDerived>
flatbuffers::Offset<geom_msgs::Twist> ToMessage(
    flatbuffers::FlatBufferBuilder& fbb, const TwistBase<Derived>& twist,
    const Eigen::MatrixBase<CDerived>& covariance) {
  auto linear = ToMessage(twist.linear());
  auto angular = ToMessage(twist.angular());
  auto cov = CovarianceMatrixToMessage<6>(fbb, covariance);
  return geom_msgs::CreateTwist(fbb, &linear, &angular, cov);
}

struct TwistWithCovariance {
  TwistF32 twist;
  std::optional<Eigen::Matrix<float, 6, 6>> covariance = std::nullopt;
};

/**
 * @brief Converts a twist message to a generalized velocity (twist) with
 * optional covariance.
 *
 * @param twist_msg input robobuffers Twist message
 * @return A expected<TwistWithCovariance, ConvertError> containing the twist
 * data and optional covariance on success, or a ConvertError indicating the
 * failure reason
 */
inline expected<TwistWithCovariance, ConvertError> FromMessage(
    const geom_msgs::Twist& twist_msg) {
  if (twist_msg.linear() == nullptr || twist_msg.angular() == nullptr) {
    return unexpected(ConvertError::kMissingFields);
  }
  auto linear = FromMessage(*twist_msg.linear());
  auto angular = FromMessage(*twist_msg.angular());
  auto twist = TwistF32(linear, angular);

  if (twist_msg.covariance()) {
    auto covariance = CovarianceMatrixFromMessage<6>(*twist_msg.covariance());
    if (covariance) {
      return TwistWithCovariance{.twist = std::move(twist),
                                 .covariance = std::move(covariance).value()};
    }
    return unexpected(covariance.error());
  }
  return TwistWithCovariance{.twist = std::move(twist)};
}

/**
 * @brief Converts a generalized velocity (twist) augmented with header info
 * to the corresponding message.
 *
 * @param fbb FlatBufferBuilder to use for allocation
 * @param timestamp timestamp
 * @param twist A twist object
 * @param frame_id optional frame ID, defaults to empty string
 * @return flatbuffers::Offset<geom_msgs::TwistStamped> corresponding
 * robobuffers message
 */
template <typename Derived, TimestampLike T>
flatbuffers::Offset<geom_msgs::TwistStamped> ToMessage(
    flatbuffers::FlatBufferBuilder& fbb, T timestamp,
    const TwistBase<Derived>& twist, std::string_view frame_id = "") {
  auto header = ToHeader(fbb, timestamp, frame_id);
  auto twist_msg = ToMessage(fbb, twist);
  return geom_msgs::CreateTwistStamped(fbb, header, twist_msg);
}

/**
 * @brief Converts a generalized velocity (twist) with covariance and header
 * info to the corresponding message.
 *
 * @param fbb FlatBufferBuilder to use for allocation
 * @param timestamp timestamp
 * @param twist A twist object
 * @param covariance input covariance matrix (must be square, float type)
 * @param frame_id optional frame ID, defaults to empty string
 * @return flatbuffers::Offset<geom_msgs::TwistStamped> corresponding
 * robobuffers message
 */
template <typename Derived, TimestampLike T, typename CDerived>
flatbuffers::Offset<geom_msgs::TwistStamped> ToMessage(
    flatbuffers::FlatBufferBuilder& fbb, T timestamp,
    const TwistBase<Derived>& twist,
    const Eigen::MatrixBase<CDerived>& covariance,
    std::string_view frame_id = "") {
  auto header = ToHeader(fbb, timestamp, frame_id);
  auto twist_msg = ToMessage(fbb, twist, covariance);
  return geom_msgs::CreateTwistStamped(fbb, header, twist_msg);
}

/**
 * @brief Converts a twist message with header to a generalized velocity
 * (twist) object with optional covariance.
 *
 * @param twist_stamped input robobuffers TwistStamped message
 * @return expected<TransformWithHeaderWithCovariance, ConvertError> A
 * WithHeader<TwistWithCovariance> containing the header info, the twist data,
 * and optional covariance on success, or a ConvertError indicating the failure
 * reason
 */
inline expected<WithHeader<TwistWithCovariance>, ConvertError> FromMessage(
    const geom_msgs::TwistStamped& twist_stamped) {
  if (twist_stamped.header() == nullptr) {
    return unexpected(ConvertError::kMissingHeader);
  }

  if (twist_stamped.twist() == nullptr) {
    return unexpected(ConvertError::kMissingFields);
  }
  auto twist_result = FromMessage(*twist_stamped.twist());
  if (!twist_result) {
    return unexpected(twist_result.error());
  }
  return AddHeader(std::move(twist_result).value(), *twist_stamped.header());
}

// Accel converters
// ----------------

/**
 * @brief Converts an acceleration (linear and angular) to the corresponding
 * message.
 *
 * @param fbb FlatBufferBuilder to use for allocation
 * @param accel input robobuffers Accel
 * @return flatbuffers::Offset<geom_msgs::Accel> corresponding robobuffers
 * message
 */
template <typename Derived>
flatbuffers::Offset<geom_msgs::Accel> ToMessage(
    flatbuffers::FlatBufferBuilder& fbb, const AccelBase<Derived>& accel) {
  geom_msgs::Vec3 linear = ToMessage(accel.linear());
  geom_msgs::Vec3 angular = ToMessage(accel.angular());
  return geom_msgs::CreateAccel(fbb, &linear, &angular);
}

/**
 * @brief Converts an acceleration (linear and angular) with covariance to the
 * corresponding message.
 *
 * @param fbb FlatBufferBuilder to use for allocation
 * @param accel input robobuffers Accel
 * @param covariance input covariance matrix (must be square, float type)
 * @return flatbuffers::Offset<geom_msgs::Accel> corresponding robobuffers
 * message
 */
template <typename Derived, typename CDerived>
flatbuffers::Offset<geom_msgs::Accel> ToMessage(
    flatbuffers::FlatBufferBuilder& fbb, const AccelBase<Derived>& accel,
    const Eigen::MatrixBase<CDerived>& covariance) {
  auto linear = ToMessage(accel.linear());
  auto angular = ToMessage(accel.angular());
  auto cov = CovarianceMatrixToMessage<6>(fbb, covariance);
  return geom_msgs::CreateAccel(fbb, &linear, &angular, cov);
}

struct AccelWithCovariance {
  AccelF32 accel;
  std::optional<Eigen::Matrix<float, 6, 6>> covariance = std::nullopt;
};

/**
 * @brief Converts an accel message to an acceleration object with optional
 * covariance.
 *
 * @param accel_msg input robobuffers Accel message
 * @return expected<AccelWithCovariance, ConvertError> An AccelWithCovariance
 * containing acceleration data and optional covariance on success, or a
 * ConvertError indicating the failure reason
 */
inline expected<AccelWithCovariance, ConvertError> FromMessage(
    const geom_msgs::Accel& accel_msg) {
  if (accel_msg.linear() == nullptr || accel_msg.angular() == nullptr) {
    return unexpected(ConvertError::kMissingFields);
  }
  auto linear = FromMessage(*accel_msg.linear());
  auto angular = FromMessage(*accel_msg.angular());
  auto accel = AccelF32(linear, angular);
  if (accel_msg.covariance()) {
    auto covariance = CovarianceMatrixFromMessage<6>(*accel_msg.covariance());
    if (covariance) {
      return AccelWithCovariance{.accel = std::move(accel),
                                 .covariance = std::move(covariance).value()};
    }
    return unexpected(covariance.error());
  }
  return AccelWithCovariance{.accel = std::move(accel)};
}

/**
 * @brief Converts an acceleration (linear and angular) augmented with header
 * info to the corresponding message.
 *
 * @param fbb FlatBufferBuilder to use for allocation
 * @param timestamp timestamp
 * @param accel An acceleration object
 * @param frame_id optional frame ID, defaults to empty string
 * @return flatbuffers::Offset<geom_msgs::AccelStamped> corresponding
 * robobuffers message
 */
template <typename Derived, TimestampLike T>
flatbuffers::Offset<geom_msgs::AccelStamped> ToMessage(
    flatbuffers::FlatBufferBuilder& fbb, T timestamp,
    const AccelBase<Derived>& accel, std::string_view frame_id = "") {
  auto header = ToHeader(fbb, timestamp, frame_id);
  auto accel_msg = ToMessage(fbb, accel);
  return geom_msgs::CreateAccelStamped(fbb, header, accel_msg);
}

/**
 * @brief Converts an acceleration (linear and angular) with covariance and
 * header info to the corresponding message.
 *
 * @param fbb FlatBufferBuilder to use for allocation
 * @param timestamp timestamp
 * @param accel An acceleration object
 * @param covariance input covariance matrix (must be square, float type)
 * @param frame_id optional frame ID, defaults to empty string
 * @return flatbuffers::Offset<geom_msgs::AccelStamped> corresponding
 * robobuffers message
 */
template <typename Derived, TimestampLike T, typename CDerived>
flatbuffers::Offset<geom_msgs::AccelStamped> ToMessage(
    flatbuffers::FlatBufferBuilder& fbb, T timestamp,
    const AccelBase<Derived>& accel,
    const Eigen::MatrixBase<CDerived>& covariance,
    std::string_view frame_id = "") {
  auto header = ToHeader(fbb, timestamp, frame_id);
  auto accel_msg = ToMessage(fbb, accel, covariance);
  return geom_msgs::CreateAccelStamped(fbb, header, accel_msg);
}

/**
 * @brief Converts an accel message with header to an acceleration object with
 * optional covariance.
 *
 * @param accel_stamped input robobuffers AccelStamped message
 * @return expected<WithHeader<AccelWithCovariance>, ConvertError> A
 * WithHeader<AccelWithCovariance> containing the header info, the acceleration
 * data, and optional covariance on success, or a ConvertError indicating the
 * failure reason
 */
inline expected<WithHeader<AccelWithCovariance>, ConvertError> FromMessage(
    const geom_msgs::AccelStamped& accel_stamped) {
  if (accel_stamped.header() == nullptr) {
    return unexpected(ConvertError::kMissingHeader);
  }
  if (accel_stamped.accel() == nullptr) {
    return unexpected(ConvertError::kMissingFields);
  }
  auto accel_result = FromMessage(*accel_stamped.accel());
  if (!accel_result) {
    return unexpected(accel_result.error());
  }
  return AddHeader(std::move(accel_result).value(), *accel_stamped.header());
}

// Wrench converters
// ----------------

/**
 * @brief Converts a wrench (force and torque) to the corresponding message.
 *
 * @param fbb FlatBufferBuilder to use for allocation
 * @param wrench input robobuffers WrenchBase
 * @return flatbuffers::Offset<geom_msgs::Wrench> corresponding
 * robobuffers message
 */
template <typename Derived>
flatbuffers::Offset<geom_msgs::Wrench> ToMessage(
    flatbuffers::FlatBufferBuilder& fbb, const WrenchBase<Derived>& wrench) {
  geom_msgs::Vec3 force = ToMessage(wrench.force());
  geom_msgs::Vec3 torque = ToMessage(wrench.torque());
  return geom_msgs::CreateWrench(fbb, &force, &torque);
}

/**
 * @brief Converts a wrench (force and torque) with covariance to the
 * corresponding message.
 *
 * @param fbb FlatBufferBuilder to use for allocation
 * @param wrench input robobuffers WrenchBase
 * @param covariance input covariance matrix (must be square, float type)
 * @return flatbuffers::Offset<geom_msgs::Wrench> corresponding
 * robobuffers message
 */
template <typename Derived, typename CDerived>
flatbuffers::Offset<geom_msgs::Wrench> ToMessage(
    flatbuffers::FlatBufferBuilder& fbb, const WrenchBase<Derived>& wrench,
    const Eigen::MatrixBase<CDerived>& covariance) {
  geom_msgs::Vec3 force = ToMessage(wrench.force());
  geom_msgs::Vec3 torque = ToMessage(wrench.torque());

  auto cov = CovarianceMatrixToMessage<6>(fbb, covariance);
  return geom_msgs::CreateWrench(fbb, &force, &torque, cov);
}

struct WrenchWithCovariance {
  WrenchF32 wrench;
  std::optional<Eigen::Matrix<float, 6, 6>> covariance = std::nullopt;
};

/**
 * @brief Converts a wrench message to a wrench object with optional covariance.
 *
 * @param wrench_msg input robobuffers Wrench message
 * @return expected<WrenchWithCovariance, ConvertError> A WrenchWithCovariance
 * containing wrench data and optional covariance on success, or a ConvertError
 * indicating the failure reason
 */
inline expected<WrenchWithCovariance, ConvertError> FromMessage(
    const geom_msgs::Wrench& wrench_msg) {
  if (wrench_msg.force() == nullptr || wrench_msg.torque() == nullptr) {
    return unexpected(ConvertError::kMissingFields);
  }
  auto force = FromMessage(*wrench_msg.force());
  auto torque = FromMessage(*wrench_msg.torque());
  auto wrench = WrenchF32(force, torque);

  if (wrench_msg.covariance()) {
    auto covariance = CovarianceMatrixFromMessage<6>(*wrench_msg.covariance());
    if (covariance) {
      return WrenchWithCovariance{.wrench = std::move(wrench),
                                  .covariance = std::move(covariance).value()};
    }
    return unexpected(covariance.error());
  }
  return WrenchWithCovariance{.wrench = std::move(wrench)};
}

/**
 * @brief Converts a wrench (force and torque) augmented with header info to
 * the corresponding message.
 *
 * @param fbb FlatBufferBuilder to use for allocation
 * @param timestamp timestamp
 * @param wrench A wrench object
 * @param frame_id optional frame ID, defaults to empty string
 * @return flatbuffers::Offset<geom_msgs::WrenchStamped> corresponding
 * robobuffers message
 */
template <typename Derived, TimestampLike T>
flatbuffers::Offset<geom_msgs::WrenchStamped> ToMessage(
    flatbuffers::FlatBufferBuilder& fbb, T timestamp,
    const WrenchBase<Derived>& wrench, std::string_view frame_id = "") {
  auto header = ToHeader(fbb, timestamp, frame_id);
  auto wrench_msg = ToMessage(fbb, wrench);
  return geom_msgs::CreateWrenchStamped(fbb, header, wrench_msg);
}

/**
 * @brief Converts a wrench (force and torque) with covariance and header
 * info to the corresponding message.
 *
 * @param fbb FlatBufferBuilder to use for allocation
 * @param timestamp timestamp
 * @param wrench A wrench object
 * @param covariance input covariance matrix (must be square, float type)
 * @param frame_id optional frame ID, defaults to empty string
 * @return flatbuffers::Offset<geom_msgs::WrenchStamped> corresponding
 * robobuffers message
 */
template <typename Derived, TimestampLike T, typename CDerived>
flatbuffers::Offset<geom_msgs::WrenchStamped> ToMessage(
    flatbuffers::FlatBufferBuilder& fbb, T timestamp,
    const WrenchBase<Derived>& wrench,
    const Eigen::MatrixBase<CDerived>& covariance,
    std::string_view frame_id = "") {
  auto header = ToHeader(fbb, timestamp, frame_id);
  auto wrench_msg = ToMessage(fbb, wrench, covariance);
  return geom_msgs::CreateWrenchStamped(fbb, header, wrench_msg);
}

/**
 * @brief Converts a wrench message with header to a wrench object with
 * optional covariance.
 *
 * @param wrench_stamped input robobuffers WrenchStamped message
 * @return expected<WithHeader<WrenchWithCovariance>, ConvertError> A
 * WithHeader<WrenchWithCovariance> containing the header info, the wrench data,
 * and optional covariance on success, or a ConvertError indicating the failure
 * reason
 */
inline expected<WithHeader<WrenchWithCovariance>, ConvertError> FromMessage(
    const geom_msgs::WrenchStamped& wrench_stamped) {
  if (wrench_stamped.header() == nullptr) {
    return unexpected(ConvertError::kMissingHeader);
  }
  if (wrench_stamped.wrench() == nullptr) {
    return unexpected(ConvertError::kMissingFields);
  }
  auto wrench_result = FromMessage(*wrench_stamped.wrench());
  if (!wrench_result) {
    return unexpected(wrench_result.error());
  }
  return AddHeader(std::move(wrench_result).value(), *wrench_stamped.header());
}

// Odometry converters
// -------------------

template <typename Derived, TimestampLike T>
flatbuffers::Offset<geom_msgs::Odometry> ToMessage(
    flatbuffers::FlatBufferBuilder& fbb, T timestamp,
    const OdometryBase<Derived>& odom, std::string_view frame_id = "") {
  auto pose = ToMessage(fbb, odom.pose());
  auto twist = ToMessage(fbb, odom.twist());
  auto header = ToHeader(fbb, timestamp, frame_id);
  return geom_msgs::CreateOdometry(fbb, header, pose, twist);
}

template <typename Derived, TimestampLike T, typename PDerived,
          typename TDerived>
flatbuffers::Offset<geom_msgs::Odometry> ToMessage(
    flatbuffers::FlatBufferBuilder& fbb, T timestamp,
    const OdometryBase<Derived>& odom,
    const Eigen::MatrixBase<PDerived>& pose_cov,
    const Eigen::MatrixBase<TDerived>& twist_cov,
    std::string_view frame_id = "") {
  auto pose = ToMessage(fbb, odom.pose(), pose_cov);
  auto twist = ToMessage(fbb, odom.twist(), twist_cov);
  auto header = ToHeader(fbb, timestamp, frame_id);
  return geom_msgs::CreateOdometry(fbb, header, pose, twist);
}

struct OdometryWithCovariance {
  OdometryF32 odometry;
  std::optional<Eigen::Matrix<float, 6, 6>> pose_covariance = std::nullopt;
  std::optional<Eigen::Matrix<float, 6, 6>> twist_covariance = std::nullopt;
};

inline expected<WithHeader<OdometryWithCovariance>, ConvertError> FromMessage(
    const geom_msgs::Odometry& odom_msg) {
  if (odom_msg.header() == nullptr) {
    return unexpected(ConvertError::kMissingHeader);
  }
  if (odom_msg.pose() == nullptr || odom_msg.twist() == nullptr) {
    return unexpected(ConvertError::kMissingFields);
  }
  auto pose_result = FromMessage(*odom_msg.pose());
  if (!pose_result) {
    return unexpected(pose_result.error());
  }
  auto twist_result = FromMessage(*odom_msg.twist());
  if (!twist_result) {
    return unexpected(twist_result.error());
  }

  auto odom = OdometryF32(pose_result->transform, twist_result->twist);

  return AddHeader(
      OdometryWithCovariance{
          .odometry = std::move(odom),
          .pose_covariance = pose_result->covariance,
          .twist_covariance = twist_result->covariance,
      },
      *odom_msg.header());
}

}  // namespace robo

#endif  // ROBOBUFFERS_CONVERTERS_HPP_
