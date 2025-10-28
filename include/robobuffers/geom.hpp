#ifndef ROBOBUFFERS_GEOM_HPP_
#define ROBOBUFFERS_GEOM_HPP_

#include "Eigen/Dense"
#include "robobuffers/meta.hpp"

namespace robo {

template <typename T>
Eigen::Matrix3<T> hat(const Eigen::Vector3<T>& v) {
  Eigen::Matrix3<T> m = Eigen::Matrix3<T>::Zero();
  m << T(0), -v.z(), v.y(), v.z(), T(0), -v.x(), -v.y(), v.x(), T(0);
  return m;
}

// Transform (translation and rotation)
// ====================================

template <typename Derived>
struct TransformTraits;

template <typename T>
class Transform;

template <typename T>
class TransformView;

template <typename Derived>
class TransformBase {
 public:
  using Scalar = typename TransformTraits<Derived>::Scalar;
  using DerivedVector = typename TransformTraits<Derived>::Vector;
  using DerivedQuaternion = typename TransformTraits<Derived>::Quaternion;
  using Vector = Eigen::Vector3<Scalar>;
  using Quaternion = Eigen::Quaternion<Scalar>;
  using RotationMatrix = Eigen::Matrix3<Scalar>;
  using TransformValue = Transform<Scalar>;
  using TransformMatrix = Eigen::Matrix4<Scalar>;
  using ParamVector = Eigen::Vector<Scalar, 7>;

  TransformBase() = default;
  TransformBase(const TransformBase&) = default;
  TransformBase(TransformBase&&) = default;

  template <typename OD>
  Derived& operator=(const TransformBase<OD>& t) {
    translation() = t.translation();
    rotation() = t.rotation();
    return derived();
  }

  Derived& operator=(const TransformBase& t) {
    translation() = t.translation();
    rotation() = t.rotation();
    return derived();
  }

  template <typename OD>
  Derived& operator=(TransformBase<OD>&& t) {
    translation() = std::move(t).translation();
    rotation() = std::move(t).rotation();
    return derived();
  }

  Derived& operator=(TransformBase&& t) {
    translation() = std::move(t).translation();
    rotation() = std::move(t).rotation();
    return derived();
  }

  template <typename OD>
  Derived& operator*=(const TransformBase<OD>& t) {
    translation() += rotation() * t.translation();
    rotation() *= t.rotation();
    return derived();
  }

  template <typename OD>
  TransformValue operator*(const TransformBase<OD>& t) const {
    Transform<Scalar> res = derived();
    res *= t;
    return res;
  }

  template <Vector3Like OD>
  Eigen::Vector<Scalar, 3> operator*(const Eigen::MatrixBase<OD>& v) const {
    return rotation() * v + translation();
  }

  Derived& derived() { return static_cast<Derived&>(*this); }
  const Derived& derived() const { return static_cast<const Derived&>(*this); }

  const DerivedVector& translation() const { return derived().translation(); }
  const DerivedQuaternion& rotation() const { return derived().rotation(); }

  DerivedVector& translation() { return derived().translation(); }
  DerivedQuaternion& rotation() { return derived().rotation(); }

  TransformValue inverse() const {
    const Quaternion r_inv = rotation().conjugate();
    const Vector t_inv = -(r_inv * translation());
    return {t_inv, r_inv};
  }

  TransformMatrix matrix() const {
    TransformMatrix res = TransformMatrix::Identity();
    res.template topLeftCorner<3, 3>() = toRotationMatrix();
    res.template topRightCorner<3, 1>() = translation();
    return res;
  }

  RotationMatrix toRotationMatrix() const {
    return rotation().toRotationMatrix();
  }

  ParamVector params() const {
    ParamVector params;
    params.template head<3>() = translation();
    params.template segment<4>(3) = rotation().coeffs();
    return params;
  }
};

template <typename T>
struct TransformTraits<Transform<T>> {
  using Scalar = T;
  using Vector = Eigen::Vector3<T>;
  using Quaternion = Eigen::Quaternion<T>;
};

template <typename Derived>
class TwistBase;

template <typename T>
class Transform : public TransformBase<Transform<T>> {
 public:
  using Base = TransformBase<Transform<T>>;
  Transform() = default;

  template <Vector3Like TDerived, typename RDerived>
  Transform(const Eigen::MatrixBase<TDerived>& t,
            const Eigen::QuaternionBase<RDerived>& r)
      : translation_(t), rotation_(r) {}

  template <Matrix4Like Derived>
  explicit Transform(const Eigen::MatrixBase<Derived>& mat)
      : translation_(mat.template topRightCorner<3, 1>()),
        rotation_(mat.template topLeftCorner<3, 3>()) {}

  template <typename OD>
  Transform(const TransformBase<OD>& other) {
    Base::operator=(other);
  }

  const Eigen::Vector3<T>& translation() const { return translation_; }
  const Eigen::Quaternion<T>& rotation() const { return rotation_; }

  Eigen::Vector3<T>& translation() { return translation_; }
  Eigen::Quaternion<T>& rotation() { return rotation_; }

  template <typename Derived>
  static Transform exp(const TwistBase<Derived>& twist);

 private:
  Eigen::Vector3<T> translation_ = Eigen::Vector3<T>::Zero();
  Eigen::Quaternion<T> rotation_ = Eigen::Quaternion<T>::Identity();
};

template <typename T>
struct TransformTraits<TransformView<T>> {
  using Scalar = T;
  using Vector = Eigen::Map<const Eigen::Vector3<T>>;
  using Quaternion = Eigen::Map<const Eigen::Quaternion<T>>;
};

template <typename T>
class TransformView : public TransformBase<TransformView<T>> {
 public:
  TransformView() = default;

  TransformView(const T* data) : translation_(data), rotation_(data + 3) {}

  const Eigen::Map<const Eigen::Vector3<T>>& translation() const {
    return translation_;
  }
  const Eigen::Map<const Eigen::Quaternion<T>>& rotation() const {
    return rotation_;
  }

 private:
  Eigen::Map<const Eigen::Vector3<T>> translation_;
  Eigen::Map<const Eigen::Quaternion<T>> rotation_;
};

// Twist (linear and angular velocity)
// ===================================

template <typename Derived>
struct TwistTraits;

template <typename T>
class Twist;

template <typename T>
class TwistView;

template <typename Derived>
class TwistBase {
 public:
  using Scalar = typename TwistTraits<Derived>::Scalar;
  using DerivedVector = typename TwistTraits<Derived>::Vector;
  using Vector = Eigen::Vector3<Scalar>;
  using TwistValue = Twist<Scalar>;
  using ParamVector = Eigen::Vector<Scalar, 6>;

  TwistBase() = default;
  TwistBase(const TwistBase&) = default;
  TwistBase(TwistBase&&) = default;

  template <typename OD>
  Derived& operator=(const TwistBase<OD>& other) {
    linear() = other.linear();
    angular() = other.angular();
    return derived();
  }

  Derived& operator=(const TwistBase& other) {
    linear() = other.linear();
    angular() = other.angular();
    return derived();
  }

  template <typename OD>
  Derived& operator=(TwistBase<OD>&& other) {
    linear() = std::move(other).linear();
    angular() = std::move(other).angular();
    return derived();
  }

  Derived& operator=(TwistBase&& other) {
    linear() = std::move(other).linear();
    angular() = std::move(other).angular();
    return derived();
  }

  template <typename OD>
  Derived& operator+=(const TwistBase<OD>& other) {
    linear() += other.linear();
    angular() += other.angular();
    return derived();
  }

  template <typename OD>
  Derived& operator-=(const TwistBase<OD>& other) {
    linear() -= other.linear();
    angular() -= other.angular();
    return derived();
  }

  Derived& operator*=(const Scalar s) {
    linear() *= s;
    angular() *= s;
    return derived();
  }

  Derived& operator/=(const Scalar s) {
    linear() /= s;
    angular() /= s;
    return derived();
  }

  template <typename OD>
  TwistValue operator+(const TwistBase<OD>& other) const {
    TwistValue res = derived();
    res += other;
    return res;
  }

  template <typename OD>
  TwistValue operator-(const TwistBase<OD>& other) const {
    TwistValue res = derived();
    res -= other;
    return res;
  }

  TwistValue operator*(const Scalar s) const {
    TwistValue res = derived();
    res *= s;
    return res;
  }

  TwistValue operator/(const Scalar s) const {
    TwistValue res = derived();
    res /= s;
    return res;
  }

  Derived& derived() { return static_cast<Derived&>(*this); }
  const Derived& derived() const { return static_cast<const Derived&>(*this); }

  const DerivedVector& linear() const { return derived().linear(); }
  const DerivedVector& angular() const { return derived().angular(); }

  DerivedVector& linear() { return derived().linear(); }
  DerivedVector& angular() { return derived().angular(); }

  ParamVector params() const {
    ParamVector params;
    params.template head<3>() = linear();
    params.template tail<3>() = angular();
    return params;
  }
};

template <typename T>
struct TwistTraits<Twist<T>> {
  using Scalar = T;
  using Vector = Eigen::Vector3<T>;
};

template <typename T>
class Twist : public TwistBase<Twist<T>> {
 public:
  Twist() = default;

  template <Vector3Like LDerived, Vector3Like ADerived>
  Twist(const Eigen::MatrixBase<LDerived>& v,
        const Eigen::MatrixBase<ADerived>& w)
      : linear_(v), angular_(w) {}

  Twist(const Twist&) = default;
  Twist& operator=(const Twist&) = default;

  Twist(Twist&&) = default;
  Twist& operator=(Twist&&) = default;

  template <typename OD>
  Twist(const TwistBase<OD>& other) {
    this->TwistBase<Twist>::operator=(other);
  }

  template <typename OD>
  Twist& operator=(const TwistBase<OD>& other) {
    this->TwistBase<Twist>::operator=(other);
    return *this;
  }

  const Eigen::Vector3<T>& linear() const { return linear_; }
  const Eigen::Vector3<T>& angular() const { return angular_; }

  Eigen::Vector3<T>& linear() { return linear_; }
  Eigen::Vector3<T>& angular() { return angular_; }

 private:
  Eigen::Vector3<T> linear_ = Eigen::Vector3<T>::Zero();
  Eigen::Vector3<T> angular_ = Eigen::Vector3<T>::Zero();
};

template <typename T>
struct TwistTraits<TwistView<T>> {
  using Scalar = T;
  using Vector = Eigen::Map<const Eigen::Vector3<T>>;
};

template <typename T>
class TwistView : public TwistBase<TwistView<T>> {
 public:
  TwistView(T const* data) : linear_(data), angular_(data + 3) {}
  const Eigen::Map<const Eigen::Vector3<T>>& linear() const { return linear_; }
  const Eigen::Map<const Eigen::Vector3<T>>& angular() const {
    return angular_;
  }

 private:
  Eigen::Map<const Eigen::Vector3<T>> linear_;
  Eigen::Map<const Eigen::Vector3<T>> angular_;
};

// Accel (linear and angular acceleration)
// =======================================

template <typename Derived>
struct AccelTraits;

template <typename T>
class Accel;

template <typename T>
class AccelView;

template <typename Derived>
class AccelBase {
 public:
  using Scalar = typename AccelTraits<Derived>::Scalar;
  using DerivedVector = typename AccelTraits<Derived>::Vector;
  using Vector = Eigen::Vector<Scalar, 3>;
  using AccelValue = Accel<Scalar>;
  using ParamVector = Eigen::Vector<Scalar, 6>;

  AccelBase() = default;
  AccelBase(const AccelBase&) = default;
  AccelBase(AccelBase&&) = default;

  template <typename OD>
  Derived& operator=(const AccelBase<OD>& other) {
    linear() = other.linear();
    angular() = other.angular();
    return derived();
  }

  Derived& operator=(const AccelBase<Derived>& other) {
    linear() = other.linear();
    angular() = other.angular();
    return derived();
  }

  template <typename OD>
  Derived& operator+=(const TwistBase<OD>& other) {
    linear() += other.linear();
    angular() += other.angular();
    return derived();
  }

  template <typename OD>
  Derived& operator-=(const TwistBase<OD>& other) {
    linear() -= other.linear();
    angular() -= other.angular();
    return derived();
  }

  Derived& operator*=(const Scalar s) {
    linear() *= s;
    angular() *= s;
    return derived();
  }

  Derived& operator/=(const Scalar s) {
    linear() /= s;
    angular() /= s;
    return derived();
  }

  template <typename OD>
  AccelValue operator+(const AccelBase<OD>& other) const {
    AccelValue res = derived();
    res += other;
    return res;
  }

  template <typename OD>
  AccelValue operator-(const AccelBase<OD>& other) const {
    AccelValue res = derived();
    res -= other;
    return res;
  }

  AccelValue operator*(const Scalar s) const {
    AccelValue res = derived();
    res *= s;
    return res;
  }

  AccelValue operator/(const Scalar s) const {
    AccelValue res = derived();
    res /= s;
    return res;
  }

  Derived& derived() { return static_cast<Derived&>(*this); }
  const Derived& derived() const { return static_cast<const Derived&>(*this); }
  const DerivedVector& linear() const { return derived().linear(); }
  const DerivedVector& angular() const { return derived().angular(); }
  DerivedVector& linear() { return derived().linear(); }
  DerivedVector& angular() { return derived().angular(); }
  ParamVector params() const {
    ParamVector params;
    params.template head<3>() = linear();
    params.template tail<3>() = angular();
    return params;
  }
};

template <typename T>
class Accel : public AccelBase<Accel<T>> {
 public:
  Accel() = default;

  template <Vector3Like LDerived, Vector3Like ADerived>
  Accel(const Eigen::MatrixBase<LDerived>& v,
        const Eigen::MatrixBase<ADerived>& w)
      : linear_(v), angular_(w) {}

  Accel(const Accel&) = default;
  Accel& operator=(const Accel&) = default;
  Accel(Accel&&) = default;
  Accel& operator=(Accel&&) = default;

  template <typename OD>
  Accel(const AccelBase<OD>& other) {
    this->AccelBase<Accel>::operator=(other);
  }

  template <typename OD>
  Accel& operator=(const AccelBase<OD>& other) {
    this->AccelBase<Accel>::operator=(other);
    return *this;
  }

  const Eigen::Vector3<T>& linear() const { return linear_; }
  const Eigen::Vector3<T>& angular() const { return angular_; }

  Eigen::Vector3<T>& linear() { return linear_; }
  Eigen::Vector3<T>& angular() { return angular_; }

 private:
  Eigen::Vector3<T> linear_ = Eigen::Vector3<T>::Zero();
  Eigen::Vector3<T> angular_ = Eigen::Vector3<T>::Zero();
};

template <typename T>
struct AccelTraits<Accel<T>> {
  using Scalar = T;
  using Vector = Eigen::Vector3<T>;
};

template <typename T>
class AccelView : public AccelBase<AccelView<T>> {
 public:
  AccelView(T const* data) : linear_(data), angular_(data + 3) {}
  const Eigen::Map<const Eigen::Vector3<T>>& linear() const { return linear_; }
  const Eigen::Map<const Eigen::Vector3<T>>& angular() const {
    return angular_;
  }

 private:
  Eigen::Map<const Eigen::Vector3<T>> linear_;
  Eigen::Map<const Eigen::Vector3<T>> angular_;
};

template <typename T>
struct AccelTraits<AccelView<T>> {
  using Scalar = T;
  using Vector = Eigen::Map<const Eigen::Vector3<T>>;
};

// Wrench (force and torque)
// =========================

template <typename Derived>
struct WrenchTraits;

template <typename T>
class Wrench;

template <typename T>
class WrenchView;

template <typename Derived>
class WrenchBase {
 public:
  using Scalar = typename WrenchTraits<Derived>::Scalar;
  using DerivedVector = typename WrenchTraits<Derived>::Vector;
  using Vector = Eigen::Vector<Scalar, 3>;
  using WrenchValue = Wrench<Scalar>;
  using ParamVector = Eigen::Vector<Scalar, 6>;

  WrenchBase() = default;
  WrenchBase(const WrenchBase&) = default;
  WrenchBase(WrenchBase&&) = default;

  template <typename OD>
  Derived& operator=(const WrenchBase<OD>& other) {
    force() = other.force();
    torque() = other.torque();
    return derived();
  }

  Derived& operator=(const WrenchBase& other) {
    force() = other.force();
    torque() = other.torque();
    return derived();
  }

  template <typename OD>
  Derived& operator+=(const WrenchBase<OD>& other) {
    force() += other.force();
    torque() += other.torque();
    return derived();
  }

  template <typename OD>
  Derived& operator-=(const WrenchBase<OD>& other) {
    force() -= other.force();
    torque() -= other.torque();
    return derived();
  }

  Derived& operator*=(const Scalar s) {
    force() *= s;
    torque() *= s;
    return derived();
  }

  Derived& operator/=(const Scalar s) {
    force() /= s;
    torque() /= s;
    return derived();
  }

  template <typename OD>
  WrenchValue operator+(const WrenchBase<OD>& other) const {
    WrenchValue res = derived();
    res += other;
    return res;
  }

  template <typename OD>
  WrenchValue operator-(const WrenchBase<OD>& other) const {
    WrenchValue res = derived();
    res -= other;
    return res;
  }

  WrenchValue operator*(const Scalar s) const {
    WrenchValue res = derived();
    res *= s;
    return res;
  }

  WrenchValue operator/(const Scalar s) const {
    WrenchValue res = derived();
    res /= s;
    return res;
  }

  Derived& derived() { return static_cast<Derived&>(*this); }
  const Derived& derived() const { return static_cast<const Derived&>(*this); }

  const DerivedVector& force() const { return derived().force(); }
  const DerivedVector& torque() const { return derived().torque(); }
  DerivedVector& force() { return derived().force(); }
  DerivedVector& torque() { return derived().torque(); }

  ParamVector params() const {
    ParamVector params;
    params.template head<3>() = force();
    params.template tail<3>() = torque();
    return params;
  }
};

template <typename T>
struct WrenchTraits<Wrench<T>> {
  using Scalar = T;
  using Vector = Eigen::Vector3<T>;
};

template <typename T>
class Wrench : public WrenchBase<Wrench<T>> {
 public:
  Wrench() = default;
  Wrench(const Eigen::Vector3<T>& f, const Eigen::Vector3<T>& t)
      : force_(f), torque_(t) {}

  Wrench(const Wrench&) = default;
  Wrench& operator=(const Wrench&) = default;

  Wrench(Wrench&&) = default;
  Wrench& operator=(Wrench&&) = default;

  template <typename OD>
  Wrench(const WrenchBase<OD>& other) {
    this->WrenchBase<Wrench>::operator=(other);
  }

  template <typename OD>
  Wrench& operator=(const WrenchBase<OD>& other) {
    this->WrenchBase<Wrench>::operator=(other);
    return *this;
  }

  const Eigen::Vector3<T>& force() const { return force_; }
  const Eigen::Vector3<T>& torque() const { return torque_; }
  Eigen::Vector3<T>& force() { return force_; }
  Eigen::Vector3<T>& torque() { return torque_; }

 private:
  Eigen::Vector3<T> force_ = Eigen::Vector3<T>::Zero();
  Eigen::Vector3<T> torque_ = Eigen::Vector3<T>::Zero();
};

template <typename T>
struct WrenchTraits<WrenchView<T>> {
  using Scalar = T;
  using Vector = Eigen::Map<const Eigen::Vector3<T>>;
};

template <typename T>
class WrenchView : public WrenchBase<WrenchView<T>> {
 public:
  WrenchView(T const* data) : force_(data), torque_(data + 3) {}
  const Eigen::Map<const Eigen::Vector3<T>>& force() const { return force_; }
  const Eigen::Map<const Eigen::Vector3<T>>& torque() const { return torque_; }

 private:
  Eigen::Map<const Eigen::Vector3<T>> force_;
  Eigen::Map<const Eigen::Vector3<T>> torque_;
};

// Odometry (pose and twist)
// =========================

template <typename T>
struct OdometryTraits;

template <typename Derived>
class OdometryBase {
 public:
  using Scalar = typename OdometryTraits<Derived>::Scalar;
  using TransformType = typename OdometryTraits<Derived>::TransformType;
  using TwistType = typename OdometryTraits<Derived>::TwistType;

  OdometryBase() = default;
  OdometryBase(const OdometryBase&) = default;
  OdometryBase(OdometryBase&&) = default;

  Derived& operator=(const OdometryBase<Derived>& other) {
    pose() = other.pose();
    twist() = other.twist();
    return derived();
  }

  template <typename OD>
  Derived& operator=(const OdometryBase<OD>& other) {
    pose() = other.pose();
    twist() = other.twist();
    return derived();
  }

  Derived& derived() { return static_cast<Derived&>(*this); }
  const Derived& derived() const { return static_cast<const Derived&>(*this); }

  const TransformType& pose() const { return derived().pose(); }
  const TwistType& twist() const { return derived().twist(); }

  TransformType& pose() { return derived().pose(); }
  TwistType& twist() { return derived().twist(); }
};

template <typename T>
class Odometry;

template <typename T>
class OdometryView;

template <typename T>
struct OdometryTraits<Odometry<T>> {
  using Scalar = T;
  using TransformType = Transform<T>;
  using TwistType = Twist<T>;
};

template <typename T>
class Odometry : public OdometryBase<Odometry<T>> {
 public:
  using Base = OdometryBase<Odometry<T>>;
  Odometry() = default;
  Odometry(const Transform<T>& p, const Twist<T>& t) : pose_(p), twist_(t) {}

  Odometry(const Odometry&) = default;
  Odometry& operator=(const Odometry&) = default;

  Odometry(Odometry&&) = default;
  Odometry& operator=(Odometry&&) = default;

  template <typename OD>
  Odometry(const OdometryBase<OD>& other) {
    this->Base::operator=(other);
  }

  const Transform<T>& pose() const { return pose_; }
  const Twist<T>& twist() const { return twist_; }
  Transform<T>& pose() { return pose_; }
  Twist<T>& twist() { return twist_; }

 private:
  Transform<T> pose_;
  Twist<T> twist_;
};

template <typename T>
struct OdometryTraits<OdometryView<T>> {
  using Scalar = T;
  using TransformType = TransformView<T>;
  using TwistType = TwistView<T>;
};

template <typename T>
class OdometryView : public OdometryBase<OdometryView<T>> {
 public:
  OdometryView() = default;
  OdometryView(const T* data) : pose_(data), twist_(data + 7) {}
  const TransformView<T>& pose() const { return pose_; }
  const TwistView<T>& twist() const { return twist_; }

 private:
  TransformView<T> pose_;
  TwistView<T> twist_;
};

using TransformF32 = Transform<float>;
using TransformViewF32 = TransformView<float>;
using TransformF64 = Transform<double>;
using TransformViewF64 = TransformView<double>;

using TwistF32 = Twist<float>;
using TwistViewF32 = TwistView<float>;
using TwistF64 = Twist<double>;
using TwistViewF64 = TwistView<double>;

using OdometryF32 = Odometry<float>;
using OdometryViewF32 = OdometryView<float>;
using OdometryF64 = Odometry<double>;
using OdometryViewF64 = OdometryView<double>;

using AccelF32 = Accel<float>;
using AccelViewF32 = AccelView<float>;
using AccelF64 = Accel<double>;
using AccelViewF64 = AccelView<double>;

using WrenchF32 = Wrench<float>;
using WrenchViewF32 = WrenchView<float>;
using WrenchF64 = Wrench<double>;
using WrenchViewF64 = WrenchView<double>;

}  // namespace robo

#endif  // ROBOBUFFERS_GEOM_HPP_
