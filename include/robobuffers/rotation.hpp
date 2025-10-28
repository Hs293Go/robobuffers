#ifndef ROBOBUFFERS_ROTATION_HPP_
#define ROBOBUFFERS_ROTATION_HPP_

#include "Eigen/Dense"
#include "robobuffers/concepts.hpp"

namespace robo {

template <Vector3Like Derived>
Eigen::Quaternion<typename Derived::Scalar> AngleAxisToQuaternion(
    const Eigen::MatrixBase<Derived>& angle_axis,
    const typename Derived::Scalar angle_tol = Derived::Scalar(1e-5)) {
  using std::cos;
  using std::sin;
  using std::sqrt;
  using Scalar = typename Derived::Scalar;
  const Scalar angle_sq = angle_axis.squaredNorm();

  Scalar imag_factor;
  Scalar real_factor;
  if (angle_sq > angle_tol * angle_tol) {
    const Scalar angle = sqrt(angle_sq);
    const Scalar half_angle = Scalar(0.5) * angle;
    const Scalar sin_half_angle = sin(half_angle);
    imag_factor = sin_half_angle / angle;
    real_factor = cos(half_angle);
  } else {
    Scalar angle_po4 = angle_sq * angle_sq;

    imag_factor = Scalar(0.5) - Scalar(1.0 / 48.0) * angle_sq +
                  Scalar(1.0 / 3840.0) * angle_po4;
    real_factor = Scalar(1) - Scalar(1.0 / 8.0) * angle_sq +
                  Scalar(1.0 / 384.0) * angle_po4;
  }

  return Eigen::Quaternion{real_factor, imag_factor * angle_axis.x(),
                           imag_factor * angle_axis.y(),
                           imag_factor * angle_axis.z()};
}

}  // namespace robo

#endif  // ROBOBUFFERS_ROTATION_HPP_
