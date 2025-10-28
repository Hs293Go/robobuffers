#ifndef ROBOBUFFERS_CONCEPTS_HPP_
#define ROBOBUFFERS_CONCEPTS_HPP_

#include <concepts>
namespace robo {
template <typename Derived>
concept EigenMatrixLike = requires {
  typename Derived::Scalar;
  { Derived::RowsAtCompileTime } -> std::convertible_to<int>;
  { Derived::ColsAtCompileTime } -> std::convertible_to<int>;
  { Derived::IsVectorAtCompileTime } -> std::convertible_to<bool>;
};

template <typename Derived>
concept Matrix4Like =
    EigenMatrixLike<Derived> && Derived::RowsAtCompileTime == 4 &&
    Derived::ColsAtCompileTime == 4;

template <typename Derived>
concept Vector3Like = EigenMatrixLike<Derived> &&
                      static_cast<bool>(Derived::IsVectorAtCompileTime) &&
                      Derived::RowsAtCompileTime == 3;

[[noreturn]] inline void unreachable() {
#if defined(__GNUC__) || defined(__clang__)
  __builtin_unreachable();
#elif defined(_MSC_VER)
  __assume(false);
#else
  std::terminate();
#endif
}
}  // namespace robo

#endif  // ROBOBUFFERS_CONCEPTS_HPP_
