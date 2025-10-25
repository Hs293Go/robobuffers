#ifndef ROBOBUFFERS_EXPECTED_HPP_
#define ROBOBUFFERS_EXPECTED_HPP_

#if defined(__cpp_lib_expected) && __cpp_lib_expected >= 202002L
#include <expected>
namespace robobuffers {
template <typename T, typename E>
using expected = std::expected<T, E>;
template <typename E>
using unexpected = std::unexpected<E>;
}  // namespace robobuffers
#else
#include "robobuffers/third_party/tl/expected.hpp"
namespace robo {
template <typename T, typename E>
using expected = tl::expected<T, E>;
template <typename E>
using unexpected = tl::unexpected<E>;
}  // namespace robo
#endif

#endif  // ROBOBUFFERS_EXPECTED_HPP_
