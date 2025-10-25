#include <random>
#include <ranges>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "robobuffers/converters.hpp"

static constexpr auto kNumTrials = 1000;

MATCHER_P(ComponentsEq, expected, "") {
  if constexpr (requires {
                  arg.w();
                  expected.w();
                }) {
    return ::testing::ExplainMatchResult(::testing::FloatEq(expected.x()),
                                         arg.x(), result_listener) &&
           ::testing::ExplainMatchResult(::testing::FloatEq(expected.y()),
                                         arg.y(), result_listener) &&
           ::testing::ExplainMatchResult(::testing::FloatEq(expected.z()),
                                         arg.z(), result_listener) &&
           ::testing::ExplainMatchResult(::testing::FloatEq(expected.w()),
                                         arg.w(), result_listener);
  } else {
    return ::testing::ExplainMatchResult(::testing::FloatEq(expected.x()),
                                         arg.x(), result_listener) &&
           ::testing::ExplainMatchResult(::testing::FloatEq(expected.y()),
                                         arg.y(), result_listener) &&
           ::testing::ExplainMatchResult(::testing::FloatEq(expected.z()),
                                         arg.z(), result_listener);
  }
}

TEST(TestConversion, testVec3) {
  Eigen::Vector3f v = Eigen::Vector3f::Random();
  auto msg = robo::ToMessage(v);
  EXPECT_THAT(v, ComponentsEq(msg));

  robo::geom_msgs::Vec3 v_msg(0.1f, 0.2f, 0.3f);
  auto v_back = robo::FromMessage(v_msg);
  EXPECT_THAT(v_back, ComponentsEq(v_msg));
}

TEST(TestConversion, testQuat) {
  Eigen::Quaternionf q = Eigen::Quaternionf::UnitRandom();
  auto msg = robo::ToMessage(q);
  EXPECT_THAT(q, ComponentsEq(msg));

  robo::geom_msgs::Quat q_msg(0.1f, 0.2f, 0.3f, 0.4f);
  auto q_back = robo::FromMessage(q_msg);
  EXPECT_THAT(q_back, ComponentsEq(q_msg));
}

TEST(TestConversion, testIntegralStampToHeader) {
  flatbuffers::FlatBufferBuilder fbb;
  auto header_offset = robo::ToHeader(fbb, 1234567890u, "frame_1");
  fbb.Finish(header_offset);
  std::span<const uint8_t> buf(fbb.GetBufferPointer(), fbb.GetSize());

  const auto* msg = robo::core_msgs::GetHeader(buf.data());
  ASSERT_NE(msg, nullptr);
  EXPECT_EQ(msg->stamp_us(), 1234567890u);
  ASSERT_NE(msg->frame_id(), nullptr);
  EXPECT_EQ(msg->frame_id()->str(), "frame_1");
}

TEST(TestConversion, testTimePointStampToHeader) {
  flatbuffers::FlatBufferBuilder fbb;
  auto now = std::chrono::steady_clock::now();
  auto header_offset = robo::ToHeader(fbb, now);
  fbb.Finish(header_offset);
  std::span<const uint8_t> buf(fbb.GetBufferPointer(), fbb.GetSize());
  const auto* msg = robo::core_msgs::GetHeader(buf.data());
  ASSERT_NE(msg, nullptr);
  EXPECT_EQ(std::chrono::microseconds(msg->stamp_us()),
            std::chrono::duration_cast<std::chrono::microseconds>(
                now.time_since_epoch()));
  EXPECT_EQ(msg->frame_id(), nullptr);
}

TEST(TestConversion, testToTransformNoCovariance) {
  Eigen::Vector3f t = Eigen::Vector3f::Random();
  Eigen::Quaternionf r = Eigen::Quaternionf::UnitRandom();

  // From owning Transform object
  {
    robo::TransformF32 transform(t, r);
    flatbuffers::FlatBufferBuilder fbb;
    auto offset = robo::ToMessage(fbb, transform);
    fbb.Finish(offset);
    std::span<uint8_t> buf(fbb.GetBufferPointer(), fbb.GetSize());

    const auto* msg = robo::geom_msgs::GetPose(buf.data());
    ASSERT_NE(msg->orientation(), nullptr);
    EXPECT_THAT(*msg->orientation(), ComponentsEq(r));
    ASSERT_NE(msg->position(), nullptr);
    EXPECT_THAT(*msg->position(), ComponentsEq(t));
    EXPECT_EQ(msg->covariance(), nullptr);
  }

  float params[7] = {t.x(), t.y(), t.z(), r.x(), r.y(), r.z(), r.w()};

  // From view over a raw float array
  {
    robo::TransformViewF32 transform(params);
    flatbuffers::FlatBufferBuilder fbb;
    auto offset = robo::ToMessage(fbb, transform);
    fbb.Finish(offset);
    std::span<uint8_t> buf(fbb.GetBufferPointer(), fbb.GetSize());
    const auto* msg = robo::geom_msgs::GetPose(buf.data());
    ASSERT_NE(msg->position(), nullptr);
    ASSERT_NE(msg->orientation(), nullptr);
    float result[7] = {msg->position()->x(),    msg->position()->y(),
                       msg->position()->z(),    msg->orientation()->x(),
                       msg->orientation()->y(), msg->orientation()->z(),
                       msg->orientation()->w()};
    EXPECT_THAT(result, testing::Pointwise(testing::FloatEq(), params));
  }
}

TEST(TestCoversion, testFromTransformNoCovariance) {
  flatbuffers::FlatBufferBuilder fbb;

  robo::geom_msgs::Vec3 p(1.0f, 2.0f, 3.0f);
  robo::geom_msgs::Quat q(0.0f, 0.0f, 0.0f, 1.0f);
  robo::geom_msgs::PoseBuilder pose_builder(fbb);
  pose_builder.add_position(&p);
  pose_builder.add_orientation(&q);
  auto pose_offset = pose_builder.Finish();
  fbb.Finish(pose_offset);
  std::span<uint8_t> buf(fbb.GetBufferPointer(), fbb.GetSize());

  const auto* msg = robo::geom_msgs::GetPose(buf.data());
  auto res = robo::FromMessage(*msg);
  ASSERT_TRUE(res.has_value());
  auto [transform_back, cov_opt] = res.value();

  EXPECT_THAT(p, ComponentsEq(transform_back.translation()));
  EXPECT_THAT(q, ComponentsEq(transform_back.rotation()));
  EXPECT_FALSE(cov_opt.has_value());
}

TEST(TestConversion, testToTransformWithCovariance) {
  Eigen::Vector3f t = Eigen::Vector3f::Random();
  Eigen::Quaternionf r = Eigen::Quaternionf::UnitRandom();
  robo::TransformF32 transform(t, r);
  Eigen::Matrix<float, 6, 6> cov = Eigen::Matrix<float, 6, 6>::Random();

  flatbuffers::FlatBufferBuilder fbb;
  auto offset = robo::ToMessage(fbb, transform, cov);
  fbb.Finish(offset);
  std::span<uint8_t> buf(fbb.GetBufferPointer(), fbb.GetSize());

  const auto* msg = robo::geom_msgs::GetPose(buf.data());
  ASSERT_NE(msg->orientation(), nullptr);
  EXPECT_THAT(*msg->orientation(), ComponentsEq(r));
  ASSERT_NE(msg->position(), nullptr);
  EXPECT_THAT(*msg->position(), ComponentsEq(t));
  ASSERT_NE(msg->covariance(), nullptr);
  EXPECT_THAT(*msg->covariance(),
              testing::Pointwise(testing::FloatEq(), cov.reshaped()));
}

TEST(TestConversion, testFromTransformWithCovariance) {
  std::random_device rd;
  std::mt19937 gen(rd());
  flatbuffers::FlatBufferBuilder fbb;
  robo::geom_msgs::Vec3 p(1.0f, 2.0f, 3.0f);
  robo::geom_msgs::Quat q(0.0f, 0.0f, 0.0f, 1.0f);
  std::vector<float> cov_data(36);
  std::ranges::generate(
      cov_data,
      std::bind_front(std::uniform_real_distribution(-10.0f, 10.0f), gen));
  auto cov_offset = fbb.CreateVector(cov_data);
  robo::geom_msgs::PoseBuilder pose_builder(fbb);
  pose_builder.add_position(&p);
  pose_builder.add_orientation(&q);
  pose_builder.add_covariance(cov_offset);
  auto pose_offset = pose_builder.Finish();
  fbb.Finish(pose_offset);
  std::span<uint8_t> buf(fbb.GetBufferPointer(), fbb.GetSize());
  const auto* msg = robo::geom_msgs::GetPose(buf.data());
  auto res = robo::FromMessage(*msg);
  ASSERT_TRUE(res.has_value());
  auto [transform_back, cov_opt] = res.value();
  EXPECT_THAT(p, ComponentsEq(transform_back.translation()));
  EXPECT_THAT(q, ComponentsEq(transform_back.rotation()));
  ASSERT_TRUE(cov_opt.has_value());
  EXPECT_THAT(cov_opt->reshaped(),
              testing::Pointwise(testing::FloatEq(), cov_data));
}

TEST(TestConversion, testFromTransformMissingField) {
  flatbuffers::FlatBufferBuilder fbb;
  robo::geom_msgs::PoseBuilder pose_builder(fbb);
  auto pose_offset = pose_builder.Finish();
  fbb.Finish(pose_offset);
  std::span<uint8_t> buf(fbb.GetBufferPointer(), fbb.GetSize());
  const auto* msg = robo::geom_msgs::GetPose(buf.data());
  auto res = robo::FromMessage(*msg);
  ASSERT_FALSE(res.has_value());
  auto err = res.error();
  EXPECT_EQ(err, robo::ConvertError::kMissingFields);
}

TEST(TestConversion, testFromTransformWrongCovarianceSize) {
  flatbuffers::FlatBufferBuilder fbb;
  std::vector<float> wrong_cov_data(25, 0.0f);
  auto cov_offset = fbb.CreateVector(wrong_cov_data);
  robo::geom_msgs::PoseBuilder pose_builder(fbb);
  auto p = robo::ToMessage(Eigen::Vector3f::Zero());
  pose_builder.add_position(&p);
  auto q = robo::ToMessage(Eigen::Quaternionf::Identity());
  pose_builder.add_orientation(&q);

  pose_builder.add_covariance(cov_offset);
  auto pose_offset = pose_builder.Finish();
  fbb.Finish(pose_offset);
  std::span<uint8_t> buf(fbb.GetBufferPointer(), fbb.GetSize());
  const auto* msg = robo::geom_msgs::GetPose(buf.data());
  auto res = robo::FromMessage(*msg);
  ASSERT_FALSE(res.has_value());
  auto err = res.error();
  EXPECT_EQ(err, robo::ConvertError::kInvalidSize);
}

TEST(TestConversion, testToTransformStamped) {
  Eigen::Vector3f t = Eigen::Vector3f::Random();
  Eigen::Quaternionf r = Eigen::Quaternionf::UnitRandom();
  robo::TransformF32 transform(t, r);

  flatbuffers::FlatBufferBuilder fbb;
  auto offset = robo::ToMessage(fbb, 1234567890u, transform, "frame_1");
  fbb.Finish(offset);
  std::span<uint8_t> buf(fbb.GetBufferPointer(), fbb.GetSize());

  const auto* msg = robo::geom_msgs::GetPoseStamped(buf.data());
  ASSERT_NE(msg->header(), nullptr);
  EXPECT_EQ(msg->header()->stamp_us(), 1234567890u);
  ASSERT_NE(msg->header()->frame_id(), nullptr);
  EXPECT_EQ(msg->header()->frame_id()->str(), "frame_1");

  ASSERT_NE(msg->pose(), nullptr);
}

TEST(TestConversion, testFromTransformStamped) {
  flatbuffers::FlatBufferBuilder fbb;

  robo::geom_msgs::Vec3 p(1.0f, 2.0f, 3.0f);
  robo::geom_msgs::Quat q(0.0f, 0.0f, 0.0f, 1.0f);
  auto frame_id_offset = fbb.CreateString("frame_1");
  robo::geom_msgs::PoseBuilder pose_builder(fbb);
  pose_builder.add_position(&p);
  pose_builder.add_orientation(&q);
  auto pose_offset = pose_builder.Finish();

  robo::core_msgs::HeaderBuilder header_builder(fbb);
  header_builder.add_stamp_us(1234567890u);
  header_builder.add_frame_id(frame_id_offset);
  auto header_offset = header_builder.Finish();

  robo::geom_msgs::PoseStampedBuilder pose_stamped_builder(fbb);
  pose_stamped_builder.add_header(header_offset);
  pose_stamped_builder.add_pose(pose_offset);
  auto pose_stamped_offset = pose_stamped_builder.Finish();
  fbb.Finish(pose_stamped_offset);

  std::span<uint8_t> buf(fbb.GetBufferPointer(), fbb.GetSize());

  const auto* msg = robo::geom_msgs::GetPoseStamped(buf.data());
  auto res = robo::FromMessage(*msg);
  ASSERT_TRUE(res.has_value());
  const auto& [stamp_us, fid, pose] = res.value();
  EXPECT_EQ(stamp_us, 1234567890u);
  EXPECT_EQ(fid, "frame_1");
  const auto& [transform_back, cov_opt] = pose;

  EXPECT_THAT(p, ComponentsEq(transform_back.translation()));
  EXPECT_THAT(q, ComponentsEq(transform_back.rotation()));
  EXPECT_FALSE(cov_opt.has_value());
}

TEST(TestConversion, testToTwistNoCovariance) {
  Eigen::Vector3f v = Eigen::Vector3f::Random();
  Eigen::Vector3f w = Eigen::Vector3f::Random();

  // From owning Twist object
  {
    robo::TwistF32 twist(v, w);
    flatbuffers::FlatBufferBuilder fbb;
    auto offset = robo::ToMessage(fbb, twist);
    fbb.Finish(offset);
    std::span<uint8_t> buf(fbb.GetBufferPointer(), fbb.GetSize());

    const auto* msg = robo::geom_msgs::GetTwist(buf.data());
    ASSERT_NE(msg->linear(), nullptr);
    EXPECT_THAT(*msg->linear(), ComponentsEq(v));
    ASSERT_NE(msg->angular(), nullptr);
    EXPECT_THAT(*msg->angular(), ComponentsEq(w));
    EXPECT_EQ(msg->covariance(), nullptr);
  }

  float params[6] = {v.x(), v.y(), v.z(), w.x(), w.y(), w.z()};

  // From view over a raw float array
  {
    robo::TwistViewF32 twist(params);
    flatbuffers::FlatBufferBuilder fbb;
    auto offset = robo::ToMessage(fbb, twist);
    fbb.Finish(offset);
    std::span<uint8_t> buf(fbb.GetBufferPointer(), fbb.GetSize());
    const auto* msg = robo::geom_msgs::GetTwist(buf.data());
    ASSERT_NE(msg->linear(), nullptr);
    ASSERT_NE(msg->angular(), nullptr);
    float result[6] = {msg->linear()->x(),  msg->linear()->y(),
                       msg->linear()->z(),  msg->angular()->x(),
                       msg->angular()->y(), msg->angular()->z()};
    EXPECT_THAT(result, testing::Pointwise(testing::FloatEq(), params));
  }
}

TEST(TestConversion, testFromTwistNoCovariance) {
  flatbuffers::FlatBufferBuilder fbb;

  robo::geom_msgs::Vec3 lin(1.0f, 2.0f, 3.0f);
  robo::geom_msgs::Vec3 ang(4.0f, 5.0f, 6.0f);
  robo::geom_msgs::TwistBuilder twist_builder(fbb);
  twist_builder.add_linear(&lin);
  twist_builder.add_angular(&ang);
  auto twist_offset = twist_builder.Finish();
  fbb.Finish(twist_offset);
  std::span<uint8_t> buf(fbb.GetBufferPointer(), fbb.GetSize());

  const auto* msg = robo::geom_msgs::GetTwist(buf.data());
  auto res = robo::FromMessage(*msg);
  ASSERT_TRUE(res.has_value());
  auto [twist_back, cov_opt] = res.value();

  EXPECT_THAT(lin, ComponentsEq(twist_back.linear()));
  EXPECT_THAT(ang, ComponentsEq(twist_back.angular()));
  EXPECT_FALSE(cov_opt.has_value());
}

TEST(TestConversion, testToTwistWithCovariance) {
  Eigen::Vector3f v = Eigen::Vector3f::Random();
  Eigen::Vector3f w = Eigen::Vector3f::Random();
  robo::TwistF32 twist(v, w);
  Eigen::Matrix<float, 6, 6> cov = Eigen::Matrix<float, 6, 6>::Random();

  flatbuffers::FlatBufferBuilder fbb;
  auto offset = robo::ToMessage(fbb, twist, cov);
  fbb.Finish(offset);
  std::span<uint8_t> buf(fbb.GetBufferPointer(), fbb.GetSize());

  const auto* msg = robo::geom_msgs::GetTwist(buf.data());
  ASSERT_NE(msg->linear(), nullptr);
  EXPECT_THAT(*msg->linear(), ComponentsEq(v));
  ASSERT_NE(msg->angular(), nullptr);
  EXPECT_THAT(*msg->angular(), ComponentsEq(w));
  ASSERT_NE(msg->covariance(), nullptr);
  EXPECT_THAT(*msg->covariance(),
              testing::Pointwise(testing::FloatEq(), cov.reshaped()));
}

TEST(TestConversion, testFromTwistWithCovariance) {
  std::random_device rd;
  std::mt19937 gen(rd());
  flatbuffers::FlatBufferBuilder fbb;
  robo::geom_msgs::Vec3 lin(1.0f, 2.0f, 3.0f);
  robo::geom_msgs::Vec3 ang(4.0f, 5.0f, 6.0f);
  std::vector<float> cov_data(36);
  std::ranges::generate(
      cov_data,
      std::bind_front(std::uniform_real_distribution(-10.0f, 10.0f), gen));
  auto cov_offset = fbb.CreateVector(cov_data);
  robo::geom_msgs::TwistBuilder twist_builder(fbb);
  twist_builder.add_linear(&lin);
  twist_builder.add_angular(&ang);
  twist_builder.add_covariance(cov_offset);
  auto twist_offset = twist_builder.Finish();
  fbb.Finish(twist_offset);
  std::span<uint8_t> buf(fbb.GetBufferPointer(), fbb.GetSize());
  const auto* msg = robo::geom_msgs::GetTwist(buf.data());
  auto res = robo::FromMessage(*msg);
  ASSERT_TRUE(res.has_value());
  auto [twist_back, cov_opt] = res.value();
  EXPECT_THAT(lin, ComponentsEq(twist_back.linear()));
  EXPECT_THAT(ang, ComponentsEq(twist_back.angular()));
  ASSERT_TRUE(cov_opt.has_value());
  EXPECT_THAT(cov_opt->reshaped(),
              testing::Pointwise(testing::FloatEq(), cov_data));
}

TEST(TestConversion, testFromTwistMissingField) {
  flatbuffers::FlatBufferBuilder fbb;
  robo::geom_msgs::TwistBuilder twist_builder(fbb);
  auto twist_offset = twist_builder.Finish();
  fbb.Finish(twist_offset);
  std::span<uint8_t> buf(fbb.GetBufferPointer(), fbb.GetSize());
  const auto* msg = robo::geom_msgs::GetTwist(buf.data());
  auto res = robo::FromMessage(*msg);
  ASSERT_FALSE(res.has_value());
  auto err = res.error();
  EXPECT_EQ(err, robo::ConvertError::kMissingFields);
}

TEST(TestConversion, testFromTwistWrongCovarianceSize) {
  flatbuffers::FlatBufferBuilder fbb;
  std::vector<float> wrong_cov_data(25, 0.0f);
  auto cov_offset = fbb.CreateVector(wrong_cov_data);
  robo::geom_msgs::TwistBuilder twist_builder(fbb);
  auto linear = robo::ToMessage(Eigen::Vector3f::Zero());
  twist_builder.add_linear(&linear);
  auto angular = robo::ToMessage(Eigen::Vector3f::Zero());
  twist_builder.add_angular(&angular);
  twist_builder.add_covariance(cov_offset);
  auto twist_offset = twist_builder.Finish();
  fbb.Finish(twist_offset);
  std::span<uint8_t> buf(fbb.GetBufferPointer(), fbb.GetSize());
  const auto* msg = robo::geom_msgs::GetTwist(buf.data());
  auto res = robo::FromMessage(*msg);
  ASSERT_FALSE(res.has_value());
  auto err = res.error();
  EXPECT_EQ(err, robo::ConvertError::kInvalidSize);
}

TEST(TestConversion, testToTwistStamped) {
  Eigen::Vector3f v = Eigen::Vector3f::Random();
  Eigen::Vector3f w = Eigen::Vector3f::Random();
  robo::TwistF32 twist(v, w);

  flatbuffers::FlatBufferBuilder fbb;
  auto offset = robo::ToMessage(fbb, 1234567890u, twist, "frame_1");
  fbb.Finish(offset);
  std::span<uint8_t> buf(fbb.GetBufferPointer(), fbb.GetSize());

  const auto* msg = robo::geom_msgs::GetTwistStamped(buf.data());
  ASSERT_NE(msg->header(), nullptr);
  EXPECT_EQ(msg->header()->stamp_us(), 1234567890u);
  ASSERT_NE(msg->header()->frame_id(), nullptr);
  EXPECT_EQ(msg->header()->frame_id()->str(), "frame_1");

  ASSERT_NE(msg->twist(), nullptr);
}

TEST(TestConversion, testFromTwistStamped) {
  flatbuffers::FlatBufferBuilder fbb;

  robo::geom_msgs::Vec3 lin(1.0f, 2.0f, 3.0f);
  robo::geom_msgs::Vec3 ang(4.0f, 5.0f, 6.0f);
  auto frame_id_offset = fbb.CreateString("frame_1");
  robo::geom_msgs::TwistBuilder twist_builder(fbb);
  twist_builder.add_linear(&lin);
  twist_builder.add_angular(&ang);
  auto twist_offset = twist_builder.Finish();

  robo::core_msgs::HeaderBuilder header_builder(fbb);
  header_builder.add_stamp_us(1234567890u);
  header_builder.add_frame_id(frame_id_offset);
  auto header_offset = header_builder.Finish();

  robo::geom_msgs::TwistStampedBuilder twist_stamped_builder(fbb);
  twist_stamped_builder.add_header(header_offset);
  twist_stamped_builder.add_twist(twist_offset);
  auto twist_stamped_offset = twist_stamped_builder.Finish();
  fbb.Finish(twist_stamped_offset);

  std::span<uint8_t> buf(fbb.GetBufferPointer(), fbb.GetSize());

  const auto* msg = robo::geom_msgs::GetTwistStamped(buf.data());
  auto res = robo::FromMessage(*msg);
  ASSERT_TRUE(res.has_value());
  const auto& [stamp_us, fid, twist] = res.value();
  EXPECT_EQ(stamp_us, 1234567890u);
  EXPECT_EQ(fid, "frame_1");
  const auto& [twist_back, cov_opt] = twist;
  EXPECT_THAT(lin, ComponentsEq(twist_back.linear()));
  EXPECT_THAT(ang, ComponentsEq(twist_back.angular()));
  EXPECT_FALSE(cov_opt.has_value());
}

TEST(TestConversion, testToAccelNoCovariance) {
  Eigen::Vector3f lin = Eigen::Vector3f::Random();
  Eigen::Vector3f ang = Eigen::Vector3f::Random();

  // From owning Accel object
  {
    robo::AccelF32 accel(lin, ang);

    flatbuffers::FlatBufferBuilder fbb;
    auto offset = robo::ToMessage(fbb, accel);
    fbb.Finish(offset);
    std::span<uint8_t> buf(fbb.GetBufferPointer(), fbb.GetSize());

    const auto* msg = robo::geom_msgs::GetAccel(buf.data());
    ASSERT_NE(msg->linear(), nullptr);
    EXPECT_THAT(*msg->linear(), ComponentsEq(lin));
    ASSERT_NE(msg->angular(), nullptr);
    EXPECT_THAT(*msg->angular(), ComponentsEq(ang));
    EXPECT_EQ(msg->covariance(), nullptr);
  }

  float params[6] = {lin.x(), lin.y(), lin.z(), ang.x(), ang.y(), ang.z()};
  // From view over a raw float array
  {
    robo::AccelViewF32 accel(params);

    flatbuffers::FlatBufferBuilder fbb;
    auto offset = robo::ToMessage(fbb, accel);
    fbb.Finish(offset);
    std::span<uint8_t> buf(fbb.GetBufferPointer(), fbb.GetSize());
    const auto* msg = robo::geom_msgs::GetAccel(buf.data());
    ASSERT_NE(msg->linear(), nullptr);
    ASSERT_NE(msg->angular(), nullptr);
    float result[6] = {msg->linear()->x(),  msg->linear()->y(),
                       msg->linear()->z(),  msg->angular()->x(),
                       msg->angular()->y(), msg->angular()->z()};
    EXPECT_THAT(result, testing::Pointwise(testing::FloatEq(), params));
  }
}

TEST(TestConversion, testFromAccelNoCovariance) {
  flatbuffers::FlatBufferBuilder fbb;

  robo::geom_msgs::Vec3 lin(1.0f, 2.0f, 3.0f);
  robo::geom_msgs::Vec3 ang(4.0f, 5.0f, 6.0f);

  robo::geom_msgs::AccelBuilder accel_builder(fbb);
  accel_builder.add_linear(&lin);
  accel_builder.add_angular(&ang);
  auto accel_offset = accel_builder.Finish();
  fbb.Finish(accel_offset);
  std::span<uint8_t> buf(fbb.GetBufferPointer(), fbb.GetSize());

  const auto* msg = robo::geom_msgs::GetAccel(buf.data());
  auto res = robo::FromMessage(*msg);
  ASSERT_TRUE(res.has_value());
  auto [accel_back, cov_opt] = res.value();

  EXPECT_THAT(lin, ComponentsEq(accel_back.linear()));
  EXPECT_THAT(ang, ComponentsEq(accel_back.angular()));
  EXPECT_FALSE(cov_opt.has_value());
}

TEST(TestConversion, testToAccelWithCovariance) {
  Eigen::Vector3f v = Eigen::Vector3f::Random();
  Eigen::Vector3f w = Eigen::Vector3f::Random();
  robo::AccelF32 accel(v, w);
  Eigen::Matrix<float, 6, 6> cov = Eigen::Matrix<float, 6, 6>::Random();

  flatbuffers::FlatBufferBuilder fbb;
  auto offset = robo::ToMessage(fbb, accel, cov);
  fbb.Finish(offset);
  std::span<uint8_t> buf(fbb.GetBufferPointer(), fbb.GetSize());

  const auto* msg = robo::geom_msgs::GetAccel(buf.data());
  ASSERT_NE(msg->linear(), nullptr);
  EXPECT_THAT(*msg->linear(), ComponentsEq(v));
  ASSERT_NE(msg->angular(), nullptr);
  EXPECT_THAT(*msg->angular(), ComponentsEq(w));
  ASSERT_NE(msg->covariance(), nullptr);
  EXPECT_THAT(*msg->covariance(),
              testing::Pointwise(testing::FloatEq(), cov.reshaped()));
}

TEST(TestConversion, testFromAccelWithCovariance) {
  std::random_device rd;
  std::mt19937 gen(rd());
  flatbuffers::FlatBufferBuilder fbb;
  robo::geom_msgs::Vec3 lin(1.0f, 2.0f, 3.0f);
  robo::geom_msgs::Vec3 ang(4.0f, 5.0f, 6.0f);
  std::vector<float> cov_data(36);
  std::ranges::generate(
      cov_data,
      std::bind_front(std::uniform_real_distribution(-10.0f, 10.0f), gen));
  auto cov_offset = fbb.CreateVector(cov_data);
  robo::geom_msgs::AccelBuilder accel_builder(fbb);
  accel_builder.add_linear(&lin);
  accel_builder.add_angular(&ang);
  accel_builder.add_covariance(cov_offset);
  auto accel_offset = accel_builder.Finish();
  fbb.Finish(accel_offset);
  std::span<uint8_t> buf(fbb.GetBufferPointer(), fbb.GetSize());
  const auto* msg = robo::geom_msgs::GetAccel(buf.data());
  auto res = robo::FromMessage(*msg);
  ASSERT_TRUE(res.has_value());
  auto [accel_back, cov_opt] = res.value();
  EXPECT_THAT(lin, ComponentsEq(accel_back.linear()));
  EXPECT_THAT(ang, ComponentsEq(accel_back.angular()));
  ASSERT_TRUE(cov_opt.has_value());
  EXPECT_THAT(cov_opt->reshaped(),
              testing::Pointwise(testing::FloatEq(), cov_data));
}

TEST(TestConversion, testFromAccelMissingField) {
  flatbuffers::FlatBufferBuilder fbb;
  robo::geom_msgs::AccelBuilder accel_builder(fbb);
  auto accel_offset = accel_builder.Finish();
  fbb.Finish(accel_offset);
  std::span<uint8_t> buf(fbb.GetBufferPointer(), fbb.GetSize());
  const auto* msg = robo::geom_msgs::GetAccel(buf.data());
  auto res = robo::FromMessage(*msg);
  ASSERT_FALSE(res.has_value());
  auto err = res.error();
  EXPECT_EQ(err, robo::ConvertError::kMissingFields);
}

TEST(TestConversion, testFromAccelWrongCovarianceSize) {
  flatbuffers::FlatBufferBuilder fbb;
  std::vector<float> wrong_cov_data(25, 0.0f);
  auto cov_offset = fbb.CreateVector(wrong_cov_data);
  robo::geom_msgs::AccelBuilder accel_builder(fbb);
  auto linear = robo::ToMessage(Eigen::Vector3f::Zero());
  accel_builder.add_linear(&linear);
  auto angular = robo::ToMessage(Eigen::Vector3f::Zero());
  accel_builder.add_angular(&angular);
  accel_builder.add_covariance(cov_offset);
  auto accel_offset = accel_builder.Finish();
  fbb.Finish(accel_offset);
  std::span<uint8_t> buf(fbb.GetBufferPointer(), fbb.GetSize());
  const auto* msg = robo::geom_msgs::GetAccel(buf.data());
  auto res = robo::FromMessage(*msg);
  ASSERT_FALSE(res.has_value());
  auto err = res.error();
  EXPECT_EQ(err, robo::ConvertError::kInvalidSize);
}

TEST(TestConversion, testToWrenchNoCovariance) {
  Eigen::Vector3f f = Eigen::Vector3f::Random();
  Eigen::Vector3f t = Eigen::Vector3f::Random();
  robo::WrenchF32 wrench(f, t);

  flatbuffers::FlatBufferBuilder fbb;
  auto offset = robo::ToMessage(fbb, wrench);
  fbb.Finish(offset);
  std::span<uint8_t> buf(fbb.GetBufferPointer(), fbb.GetSize());
}

TEST(TestConversion, testFromWrenchNoCovariance) {
  flatbuffers::FlatBufferBuilder fbb;

  robo::geom_msgs::Vec3 force(1.0f, 2.0f, 3.0f);
  robo::geom_msgs::Vec3 torque(4.0f, 5.0f, 6.0f);
  robo::geom_msgs::WrenchBuilder wrench_builder(fbb);
  wrench_builder.add_force(&force);
  wrench_builder.add_torque(&torque);
  auto wrench_offset = wrench_builder.Finish();
  fbb.Finish(wrench_offset);
  std::span<uint8_t> buf(fbb.GetBufferPointer(), fbb.GetSize());

  const auto* msg = robo::geom_msgs::GetWrench(buf.data());
  auto res = robo::FromMessage(*msg);
  ASSERT_TRUE(res.has_value());
  auto [wrench_back, cov_opt] = res.value();

  EXPECT_THAT(force, ComponentsEq(wrench_back.force()));
  EXPECT_THAT(torque, ComponentsEq(wrench_back.torque()));
  EXPECT_FALSE(cov_opt.has_value());
}

TEST(TestConversion, testToWrenchWithCovariance) {
  Eigen::Vector3f f = Eigen::Vector3f::Random();
  Eigen::Vector3f t = Eigen::Vector3f::Random();
  robo::WrenchF32 wrench(f, t);
  Eigen::Matrix<float, 6, 6> cov = Eigen::Matrix<float, 6, 6>::Random();

  flatbuffers::FlatBufferBuilder fbb;
  auto offset = robo::ToMessage(fbb, wrench, cov);
  fbb.Finish(offset);
  std::span<uint8_t> buf(fbb.GetBufferPointer(), fbb.GetSize());

  const auto* msg = robo::geom_msgs::GetWrench(buf.data());
  ASSERT_NE(msg->force(), nullptr);
  EXPECT_THAT(*msg->force(), ComponentsEq(f));
  ASSERT_NE(msg->torque(), nullptr);
  EXPECT_THAT(*msg->torque(), ComponentsEq(t));
  ASSERT_NE(msg->covariance(), nullptr);
  EXPECT_THAT(*msg->covariance(),
              testing::Pointwise(testing::FloatEq(), cov.reshaped()));
}

TEST(TestConversion, testFromWrenchWithCovariance) {
  std::random_device rd;
  std::mt19937 gen(rd());
  flatbuffers::FlatBufferBuilder fbb;
  robo::geom_msgs::Vec3 force(1.0f, 2.0f, 3.0f);
  robo::geom_msgs::Vec3 torque(4.0f, 5.0f, 6.0f);
  std::vector<float> cov_data(36);
  std::ranges::generate(
      cov_data,
      std::bind_front(std::uniform_real_distribution(-10.0f, 10.0f), gen));
  auto cov_offset = fbb.CreateVector(cov_data);
  robo::geom_msgs::WrenchBuilder wrench_builder(fbb);
  wrench_builder.add_force(&force);
  wrench_builder.add_torque(&torque);
  wrench_builder.add_covariance(cov_offset);
  auto wrench_offset = wrench_builder.Finish();
  fbb.Finish(wrench_offset);
  std::span<uint8_t> buf(fbb.GetBufferPointer(), fbb.GetSize());
  const auto* msg = robo::geom_msgs::GetWrench(buf.data());
  auto res = robo::FromMessage(*msg);
  ASSERT_TRUE(res.has_value());
  auto [wrench_back, cov_opt] = res.value();
  EXPECT_THAT(force, ComponentsEq(wrench_back.force()));
  EXPECT_THAT(torque, ComponentsEq(wrench_back.torque()));
  ASSERT_TRUE(cov_opt.has_value());
  EXPECT_THAT(cov_opt->reshaped(),
              testing::Pointwise(testing::FloatEq(), cov_data));
}

TEST(TestConversion, testFromWrenchMissingField) {
  flatbuffers::FlatBufferBuilder fbb;
  robo::geom_msgs::WrenchBuilder wrench_builder(fbb);
  auto wrench_offset = wrench_builder.Finish();
  fbb.Finish(wrench_offset);
  std::span<uint8_t> buf(fbb.GetBufferPointer(), fbb.GetSize());
  const auto* msg = robo::geom_msgs::GetWrench(buf.data());
  auto res = robo::FromMessage(*msg);
  ASSERT_FALSE(res.has_value());
  auto err = res.error();
  EXPECT_EQ(err, robo::ConvertError::kMissingFields);
}

TEST(TestConversion, testFromWrenchWrongCovarianceSize) {
  flatbuffers::FlatBufferBuilder fbb;
  std::vector<float> wrong_cov_data(25, 0.0f);
  auto cov_offset = fbb.CreateVector(wrong_cov_data);
  robo::geom_msgs::WrenchBuilder wrench_builder(fbb);
  auto force = robo::ToMessage(Eigen::Vector3f::Zero());
  wrench_builder.add_force(&force);
  auto torque = robo::ToMessage(Eigen::Vector3f::Zero());
  wrench_builder.add_torque(&torque);
  wrench_builder.add_covariance(cov_offset);
  auto wrench_offset = wrench_builder.Finish();
  fbb.Finish(wrench_offset);
  std::span<uint8_t> buf(fbb.GetBufferPointer(), fbb.GetSize());
  const auto* msg = robo::geom_msgs::GetWrench(buf.data());
  auto res = robo::FromMessage(*msg);
  ASSERT_FALSE(res.has_value());
  auto err = res.error();
  EXPECT_EQ(err, robo::ConvertError::kInvalidSize);
}

TEST(TestConversion, testToOdometryNoCovariance) {
  Eigen::Vector3f t = Eigen::Vector3f::Random();
  Eigen::Quaternionf r = Eigen::Quaternionf::UnitRandom();
  robo::TransformF32 transform(t, r);
  Eigen::Vector3f v = Eigen::Vector3f::Random();
  Eigen::Vector3f w = Eigen::Vector3f::Random();
  robo::TwistF32 twist(v, w);

  {
    robo::OdometryF32 odom(transform, twist);

    flatbuffers::FlatBufferBuilder fbb;
    auto offset = robo::ToMessage(fbb, 1234567890u, odom, "odom");
    fbb.Finish(offset);
    std::span<uint8_t> buf(fbb.GetBufferPointer(), fbb.GetSize());
    const auto* msg = robo::geom_msgs::GetOdometry(buf.data());
    ASSERT_NE(msg->header(), nullptr);
    EXPECT_EQ(msg->header()->stamp_us(), 1234567890u);
    ASSERT_NE(msg->header()->frame_id(), nullptr);
    EXPECT_EQ(msg->header()->frame_id()->str(), "odom");
    ASSERT_NE(msg->pose(), nullptr);
    ASSERT_NE(msg->twist(), nullptr);
    ASSERT_NE(msg->pose()->position(), nullptr);
    EXPECT_THAT(*msg->pose()->position(), ComponentsEq(t));
    ASSERT_NE(msg->pose()->orientation(), nullptr);
    EXPECT_THAT(*msg->pose()->orientation(), ComponentsEq(r));
    ASSERT_NE(msg->twist()->linear(), nullptr);
    EXPECT_THAT(*msg->twist()->linear(), ComponentsEq(v));
    ASSERT_NE(msg->twist()->angular(), nullptr);
    EXPECT_THAT(*msg->twist()->angular(), ComponentsEq(w));
    EXPECT_EQ(msg->pose()->covariance(), nullptr);
    EXPECT_EQ(msg->twist()->covariance(), nullptr);
  }
  float params[13] = {t.x(), t.y(), t.z(), r.x(), r.y(), r.z(), r.w(),
                      v.x(), v.y(), v.z(), w.x(), w.y(), w.z()};
  {
    robo::OdometryViewF32 odom(params);
    flatbuffers::FlatBufferBuilder fbb;
    auto offset = robo::ToMessage(fbb, 1234567890u, odom, "odom");
    fbb.Finish(offset);
    std::span<uint8_t> buf(fbb.GetBufferPointer(), fbb.GetSize());
    const auto* msg = robo::geom_msgs::GetOdometry(buf.data());
    ASSERT_NE(msg->header(), nullptr);
    EXPECT_EQ(msg->header()->stamp_us(), 1234567890u);
    ASSERT_NE(msg->header()->frame_id(), nullptr);
    EXPECT_EQ(msg->header()->frame_id()->str(), "odom");
    ASSERT_NE(msg->pose(), nullptr);
    ASSERT_NE(msg->twist(), nullptr);
    ASSERT_NE(msg->pose()->position(), nullptr);
    ASSERT_NE(msg->pose()->orientation(), nullptr);
    ASSERT_NE(msg->twist()->linear(), nullptr);
    ASSERT_NE(msg->twist()->angular(), nullptr);
    EXPECT_EQ(msg->pose()->covariance(), nullptr);
    EXPECT_EQ(msg->twist()->covariance(), nullptr);

    float result[13] = {
        msg->pose()->position()->x(),    msg->pose()->position()->y(),
        msg->pose()->position()->z(),    msg->pose()->orientation()->x(),
        msg->pose()->orientation()->y(), msg->pose()->orientation()->z(),
        msg->pose()->orientation()->w(), msg->twist()->linear()->x(),
        msg->twist()->linear()->y(),     msg->twist()->linear()->z(),
        msg->twist()->angular()->x(),    msg->twist()->angular()->y(),
        msg->twist()->angular()->z()};

    EXPECT_THAT(result, testing::Pointwise(testing::FloatEq(), params));
  }
}

TEST(TestConversion, testFromOdometryNoCovariance) {
  flatbuffers::FlatBufferBuilder fbb;
  robo::core_msgs::HeaderBuilder header_builder(fbb);
  header_builder.add_stamp_us(0u);
  auto header_offset = header_builder.Finish();

  robo::geom_msgs::Vec3 p(1.0f, 2.0f, 3.0f);
  robo::geom_msgs::Quat q(0.0f, 0.0f, 0.0f, 1.0f);
  robo::geom_msgs::PoseBuilder pose_builder(fbb);
  pose_builder.add_position(&p);
  pose_builder.add_orientation(&q);
  auto pose_offset = pose_builder.Finish();

  robo::geom_msgs::Vec3 lin(1.0f, 2.0f, 3.0f);
  robo::geom_msgs::Vec3 ang(4.0f, 5.0f, 6.0f);
  robo::geom_msgs::TwistBuilder twist_builder(fbb);
  twist_builder.add_linear(&lin);
  twist_builder.add_angular(&ang);
  auto twist_offset = twist_builder.Finish();

  robo::geom_msgs::OdometryBuilder odom_builder(fbb);
  odom_builder.add_header(header_offset);
  odom_builder.add_pose(pose_offset);
  odom_builder.add_twist(twist_offset);
  auto odom_offset = odom_builder.Finish();
  fbb.Finish(odom_offset);
  std::span<uint8_t> buf(fbb.GetBufferPointer(), fbb.GetSize());

  const auto* msg = robo::geom_msgs::GetOdometry(buf.data());
  auto res = robo::FromMessage(*msg);
  ASSERT_TRUE(res.has_value());
  const auto& [stamp_us, fid, odom_res] = res.value();
  EXPECT_EQ(stamp_us, 0u);
  EXPECT_EQ(fid, "");
  const auto& [odom, pose_cov_opt, twist_cov_opt] = odom_res;
  const auto& transform_back = odom.pose();
  const auto& twist_back = odom.twist();
  EXPECT_THAT(p, ComponentsEq(transform_back.translation()));
  EXPECT_THAT(q, ComponentsEq(transform_back.rotation()));
  EXPECT_THAT(lin, ComponentsEq(twist_back.linear()));
  EXPECT_THAT(ang, ComponentsEq(twist_back.angular()));
  EXPECT_FALSE(pose_cov_opt.has_value());
  EXPECT_FALSE(twist_cov_opt.has_value());
}

TEST(TestRoundTrip, testVec3) {
  Eigen::Vector3f v;

  for (auto idx : std::views::iota(0, kNumTrials)) {
    v = Eigen::Vector3f::Random();
    auto msg = robo::ToMessage(v);
    auto v_back = robo::FromMessage(msg);
    EXPECT_EQ(v, v_back) << "Trial " << idx;
  }
}

TEST(TestRoundTrip, testQuat) {
  Eigen::Quaternionf q;

  for (auto idx : std::views::iota(0, kNumTrials)) {
    q = Eigen::Quaternionf::UnitRandom();
    auto msg = robo::ToMessage(q);
    auto q_back = robo::FromMessage(msg);
    EXPECT_EQ(q, q_back) << "Trial " << idx;
  }
}

TEST(TestRoundTrip, testTransform) {
  for (auto idx : std::views::iota(0, kNumTrials)) {
    Eigen::Vector3f t = Eigen::Vector3f::Random();
    Eigen::Quaternionf r = Eigen::Quaternionf::UnitRandom();
    robo::TransformF32 transform(t, r);

    flatbuffers::FlatBufferBuilder fbb;  // <-- moved inside loop
    auto offset = robo::ToMessage(fbb, transform);
    fbb.Finish(offset);
    std::span<uint8_t> buf(fbb.GetBufferPointer(), fbb.GetSize());

    const auto* msg = robo::geom_msgs::GetPose(buf.data());
    auto res = robo::FromMessage(*msg);
    ASSERT_TRUE(res.has_value()) << "Trial " << idx;
    auto [transform_back, cov_opt] = res.value();

    EXPECT_EQ(t, transform_back.translation()) << "Trial " << idx;
    EXPECT_EQ(r.coeffs(), transform_back.rotation().coeffs())
        << "Trial " << idx;
    EXPECT_FALSE(cov_opt.has_value()) << "Trial " << idx;
  }
}

TEST(TestRoundTrip, testTwist) {
  for (auto idx : std::views::iota(0, kNumTrials)) {
    Eigen::Vector3f v = Eigen::Vector3f::Random();
    Eigen::Vector3f w = Eigen::Vector3f::Random();
    robo::TwistF32 twist(v, w);

    flatbuffers::FlatBufferBuilder fbb;  // <-- moved inside loop
    auto offset = robo::ToMessage(fbb, twist);
    fbb.Finish(offset);
    std::span<uint8_t> buf(fbb.GetBufferPointer(), fbb.GetSize());

    const auto* msg = robo::geom_msgs::GetTwist(buf.data());
    auto res = robo::FromMessage(*msg);
    ASSERT_TRUE(res.has_value()) << "Trial " << idx;
    auto [twist_back, cov_opt] = res.value();

    EXPECT_EQ(v, twist_back.linear()) << "Trial " << idx;
    EXPECT_EQ(w, twist_back.angular()) << "Trial " << idx;
    EXPECT_FALSE(cov_opt.has_value()) << "Trial " << idx;
  }
}

TEST(TestRoundTrip, testAccel) {
  for (auto idx : std::views::iota(0, kNumTrials)) {
    Eigen::Vector3f v = Eigen::Vector3f::Random();
    Eigen::Vector3f w = Eigen::Vector3f::Random();
    robo::AccelF32 accel(v, w);

    flatbuffers::FlatBufferBuilder fbb;  // <-- moved inside loop
    auto offset = robo::ToMessage(fbb, accel);
    fbb.Finish(offset);
    std::span<uint8_t> buf(fbb.GetBufferPointer(), fbb.GetSize());

    const auto* msg = robo::geom_msgs::GetAccel(buf.data());
    auto res = robo::FromMessage(*msg);
    ASSERT_TRUE(res.has_value()) << "Trial " << idx;
    auto [accel_back, cov_opt] = res.value();

    EXPECT_EQ(v, accel_back.linear()) << "Trial " << idx;
    EXPECT_EQ(w, accel_back.angular()) << "Trial " << idx;
    EXPECT_FALSE(cov_opt.has_value()) << "Trial " << idx;
  }
}

TEST(TestRoundTrip, testOdometry) {
  for (auto idx : std::views::iota(0, kNumTrials)) {
    Eigen::Vector3f t = Eigen::Vector3f::Random();
    Eigen::Quaternionf r = Eigen::Quaternionf::UnitRandom();
    robo::TransformF32 transform(t, r);
    Eigen::Vector3f v = Eigen::Vector3f::Random();
    Eigen::Vector3f w = Eigen::Vector3f::Random();
    robo::TwistF32 twist(v, w);

    robo::OdometryF32 odom(transform, twist);

    flatbuffers::FlatBufferBuilder fbb;  // <-- moved inside loop
    auto offset = robo::ToMessage(fbb, 1234567890u, odom, "odom");
    fbb.Finish(offset);
    std::span<uint8_t> buf(fbb.GetBufferPointer(), fbb.GetSize());

    const auto* msg = robo::geom_msgs::GetOdometry(buf.data());
    auto res = robo::FromMessage(*msg);
    ASSERT_TRUE(res.has_value()) << "Trial " << idx;
    const auto& [stamp_us, fid, odom_res] = res.value();
    EXPECT_EQ(stamp_us, 1234567890u) << "Trial " << idx;
    EXPECT_EQ(fid, "odom") << "Trial " << idx;
    const auto& [odom_back, pose_cov_opt, twist_cov_opt] = odom_res;
    const auto& transform_back = odom_back.pose();
    const auto& twist_back = odom_back.twist();
    EXPECT_EQ(t, transform_back.translation()) << "Trial " << idx;
    EXPECT_EQ(r.coeffs(), transform_back.rotation().coeffs())
        << "Trial " << idx;
    EXPECT_EQ(v, twist_back.linear()) << "Trial " << idx;
    EXPECT_EQ(w, twist_back.angular()) << "Trial " << idx;
    EXPECT_FALSE(pose_cov_opt.has_value()) << "Trial " << idx;
    EXPECT_FALSE(twist_cov_opt.has_value()) << "Trial " << idx;
  }
}
