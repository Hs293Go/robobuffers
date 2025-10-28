#ifndef ROBOBUFFERS_MESSAGING_HPP_
#define ROBOBUFFERS_MESSAGING_HPP_

#include <string>

#include "flatbuffers/flatbuffer_builder.h"
#include "flatbuffers/verifier.h"
#include "robobuffers/expected.hpp"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"
#include "zmq.hpp"

namespace robo {

template <typename Msg>
concept FlatbuffersTable = std::is_base_of_v<flatbuffers::Table, Msg>;

struct PubOptions {
  int queue_size = 10;  // like ROS
  bool latch = false;   // if true, resend last message to new subs
  std::string topic;    // optional topic prefix
};

struct SubOptions {
  int queue_size = 10;
  std::string topic;
};

template <FlatbuffersTable Msg>
class Subscriber;

template <FlatbuffersTable Msg>
class Publisher;

class Context {
 public:
  explicit Context(int io_threads = 1) : context_(io_threads) {
    if (auto existing = spdlog::get("robo")) {
      logger_ = existing;
    } else {
      logger_ = spdlog::stdout_color_mt("robo");
    }
  }

  /**
   * @brief Create a Subscriber for the given message type. The returned
   * Subscriber can be used to receive messages from the given endpoint.
   *
   * @param endpoint The endpoint to connect to (e.g., "tcp://localhost:5555")
   * @param cb The callback function to invoke when a message is received
   * @param opts Subscription options
   */
  template <FlatbuffersTable Msg, typename Callback>
  Subscriber<Msg> subscribe(const std::string& endpoint, Callback cb,
                            SubOptions opts = {}) {
    logger_->debug("Creating subscriber for endpoint {}", endpoint);
    return Subscriber<Msg>(*this, endpoint, std::move(cb), std::move(opts));
  }

  /**
   * @brief Create a Subscriber for the given message type. The returned
   * Subscriber can be used to receive messages from the given endpoint. This
   * overload takes a function pointer as the callback and deduces the message
   * type from the function parmeter.
   *
   * @param endpoint The endpoint to connect to (e.g., "tcp://localhost:5555")
   * @param cb The callback function to invoke when a message is received
   * @param opts Subscription options
   */
  template <FlatbuffersTable Msg>
  Subscriber<Msg> subscribe(const std::string& endpoint, void (*cb)(const Msg&),
                            SubOptions opts = {}) {
    logger_->debug("Creating subscriber for endpoint {}", endpoint);
    return Subscriber<Msg>(*this, endpoint, cb, std::move(opts));
  }

  template <FlatbuffersTable Msg>
  Publisher<Msg> advertise(const std::string& endpoint, PubOptions opts = {}) {
    logger_->debug("Creating publisher for endpoint {}", endpoint);
    return Publisher<Msg>(*this, endpoint, std::move(opts));
  }

  void enableLogDebug() { logger_->set_level(spdlog::level::debug); }
  void disableLogDebug() { logger_->set_level(spdlog::level::info); }

 private:
  friend class PublisherBase;
  friend class SubscriberBase;

  zmq::context_t context_;
  std::shared_ptr<spdlog::logger> logger_;
};

class PublisherBase {
 protected:
  void publishRaw(std::span<const uint8_t> data) {
    if (!opts_.topic.empty()) {
      socket_.send(zmq::buffer(opts_.topic), zmq::send_flags::sndmore);
    }
    socket_.send(zmq::buffer(data.data(), data.size()), zmq::send_flags::none);
  }

  friend class Context;
  PublisherBase(Context& context, const std::string& endpoint,
                PubOptions opts = {})
      : opts_(std::move(opts)),
        socket_(context.context_, zmq::socket_type::pub),
        logger_(context.logger_) {
    socket_.set(zmq::sockopt::sndhwm, opts_.queue_size);
    socket_.bind(endpoint);
  }

  PubOptions opts_;
  zmq::socket_t socket_;
  std::shared_ptr<spdlog::logger> logger_;
};

template <FlatbuffersTable Msg>
class Publisher : public PublisherBase {
 public:
  using PublisherBase::PublisherBase;

  void publish(flatbuffers::FlatBufferBuilder& fbb,
               flatbuffers::Offset<Msg> msg) {
    fbb.Finish(msg);

    const auto* data = fbb.GetBufferPointer();
    const auto size = fbb.GetSize();

    publishRaw(std::span(data, size));

    if (opts_.latch) {
      last_msg_.assign(data, data + size);
    }
  }

 private:
  std::vector<uint8_t> last_msg_;
};

enum class ReceiveError {
  kTopicFrameFailed,
  kMessageFrameFailed,
  kInvalidMessage,
};

struct TopicAndContent {
  std::string topic;
  zmq::message_t content;
};

class SubscriberBase {
 protected:
  SubOptions opts_;

  expected<TopicAndContent, ReceiveError> receiveRaw() {
    TopicAndContent result;

    zmq::message_t topic_frame;
    zmq::message_t msg;
    if (!opts_.topic.empty()) {
      if (!socket_.recv(topic_frame, zmq::recv_flags::none)) {
        logger_->error("Failed to receive topic frame");
        return unexpected(ReceiveError::kTopicFrameFailed);
      }
      result.topic.assign(static_cast<char*>(topic_frame.data()),
                          topic_frame.size());
    }

    if (!socket_.recv(msg, zmq::recv_flags::none)) {
      logger_->error("Failed to receive message frame for topic {:?}",
                     result.topic);
      return unexpected(ReceiveError::kMessageFrameFailed);
    }
    result.content = std::move(msg);
    return result;
  }

  SubscriberBase(Context& ctx, const std::string& endpoint,
                 SubOptions opts = {})
      : opts_(std::move(opts)),
        socket_(ctx.context_, zmq::socket_type::sub),
        logger_(ctx.logger_) {
    socket_.set(zmq::sockopt::rcvhwm, opts_.queue_size);
    socket_.connect(endpoint);
    socket_.set(zmq::sockopt::subscribe, opts_.topic);
  }

  zmq::socket_t socket_;
  std::shared_ptr<spdlog::logger> logger_;
};

template <FlatbuffersTable Msg>
class Subscriber : public SubscriberBase {
 public:
  using Callback = std::function<void(const Msg&)>;

  Subscriber(Context& ctx, const std::string& endpoint, Callback cb,
             SubOptions opts = {})
      : SubscriberBase(ctx, endpoint, std::move(opts)),
        callback_(std::move(cb)) {}

  void spin() {
    while (true) {
      auto res = receiveRaw();
      if (!res) {
        continue;
      }
      const auto& [topic, content] = res.value();

      flatbuffers::Verifier verifier(
          reinterpret_cast<const uint8_t*>(content.data()), content.size());
      if (!verifier.VerifyBuffer<Msg>()) {
        logger_->error("Message verification failed for topic '{}'", topic);
        continue;
      }

      const auto root = flatbuffers::GetRoot<Msg>(content.data());
      callback_(*root);
    }
  }

 private:
  friend class Context;

  Callback callback_;
};
}  // namespace robo

#endif  // ROBOBUFFERS_MESSAGING_HPP_
