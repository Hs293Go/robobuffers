#ifndef ROBOBUFFERS_MESSAGING_HPP_
#define ROBOBUFFERS_MESSAGING_HPP_

#include <string>

#include "flatbuffers/flatbuffer_builder.h"
#include "flatbuffers/verifier.h"
#include "robobuffers/meta.hpp"
#include "robobuffers/expected.hpp"
#include "spdlog/fmt/fmt.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"
#include "zmq.hpp"

namespace robo {

enum class PubSubError {
  kTopicFrameFailed,
  kMessageFrameFailed,
  kInvalidMessage,
  kMissingName,
  kMissingPort,
  kPortOutOfRange,
  kIrrelevantOptions
};

enum class ConnMode { kBind, kConnect };

enum class Transport { kInproc, kIpc, kTcp };

class Endpoint {
 public:
  const std::string& str() const { return endpoint_; }

  ConnMode mode() const { return mode_; }

 private:
  Endpoint(const std::string& endpoint, ConnMode mode)
      : endpoint_(endpoint), mode_(mode) {}
  friend class EndpointBuilder;
  std::string endpoint_;
  ConnMode mode_;
};

class EndpointBuilder {
 public:
  EndpointBuilder() = delete;

  explicit EndpointBuilder(Transport t) { transport_ = t; }

  EndpointBuilder& host(const std::string& host) {
    host_ = host;
    return *this;
  }

  EndpointBuilder& name(const std::string& name) {
    name_ = name;
    return *this;
  }

  EndpointBuilder& port(int port) {
    port_ = port;
    return *this;
  }

  EndpointBuilder& mode(ConnMode mode) {
    mode_ = mode;
    return *this;
  }

  expected<Endpoint, PubSubError> build() const {
    // Don't use fmt to build std::strings so this builder can be copied out of
    // the project
    switch (transport_) {
      case Transport::kInproc:
        if (!name_.has_value()) {
          return unexpected(PubSubError::kMissingName);
        }
        if (port_.has_value() || host_.has_value()) {
          return unexpected(PubSubError::kIrrelevantOptions);
        }

        return Endpoint("inproc://" + name_.value(), mode_);
      case Transport::kIpc:
        if (!name_.has_value()) {
          return unexpected(PubSubError::kMissingName);
        }
        if (port_.has_value() || host_.has_value()) {
          return unexpected(PubSubError::kIrrelevantOptions);
        }
        return Endpoint("ipc://" + name_.value(), mode_);
      case Transport::kTcp:
        if (!port_.has_value()) {
          return unexpected(PubSubError::kMissingPort);
        }

        switch (mode_) {
          case ConnMode::kBind:
            return Endpoint("tcp://*:" + std::to_string(port_.value()),
                            ConnMode::kBind);
            break;
          case ConnMode::kConnect:
            if (port_ <= 0 || port_ > 65535) {
              return unexpected(PubSubError::kPortOutOfRange);
            }

            return Endpoint("tcp://" + host_.value_or("localhost") + ":" +
                                std::to_string(port_.value()),
                            ConnMode::kConnect);

            break;
        }
    }
    unreachable();
  }

 private:
  Transport transport_;
  std::optional<std::string> host_;
  std::optional<std::string> name_;
  std::optional<int> port_;
  ConnMode mode_ = ConnMode::kBind;
};

template <typename Msg>
concept FlatbuffersTable = std::is_base_of_v<flatbuffers::Table, Msg>;

struct PubOptions {
  int queue_size = 10;    // like ROS
  int linger_ms = 0;      // socket linger time in ms
  bool immediate = true;  // ZMQ_IMMEDIATE: drop messages if no subs connected
  int tcp_keepalive = 1;  // 0/1/2 per libzmq
  std::string topic;      // optional topic prefix
  // bool latch = false;   // if true, resend last message to new subs
};

struct SubOptions {
  int queue_size = 10;
  int linger_ms = 0;
  bool immediate = true;
  int tcp_keepalive = 1;
  bool conflate = false;  // ZMQ_CONFLATE: keep only last message
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
  Subscriber<Msg> subscribe(const Endpoint& endpoint, Callback cb,
                            SubOptions opts = {}) {
    logger_->debug("Creating subscriber for endpoint {} with options {}",
                   endpoint, opts);
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
  Subscriber<Msg> subscribe(const Endpoint& endpoint, void (*cb)(const Msg&),
                            SubOptions opts = {}) {
    logger_->debug("Creating subscriber for endpoint {} with options {}",
                   endpoint, opts);
    return Subscriber<Msg>(*this, endpoint, cb, std::move(opts));
  }

  template <FlatbuffersTable Msg>
  Publisher<Msg> advertise(const Endpoint& endpoint, PubOptions opts = {}) {
    logger_->debug("Creating publisher for endpoint {} with options {}",
                   endpoint, opts);
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
    // Always send topic frame, even if empty
    socket_.send(zmq::buffer(opts_.topic), zmq::send_flags::sndmore);
    socket_.send(zmq::buffer(data.data(), data.size()), zmq::send_flags::none);
  }

  friend class Context;
  PublisherBase(Context& ctx, const Endpoint& endpoint, PubOptions opts = {})
      : opts_(std::move(opts)),
        socket_(ctx.context_, zmq::socket_type::pub),
        logger_(ctx.logger_) {
    socket_.set(zmq::sockopt::sndhwm, opts_.queue_size);
    socket_.set(zmq::sockopt::linger, opts_.linger_ms);
    socket_.set(zmq::sockopt::immediate, opts_.immediate);
    socket_.set(zmq::sockopt::tcp_keepalive, opts_.tcp_keepalive);

    if (endpoint.mode() == ConnMode::kBind) {
      socket_.bind(endpoint.str());
      logger_->debug("PUB bind {}", endpoint.str());
    } else {
      socket_.connect(endpoint.str());
      logger_->debug("PUB connect {}", endpoint.str());
    }
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

    // if (opts_.latch) {
    // last_msg_.assign(data, data + size);
    // }
  }

 private:
  // std::vector<uint8_t> last_msg_;
};

struct TopicAndContent {
  std::string topic;
  zmq::message_t content;
};

class SubscriberBase {
 protected:
  SubOptions opts_;

  expected<TopicAndContent, PubSubError> receiveRaw() {
    TopicAndContent result;

    zmq::message_t topic_frame;
    zmq::message_t msg;

    if (!socket_.recv(topic_frame, zmq::recv_flags::none)) {
      logger_->error("Failed to receive topic frame");
      return unexpected(PubSubError::kTopicFrameFailed);
    }
    result.topic.assign(static_cast<char*>(topic_frame.data()),
                        topic_frame.size());

    if (!socket_.recv(msg, zmq::recv_flags::none)) {
      logger_->error("Failed to receive message frame for topic {}",
                     result.topic);
      return unexpected(PubSubError::kMessageFrameFailed);
    }
    result.content = std::move(msg);
    return result;
  }

  SubscriberBase(Context& ctx, const Endpoint& endpoint, SubOptions opts = {})
      : opts_(std::move(opts)),
        socket_(ctx.context_, zmq::socket_type::sub),
        logger_(ctx.logger_) {
    socket_.set(zmq::sockopt::rcvhwm, opts_.queue_size);
    socket_.set(zmq::sockopt::linger, opts_.linger_ms);
    socket_.set(zmq::sockopt::immediate, opts_.immediate);
    socket_.set(zmq::sockopt::tcp_keepalive, opts_.tcp_keepalive);
    if (opts_.conflate) {
      socket_.set(zmq::sockopt::conflate, 1);
    }

    if (endpoint.mode() == ConnMode::kBind) {
      socket_.bind(endpoint.str());
      logger_->debug("SUB bind {}", endpoint.str());
    } else {
      socket_.connect(endpoint.str());
      logger_->debug("SUB connect {}", endpoint.str());
    }

    socket_.set(zmq::sockopt::subscribe, opts_.topic);
  }

  zmq::socket_t socket_;
  std::shared_ptr<spdlog::logger> logger_;
};

template <FlatbuffersTable Msg>
class Subscriber : public SubscriberBase {
 public:
  using Callback = std::function<void(const Msg&)>;

  Subscriber(Context& ctx, const Endpoint& endpoint, Callback cb,
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

namespace fmt {
template <>
struct formatter<robo::Endpoint> : formatter<std::string_view> {
  template <typename FormatContext>
  auto format(const robo::Endpoint& endpoint, FormatContext& ctx) {
    return format_to(ctx.out(), "{}", endpoint.str());
  }
};

template <>
struct formatter<robo::ConnMode> : formatter<std::string_view> {
  template <typename FormatContext>
  auto format(const robo::ConnMode& mode, FormatContext& ctx) {
    std::string mode_str;
    switch (mode) {
      case robo::ConnMode::kBind:
        mode_str = "bind";
        break;
      case robo::ConnMode::kConnect:
        mode_str = "connect";
        break;
    }
    return format_to(ctx.out(), "{}", mode_str);
  }
};

template <>
struct formatter<robo::SubOptions> : formatter<std::string_view> {
  template <typename FormatContext>
  auto format(const robo::SubOptions& opts, FormatContext& ctx) {
    return format_to(ctx.out(),
                     "{{queue_size={}, linger_ms={}, immediate={}, "
                     "tcp_keepalive={}, conflate={}, topic='{}'}}",
                     opts.queue_size, opts.linger_ms, opts.immediate,
                     opts.tcp_keepalive, opts.conflate, opts.topic);
  }
};

template <>
struct formatter<robo::PubOptions> : formatter<std::string_view> {
  constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const robo::PubOptions& opts, FormatContext& ctx) {
    return format_to(ctx.out(),
                     "{{queue_size={}, linger_ms={}, immediate={}, "
                     "tcp_keepalive={}, topic='{}'}}",
                     opts.queue_size, opts.linger_ms, opts.immediate,
                     opts.tcp_keepalive, opts.topic);
  }
};

}  // namespace fmt

#endif  // ROBOBUFFERS_MESSAGING_HPP_
