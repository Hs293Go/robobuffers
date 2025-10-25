#ifndef ROBOBUFFERS_MESSAGING_HPP_
#define ROBOBUFFERS_MESSAGING_HPP_

#include <string>

#include "flatbuffers/flatbuffer_builder.h"
#include "robobuffers/converters.hpp"
#include "zmq.hpp"

namespace robo {

struct PubOptions {
  int queue_size = 10;  // like ROS
  bool latch = false;   // if true, resend last message to new subs
  std::string topic;    // optional topic prefix
};

struct SubOptions {
  int queue_size = 10;
  std::string topic;
};

template <typename Msg>
class Subscriber;

template <typename Msg>
class Publisher;

class Context {
 public:
  Context(int io_threads) : context_(io_threads) {}

  template <typename Msg>
  Subscriber<Msg> subscribe(const std::string& endpoint,
                            typename Subscriber<Msg>::Callback cb,
                            SubOptions opts = {}) {
    return Subscriber<Msg>(*this, endpoint, std::move(cb), std::move(opts));
  }

  zmq::context_t& context() { return context_; }

  template <typename Msg>
  Publisher<Msg> advertise(const std::string& endpoint, PubOptions opts = {}) {
    return Publisher<Msg>(*this, endpoint, std::move(opts));
  }

 private:
  zmq::context_t context_{1};
};

template <typename Msg>
class Publisher {
 public:
  void publish(flatbuffers::FlatBufferBuilder& fbb,
               flatbuffers::Offset<Msg> msg) {
    fbb.Finish(msg);

    const auto* data = fbb.GetBufferPointer();
    const auto size = fbb.GetSize();

    if (!opts_.topic.empty()) {
      socket_.send(zmq::buffer(opts_.topic), zmq::send_flags::sndmore);
    }
    socket_.send(zmq::buffer(data, size), zmq::send_flags::none);

    if (opts_.latch) {
      last_msg_.assign(data, data + size);
    }
  }

 private:
  friend class Context;
  Publisher(Context& context, const std::string& endpoint, PubOptions opts = {})
      : opts_(std::move(opts)),
        socket_(context.context(), zmq::socket_type::pub) {
    socket_.set(zmq::sockopt::sndhwm, opts_.queue_size);
    socket_.bind(endpoint);
  }
  PubOptions opts_;
  zmq::socket_t socket_;
  std::vector<uint8_t> last_msg_;
};

template <typename Msg>
class Subscriber {
 public:
  using Callback = std::function<void(const Msg&)>;

  void spin() {
    zmq::message_t topic_frame;
    zmq::message_t msg;

    while (true) {
      if (!opts_.topic.empty()) {
        if (!socket_.recv(topic_frame, zmq::recv_flags::none)) {
          continue;
        }
      }
      if (!socket_.recv(msg, zmq::recv_flags::none)) {
        continue;
      }
      flatbuffers::Verifier verifier(
          reinterpret_cast<const uint8_t*>(msg.data()), msg.size());
      if (!verifier.VerifyBuffer<Msg>()) {
        continue;
      }

      const auto root = flatbuffers::GetRoot<Msg>(msg.data());
      callback_(*root);
    }
  }

 private:
  friend class Context;
  Subscriber(Context& ctx, const std::string& endpoint, Callback cb,
             SubOptions opts = {})
      : opts_(std::move(opts)),
        socket_(ctx.context(), zmq::socket_type::sub),
        callback_(std::move(cb)) {
    socket_.set(zmq::sockopt::rcvhwm, opts_.queue_size);
    socket_.connect(endpoint);
    socket_.set(zmq::sockopt::subscribe, opts_.topic);
  }

  SubOptions opts_;
  zmq::socket_t socket_;
  Callback callback_;
};
}  // namespace robo

#endif  // ROBOBUFFERS_MESSAGING_HPP_
