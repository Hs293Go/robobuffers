#include <chrono>
#include <iostream>
#include <thread>
#include <zmq.hpp>

#include "robobuffers/converters.hpp"
#include "robobuffers/messaging.hpp"

using namespace std::chrono_literals;

void PublisherThread(robo::Context& context) {
  robo::PubOptions opts{.queue_size = 5, .latch = false, .topic = "pose"};

  std::this_thread::sleep_for(500ms);  // allow subscriber to connect

  // auto pub = context.
  auto pub = context.advertise<robo::geom_msgs::Pose>("tcp://*:5555", opts);
  while (true) {
    flatbuffers::FlatBufferBuilder fbb;

    robo::TransformF32 transform(Eigen::Vector3f(1.0f, 2.0f, 3.0f),
                                 Eigen::Quaternionf::UnitRandom());

    auto pose = robo::ToMessage(fbb, transform);  // convert to Pose message
    std::cout << "Publishing PoseStamped\n";
    pub.publish(fbb, pose);
    std::this_thread::sleep_for(1s);
  }
}

void SubscriberThread(robo::Context& context) {
  robo::SubOptions opts;
  opts.queue_size = 5;
  opts.topic = "pose";

  // Subscriber handles deserialization and callback invocation
  robo::Subscriber sub = context.subscribe<robo::geom_msgs::Pose>(
      "tcp://localhost:5555",
      [](auto&& msg) {
        auto res = robo::FromMessage(msg);
        if (!res) {
          std::cerr << "Error converting Pose message: "
                    << static_cast<int>(res.error()) << "\n";
          return;
        }
        auto transform = res.value().transform;
        std::cout << "Received PoseStamped:\n"
                  << "  translation: " << transform.translation().transpose()
                  << "\n"
                  << "  rotation (xyzw): "
                  << transform.rotation().coeffs().transpose() << "\n";
      },
      opts);

  sub.spin();  // Blocking receive loop with callback
}

int main() {
  robo::Context context(1);
  std::thread pub_thread(PublisherThread, std::ref(context));
  std::thread sub_thread(SubscriberThread, std::ref(context));
  pub_thread.join();
  sub_thread.join();
  return 0;
}
