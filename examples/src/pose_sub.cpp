#include <iostream>
#include <thread>
#include <zmq.hpp>

#include "robobuffers/converters.hpp"
#include "robobuffers/messaging.hpp"

using namespace std::chrono_literals;
void poseStampedCallback(const robo::geom_msgs::PoseStamped& msg) {
  auto res = robo::FromMessage(msg);
  if (!res) {
    std::cerr << "Error converting Pose message: "
              << static_cast<int>(res.error()) << "\n";
    return;
  }
  const auto& [time_us, _, transform] = res.value();
  std::cout << "Received PoseStamped:\n"
            << "  translation: "
            << transform.transform.translation().transpose() << "\n"
            << "  rotation (xyzw): "
            << transform.transform.rotation().coeffs().transpose() << "\n";
}

void SubscriberThread(robo::Context& context) {
  robo::SubOptions opts;
  opts.queue_size = 5;
  opts.topic = "pose";

  // Subscriber handles deserialization and callback invocation
  robo::Subscriber sub =
      context.subscribe("tcp://localhost:5555", poseStampedCallback, opts);

  sub.spin();  // Blocking receive loop with callback
}

int main() {
  robo::Context context(1);
  context.enableLogDebug();
  std::jthread sub_thread(SubscriberThread, std::ref(context));
  return 0;
}
