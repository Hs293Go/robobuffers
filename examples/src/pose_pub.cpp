#include <iostream>
#include <thread>
#include <zmq.hpp>

#include "robobuffers/converters.hpp"
#include "robobuffers/messaging.hpp"

using namespace std::chrono_literals;

void PublisherThread(robo::Context& context) {
  robo::PubOptions opts{.queue_size = 5, .topic = "pose"};

  std::this_thread::sleep_for(500ms);  // allow subscriber to connect

  auto endpoint = robo::EndpointBuilder(robo::Transport::kTcp)
                      .port(5555)
                      .mode(robo::ConnMode::kBind)
                      .build();
  if (!endpoint) {
    std::cerr << "Error building endpoint\n";
    return;
  }
  robo::Publisher pub =
      context.advertise<robo::geom_msgs::PoseStamped>(endpoint.value(), opts);
  while (true) {
    flatbuffers::FlatBufferBuilder fbb;

    robo::TransformF32 transform(Eigen::Vector3f(1.0f, 2.0f, 3.0f),
                                 Eigen::Quaternionf::UnitRandom());

    auto pose = robo::ToMessage(fbb, std::chrono::system_clock::now(),
                                transform);  // convert to Pose message
    std::cout << "Publishing PoseStamped\n";
    pub.publish(fbb, pose);
    std::this_thread::sleep_for(1s);
  }
}

int main() {
  robo::Context context(1);
  context.enableLogDebug();
  std::jthread pub_thread(PublisherThread, std::ref(context));
  return 0;
}
