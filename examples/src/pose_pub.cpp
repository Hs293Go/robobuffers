#include <iostream>
#include <thread>
#include <zmq.hpp>

#include "robobuffers/converters.hpp"
#include "robobuffers/messaging.hpp"

using namespace std::chrono_literals;

void PublisherThread(robo::Context& context) {
  robo::PubOptions opts{.queue_size = 5, .latch = false, .topic = "pose"};

  std::this_thread::sleep_for(500ms);  // allow subscriber to connect

  robo::Publisher pub =
      context.advertise<robo::geom_msgs::PoseStamped>("tcp://*:5555", opts);
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
