add_executable(pose_pub src/pose_pub.cpp)
target_link_libraries(pose_pub PRIVATE robobuffers::robobuffers)
add_executable(pose_sub src/pose_sub.cpp)
target_link_libraries(pose_sub PRIVATE robobuffers::robobuffers)
