cmake_minimum_required(VERSION 3.20)
project(robobuffers LANGUAGES CXX)

option(ROBOBUFFERS_BUILD_TESTING "Build robobuffers tests" ON)
option(ROBOBUFFERS_BUILD_EXAMPLES "Build robobuffers examples" ON)

find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)

add_subdirectory(third_party SYSTEM)

# --- FlatBuffers code generation
set(ROBOBUFFERS_CODEGEN_BASE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)
set(ROBOBUFFERS_CODEGEN_DIR ${ROBOBUFFERS_CODEGEN_BASE_DIR}/robobuffers)

set(MSG_GROUPS core geom)
set(core_MSG_TYPES header)
set(geom_MSG_TYPES
    accel
    accel_stamped
    odometry
    pose
    pose_stamped
    quat
    twist
    twist_stamped
    vec3
    wrench
    wrench_stamped)
foreach(GRP IN LISTS MSG_GROUPS)
  foreach(MSG IN LISTS ${GRP}_MSG_TYPES)
    set(_new_fbs_file ${CMAKE_CURRENT_SOURCE_DIR}/msgs/${GRP}/${MSG}.fbs)
    if(NOT EXISTS ${_new_fbs_file})
      message(FATAL_ERROR "FlatBuffers schema file not found: ${_new_fbs_file}")
    endif()
    list(APPEND FBS_FILES ${_new_fbs_file})
    list(APPEND MSG_FILES ${ROBOBUFFERS_CODEGEN_DIR}/${MSG}_generated.h)
  endforeach()
endforeach()

add_custom_command(
  OUTPUT ${MSG_FILES}
  COMMAND flatc --cpp -I ${CMAKE_CURRENT_SOURCE_DIR}/msgs -o
          ${ROBOBUFFERS_CODEGEN_DIR} ${FBS_FILES}
  DEPENDS ${FBS_FILES} flatc
  COMMENT "Generating FlatBuffers C++ headers")

add_custom_target(robobuffers_codegen DEPENDS ${MSG_FILES})

# --- Header-only library for generated types
add_library(robobuffers_msgs INTERFACE)
target_sources(
  robobuffers_msgs INTERFACE FILE_SET HEADERS BASE_DIRS
                             ${ROBOBUFFERS_CODEGEN_BASE_DIR} FILES ${MSG_FILES})
add_dependencies(robobuffers_msgs robobuffers_codegen)
target_link_libraries(robobuffers_msgs INTERFACE flatbuffers)
add_library(robobuffers::msgs ALIAS robobuffers_msgs)
set_target_properties(robobuffers_msgs PROPERTIES EXPORT_NAME msgs)

add_library(robobuffers_robobuffers INTERFACE)
target_sources(
  robobuffers_robobuffers
  INTERFACE
    FILE_SET
    HEADERS
    BASE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/robobuffers/geom.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/robobuffers/expected.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/robobuffers/third_party/tl/expected.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/robobuffers/third_party/concurrentqueue.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/robobuffers/messaging.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/robobuffers/converters.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/robobuffers/meta.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/robobuffers/thread_pool.hpp)
target_compile_features(robobuffers_robobuffers INTERFACE cxx_std_20)
target_link_libraries(
  robobuffers_robobuffers INTERFACE robobuffers::msgs Eigen3::Eigen cppzmq
                                    spdlog::spdlog)
add_library(robobuffers::robobuffers ALIAS robobuffers_robobuffers)

# --- Example
if(ROBOBUFFERS_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

include(CTest)
set(BUILD_TESTING ${ROBOBUFFERS_BUILD_TESTING})
if(ROBOBUFFERS_BUILD_TESTING)
  add_subdirectory(tests)
endif()
