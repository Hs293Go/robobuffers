set(FLATBUFFERS_BUILD_TESTS
    OFF
    CACHE BOOL "Do not build tests for third party module" FORCE)
add_subdirectory(flatbuffers SYSTEM)

set(CPPZMQ_BUILD_TESTS
    OFF
    CACHE BOOL "Do not build tests for zmq" FORCE)
add_subdirectory(cppzmq SYSTEM)

if(ROBOBUFFERS_BUILD_TESTING)
  add_subdirectory(googletest SYSTEM)
endif()
