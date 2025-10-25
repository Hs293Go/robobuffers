add_executable(test_converters src/test_converters.cpp)
target_link_libraries(
  test_converters PRIVATE robobuffers::robobuffers GTest::gtest
                          GTest::gtest_main GTest::gmock)
