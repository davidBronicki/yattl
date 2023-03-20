cmake_minimum_required(VERSION 3.7.2)
project(YATTL CXX)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wextra -Wpedantic -O0 -fsanitize=address")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -Wextra -Wpedantic -O3")

add_executable(BasicExample "basics.cpp")

set_target_properties(BasicExample PROPERTIES CXX_STANDARD 20)
