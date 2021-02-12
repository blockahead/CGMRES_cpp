cmake_minimum_required(VERSION 3.0.0)
project(CGMRES VERSION 0.1.0)

include(CTest)
enable_testing()

add_executable(CGMRES main.cpp cgmres.cpp)

set(CMAKE_CXX_FLAGS "-O3 -Wall")
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
