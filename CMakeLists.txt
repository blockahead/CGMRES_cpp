cmake_minimum_required(VERSION 3.0.0)
project(CGMRES_cpp)

# Semiactive damper model
add_executable(semiactive_damper
  semiactive_damper/main.cpp
)
target_include_directories(semiactive_damper
  PUBLIC include
  PUBLIC semiactive_damper
)

# Mass spring damper model
add_executable(mass_spring_damper
  mass_spring_damper/main.cpp
)
target_include_directories(mass_spring_damper
  PUBLIC include
  PUBLIC mass_spring_damper
)

# Arm type inverted pendulum model
add_executable(arm_type_inverted_pendulum
  arm_type_inverted_pendulum/main.cpp
)
target_include_directories(arm_type_inverted_pendulum
  PUBLIC include
  PUBLIC arm_type_inverted_pendulum
)

set(CMAKE_CXX_FLAGS "-O3 -Wall")
