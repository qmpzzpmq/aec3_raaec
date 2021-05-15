cmake_minimum_required(VERSION 3.14)

# set the project name and version
project(AEC3 VERSION 1.0)

set(CMAKE_CXX_STANDARD 14)
set(WEBRTC_SRC /data1/tanghaoyu/webrtc/src)
set(WEBRTC_COMPILED /data1/tanghaoyu/webrtc/src/out/Release)

add_subdirectory(wavio)
list(APPEND EXTRA_LIBS wavio)

include_directories(
                   ${WEBRTC_SRC}
                   ${WEBRTC_SRC}/third_party/abseil-cpp
                   )
link_directories(
                ${WEBRTC_COMPILED}/obj/api/audio/
                ${WEBRTC_COMPILED}/obj/modules/audio_processing/aec3
                )

# add_subdirectory(src)

add_executable(unit_test unit_test.cc)
target_link_libraries(unit_test PUBLIC ${EXTRA_LIBS})

add_executable(demo demo.cc)
target_link_libraries(demo PUBLIC ${EXTRA_LIBS})
