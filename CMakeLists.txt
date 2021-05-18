cmake_minimum_required(VERSION 3.14)

SET(ENV{CC} "/usr/bin/clang")
SET(ENV{CXX} "/usr/bin/clang++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++ -v")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++ -lc++abi")

# set the project name and version
project(AEC3 VERSION 1.0)

set(WEBRTC_SRC /data1/tanghaoyu/webrtc/src)
set(WEBRTC_COMPILED /data1/tanghaoyu/webrtc/src/out/Release)
set(WEBRTC_APM_DEBUG_DUMP 0)

set(CMAKE_CXX_STANDARD 14)

add_subdirectory(wavio)
list(APPEND EXTRA_LIBS wavio)

include_directories(
                   ${WEBRTC_SRC}
                   ${WEBRTC_SRC}/third_party/abseil-cpp
                   ${WEBRTC_SRC}/third_party/tcmalloc
                   ${WEBRTC_SRC}/third_party/blink
                   /usr/include/glib-2.0
                   /usr/lib/glib-2.0/include
                   )
link_directories(
                /usr/lib/x86_64-linux-gnu
                ${WEBRTC_COMPILED}/obj
                /usr/lib
                )

link_libraries(
            webrtc
            pthread
            gthread-2.0
            glib-2.0
            )

add_executable(unit_test unit_test.cc)
target_link_libraries(unit_test PUBLIC ${EXTRA_LIBS})

add_executable(demo demo.cc)
target_link_libraries(demo PUBLIC ${EXTRA_LIBS})
add_definitions(-DWEBRTC_APM_DEBUG_DUMP=${WEBRTC_APM_DEBUG_DUMP})
add_definitions(-DWEBRTC_POSIX)
