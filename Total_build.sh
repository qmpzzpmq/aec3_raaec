#!/bin/bash
./Build_webrtc.sh
mkdir build && cd build
cmake .. -DWEBRTC_SRC=../webrtc/src -DWEBRTC_COMPILED=../webrtc/src/out/Release
make