#!/bin/bash
mkdir webrtc
cd webrtc
PWD=`pwd`
git clone https://chromium.googlesource.com/chromium/tools/depot_tools.git
export PATH=$PWD/depot_tools:$PATH
fetch --nohooks webrtc
gclient sync
cd src
git checkout -b m89 branch-heads/4389
gclient sync
gn gen out/Release --args="is_debug=false"
ninja -C out/Release
echo "please using cmake with variable -DWEBRTC_SRC=$PWD/src -DWEBRTC_COMPILED=$PWD/src/out/Release"