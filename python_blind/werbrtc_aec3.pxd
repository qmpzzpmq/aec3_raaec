# distutils: language = c++

cdef extern from "api/audio/echo_canceller3_factory.h" namespace "webrtc":
    cdef struct EchoCanceller3Config:
        pass

# def extern from "api/audio/echo_canceller3_factory.h"" namespace "webrtc":
#    cdef cppclass EchoCanceller3Factory:
#        EchoCanceller3Factory() except +
#        EchoCanceller3Factory(EchoCanceller3Config) except +
#        Create(
#            int sample_rate_hz,
#            int num_render_channels,
#            int num_capture_channels,
#            )
#        
# 
# 
# def extern from "api/audio/audio_frame.h" namespace "webrtc":
# 
# def extern from "modules/audio_processing/audio_buffer.h" namespace "webrtc":
# 
# def extern from "modules/audio_processing/high_pass_filter.h" namespace "webrtc":
# 
# def extern from "api/audio/audio_frame.h" namespace "webrtc":
# 
# 