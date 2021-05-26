# distutils: language = c++

from libc.stdint cimport int16_t

from libcpp cimport bool
from libcpp.cast cimport reinterpret_cast
from libcpp.memory cimport unique_ptr, allocator, make_unique
from libcpp.vector cimport vector


cdef extern from "api/audio/echo_canceller3_config.h" namespace "webrtc":
    cdef struct Filter:
        bool export_linear_aec_output
    cdef struct EchoCanceller3Config:
        Filter filter

cdef extern from "api/audio/echo_canceller3_factory.h" namespace "webrtc":
    cdef cppclass EchoCanceller3Factory:
       EchoCanceller3Factory() except +
       EchoCanceller3Factory(EchoCanceller3Config) except +
       unique_ptr[EchoControl] Create(
           int sample_rate_hz,
           int num_render_channels,
           int num_capture_channels,
           )

cdef extern from "api/audio/audio_frame.h" namespace "webrtc":
    cdef cppclass AudioFrame:
        pass

cdef extern from "modules/audio_processing/audio_buffer.h" namespace "webrtc":
    cdef cppclass AudioBuffer:
        void CopyFrom(
            const int16_t* const interleaved_data, const EchoCanceller3Config stream_config)
        pass

cdef extern from "modules/audio_processing/aec3/echo_canceller3.h" namespace "webrtc":
    cdef cppclass EchoCanceller3:
        EchoCanceller3Config CreateDefaultConfig(size_t num_render_channels, size_t num_capture_channels)

cdef extern from "modules/audio_processing/high_pass_filter.h" namespace "webrtc":
    cdef cppclass HighPassFilter:
        HighPassFilter(int sample_rate_hz, size_t num_channels)
        void Reset()
        void Reset(size_t num_channels)

cdef extern from "api/audio/echo_control.h" namespace "webrtc":
    cdef cppclass EchoControl:
        void AnalyzeRender(AudioBuffer* render)
        void AnalyzeCapture(AudioBuffer* capture)
        void ProcessCapture(AudioBuffer* capture, bool level_change)
        ProcessCapture(AudioBuffer* capture, AudioBuffer* linear_output, bool level_change)
