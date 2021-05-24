# distutils: language=c++

from webrtc_aec3 cimport HighPassFilter as webrtc_HighPassFilter
from webrtc_aec3 cimport EchoControl as webrtc_EchoControl
from webrtc_aec3 cimport AudioBuffer as webrtc_AudioBuffer
from webrtc_aec3 cimport EchoCanceller3Factory as webrtc_EchoCanceller3Factory
from webrtc_aec3 cimport EchoCanceller3 as webrtc_EchoCanceller3
from webrtc_aec3 cimport EchoCanceller3Config as webrtc_EchoCanceller3Config

cdef class AEC3:
    cdef:
        unique_ptr[webrtc_EchoControl] echo_controler
        webrtc_HighPassFilter *hp_filter
        webrtc_AudioBuffer *audio_buffer
        readonly int fs
        readonly int num_ref_channel
        readonly int num_rec_channel
        readonly int bytes_per_frame
        readonly int bits_per_sample

    def __cinit__(
            self, num_ref_channel=1, num_rec_channel=1,
            fs=16000, bits_per_sample=16):
        print("__cinit__ method executed")
        self.num_ref_channel = num_ref_channel
        self.num_rec_channel = num_rec_channel
        self.fs = fs
        self.bits_per_sample = bits_per_sample
        cdef int samples_per_frame = fs / 100
        self.bytes_per_frame = samples_per_frame * bits_per_sample / 8
        self.hp_filter = new webrtc_HighPassFilter(fs, num_rec_channel)
        cdef webrtc_EchoCanceller3Config aec3_config;
        aec3_config.filter.export_linear_aec_output = True
        cdef webrtc_EchoCanceller3Factory *aec3_factory = new webrtc_EchoCanceller3Factory(aec3_config)
        self.echo_controler = aec3_factory[0].Create(fs, num_ref_channel, num_rec_channel)

    def __call__(self):
        print("__call__method executed")

    def __dealloc__(self):
        print("__dealloc__method executed")
        del self.hp_filter
        # del self.audio_buffer
