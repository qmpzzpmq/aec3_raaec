# distutils: language=c++

import numpy as np
cimport numpy as cnp

from webrtc_aec3 cimport HighPassFilter as webrtc_HighPassFilter
from webrtc_aec3 cimport EchoControl as webrtc_EchoControl
from webrtc_aec3 cimport AudioBuffer as webrtc_AudioBuffer
from webrtc_aec3 cimport EchoCanceller3Factory as webrtc_EchoCanceller3Factory
from webrtc_aec3 cimport EchoCanceller3 as webrtc_EchoCanceller3
from webrtc_aec3 cimport EchoCanceller3Config as webrtc_EchoCanceller3Config

ctypedef cnp.int16_t DTYPE_t
ctypedef int16_t * INT16_T_PTR

cdef vector[int16_t] arrayToVector(cnp.ndarray[DTYPE_t, ndim=1] array):
    cdef long size = array.size
    cdef vector[int16_t] vec
    cdef long i
    for i in range(size):
        vec.push_back(array[i])
    return vec

cdef class AEC3:
    cdef:
        unique_ptr[webrtc_EchoControl] echo_controler
        webrtc_HighPassFilter *hp_filter
        webrtc_AudioBuffer *audio_buffer
        webrtc_EchoCanceller3Config aec3_config
        readonly int fs
        readonly int num_ref_channel
        readonly int num_rec_channel
        readonly int bytes_per_frame
        readonly int bits_per_sample

    def __cinit__(
            self, int num_ref_channel=1, int num_rec_channel=1,
            int fs=16000, int bits_per_sample=16):
        print("__cinit__ method executed")
        self.num_ref_channel = num_ref_channel
        self.num_rec_channel = num_rec_channel
        self.fs = fs
        self.bits_per_sample = bits_per_sample
        cdef int samples_per_frame = int(fs / 100)
        self.bytes_per_frame = int(samples_per_frame * bits_per_sample / 8)
        self.hp_filter = new webrtc_HighPassFilter(fs, num_rec_channel)
        self.aec3_config.filter.export_linear_aec_output = True
        cdef webrtc_EchoCanceller3Factory *aec3_factory = new webrtc_EchoCanceller3Factory(
            self.aec3_config)
        # self.echo_controler = aec3_factory[0].Create(
        #     fs, num_ref_channel, num_rec_channel)
        self.echo_controler = aec3_factory[0].Create(
            16000, 1, 1)

    def process_chunk(
            self,
            cnp.ndarray[DTYPE_t, ndim=1, mode = 'c'] ref,
            cnp.ndarray[DTYPE_t, ndim=1, mode = 'c'] rec,
        ):
        print("process_chunk method executed")
        cdef unique_ptr[webrtc_AudioBuffer] ref_buffer = make_unique[webrtc_AudioBuffer](
            self.fs, self.num_ref_channel,
            self.fs, self.num_ref_channel,
            self.fs, self.num_ref_channel,
        )
        cdef unique_ptr[webrtc_AudioBuffer] rec_buffer = make_unique[webrtc_AudioBuffer](
            self.fs, self.num_rec_channel,
            self.fs, self.num_rec_channel,
            self.fs, self.num_rec_channel,
        )
        # ref_vecotr = arrayToVector(ref)
        cdef cnp.ndarray[DTYPE_t, ndim=1, mode = 'c'] np_buff = np.ascontiguousarray(
            ref, dtype = np.int16_t)
        cdef int16_t* ref_ptr = &np_buff[0]
        # reinterpret_cast[B*](a)
        # ref_buffer.get().CopyFrom(
        #     reinterpret_cast<INT16_T_PTR>(), self.config)

        #$out_array = np.asarray(<cnp.float_t> rec_buffer)
        #$return 

    def __call__(self, cnp.ndarray[DTYPE_t, ndim=1] ref, cnp.ndarray[DTYPE_t, ndim=1] rec):
        print("__call__method executed")


    def __dealloc__(self):
        print("__dealloc__method executed")
        del self.hp_filter
