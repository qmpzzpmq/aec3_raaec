# distutils: language=c++

import numpy as np
cimport numpy as cnp

from webrtc_aec3 cimport HighPassFilter as webrtc_HighPassFilter
from webrtc_aec3 cimport EchoControl as webrtc_EchoControl
from webrtc_aec3 cimport AudioBuffer as webrtc_AudioBuffer
from webrtc_aec3 cimport EchoCanceller3Factory as webrtc_EchoCanceller3Factory
from webrtc_aec3 cimport EchoCanceller3 as webrtc_EchoCanceller3
from webrtc_aec3 cimport EchoCanceller3Config as webrtc_EchoCanceller3Config
from webrtc_aec3 cimport StreamConfig as webrtc_StreamConfig

ctypedef cnp.int16_t CNP_DTYPE_t
ctypedef int16_t DTYPE_t

cdef vector[DTYPE_t] arrayToVector(cnp.ndarray[CNP_DTYPE_t, ndim=1] array):
    cdef long size = array.size
    cdef vector[DTYPE_t] vec
    cdef long i
    for i in range(size):
        vec.push_back(array[i])
    return vec

cdef class AEC3:
    cdef:
        unique_ptr[webrtc_EchoControl] echo_controler
        webrtc_HighPassFilter *hp_filter
        webrtc_StreamConfig *config
        readonly int fs
        readonly int num_ref_channel
        readonly int num_rec_channel
        readonly int bytes_per_frame
        readonly int bits_per_sample

    def __cinit__(
            self, int num_ref_channel=1, int num_rec_channel=1,
            int fs=16000, int bits_per_sample=16):
        self.num_ref_channel = num_ref_channel
        self.num_rec_channel = num_rec_channel
        self.fs = fs
        self.bits_per_sample = bits_per_sample
        cdef int samples_per_frame = int(fs / 100)
        self.bytes_per_frame = int(samples_per_frame * bits_per_sample / 8)
        self.hp_filter = new webrtc_HighPassFilter(fs, num_rec_channel)
        cdef webrtc_EchoCanceller3Config aec3_config
        aec3_config.filter.export_linear_aec_output = True
        self.config = new webrtc_StreamConfig(fs, num_rec_channel, False)
        cdef webrtc_EchoCanceller3Factory *aec3_factory = new webrtc_EchoCanceller3Factory(
            aec3_config)
        self.echo_controler = aec3_factory[0].Create(
            fs, num_ref_channel, num_rec_channel)
        print("__cinit__ method executed")

    cpdef process_chunk(
            self,
            cnp.ndarray[CNP_DTYPE_t, ndim=1, mode = 'c'] ref,
            cnp.ndarray[CNP_DTYPE_t, ndim=1, mode = 'c'] rec,
            cnp.ndarray[CNP_DTYPE_t, ndim=1, mode = 'c'] linear,
            cnp.ndarray[CNP_DTYPE_t, ndim=1, mode = 'c'] out,
        ):
        cdef webrtc_AudioBuffer *ref_buffer = new webrtc_AudioBuffer(
            self.fs, self.num_ref_channel,
            self.fs, self.num_ref_channel,
            self.fs, self.num_ref_channel,
        )
        cdef webrtc_AudioBuffer *rec_buffer = new webrtc_AudioBuffer(
            self.fs, self.num_rec_channel,
            self.fs, self.num_rec_channel,
            self.fs, self.num_rec_channel,
        )
        cdef webrtc_AudioBuffer *linear_buffer = new webrtc_AudioBuffer(
            self.fs, self.num_rec_channel,
            self.fs, self.num_rec_channel,
            self.fs, self.num_rec_channel,
        )

        cdef DTYPE_t* ref_ptr = &ref[0]
        cdef DTYPE_t* rec_ptr = &rec[0]
        ref_buffer[0].CopyFrom(ref_ptr, self.config[0])
        rec_buffer[0].CopyFrom(rec_ptr, self.config[0])

        self.echo_controler.get().AnalyzeRender(ref_buffer)
        self.echo_controler.get().AnalyzeCapture(rec_buffer)
        self.hp_filter[0].Process(rec_buffer, True);
        self.echo_controler.get().SetAudioBufferDelay(0)
        self.echo_controler.get().ProcessCapture(rec_buffer, linear_buffer, False)

        cdef DTYPE_t* linear_ptr = &linear[0]
        linear_buffer[0].CopyTo(self.config[0], linear_ptr)

        cdef DTYPE_t* out_ptr = &out[0]
        rec_buffer[0].CopyTo(self.config[0], out_ptr)

        free(ref_ptr)
        free(rec_ptr)
        # free(linear_ptr)
        # free(out_ptr)
        print("free done")
        del ref_buffer
        del rec_buffer
        del linear_buffer
        print("del done")
        print("process_chunk method executed")

    def run(self, cnp.ndarray[CNP_DTYPE_t, ndim=1] ref, cnp.ndarray[CNP_DTYPE_t, ndim=1] rec):
        out = np.zeros_like(rec)
        linear = np.zeros_like(rec)
        self.process_chunk(ref, rec, linear, out)
        print("__call__ method executed")


    def __dealloc__(self):
        if self.hp_filter != NULL:
            del self.hp_filter
        if self.config != NULL:
            del self.config
        print("__dealloc__ method executed")