# distutils: language=c++

from tqdm import tqdm
import wave
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
        readonly int out_fs
        readonly int num_ref_channel
        readonly int num_rec_channel
        readonly int samples_per_frame
        readonly int bits_per_sample

    def __cinit__(
            self, int num_ref_channel=1, int num_rec_channel=1,
            int fs=16000, int bits_per_sample=16, out_fs=None, float len_frame=0.01):
        self.num_ref_channel = num_ref_channel
        self.num_rec_channel = num_rec_channel
        self.fs = fs
        if out_fs == None:
            self.out_fs = fs
        self.bits_per_sample = bits_per_sample
        self.samples_per_frame = int(fs * len_frame)
        self.hp_filter = new webrtc_HighPassFilter(fs, num_rec_channel)
        cdef webrtc_EchoCanceller3Config aec3_config
        aec3_config.filter.export_linear_aec_output = True
        self.config = new webrtc_StreamConfig(fs, num_rec_channel, False)
        cdef webrtc_EchoCanceller3Factory *aec3_factory = new webrtc_EchoCanceller3Factory(
            aec3_config)
        self.echo_controler = aec3_factory[0].Create(
            fs, num_ref_channel, num_rec_channel)
        print("__cinit__ method executed")

    cdef process_chunk(
            self,
            webrtc_AudioBuffer *ref_buffer,
            webrtc_AudioBuffer *rec_buffer,
            webrtc_AudioBuffer *linear_buffer,
            webrtc_AudioBuffer *out_buffer,
        ):
        self.echo_controler.get().AnalyzeRender(ref_buffer)
        self.echo_controler.get().AnalyzeCapture(rec_buffer)
        self.hp_filter[0].Process(rec_buffer, True);
        self.echo_controler.get().SetAudioBufferDelay(0)
        self.echo_controler.get().ProcessCapture(rec_buffer, linear_buffer, False)

    def linear_run(
            self, cnp.ndarray[CNP_DTYPE_t, ndim=1] ref,
            cnp.ndarray[CNP_DTYPE_t, ndim=1] rec,
            str linear_path = "",
            str out_path = "",
            int chunk = -1,
        ):
        cdef cnp.ndarray[CNP_DTYPE_t, ndim=1, mode = 'c'] linear = np.zeros_like(rec)
        cdef cnp.ndarray[CNP_DTYPE_t, ndim=1, mode = 'c'] out = np.zeros_like(rec)
        cdef int start
        cdef int end
        cdef DTYPE_t* ref_ptr = &ref[0]
        cdef DTYPE_t* rec_ptr = &rec[0]
        cdef DTYPE_t* linear_ptr = &linear[0]
        cdef DTYPE_t* out_ptr = &out[0]

        cdef int total = int(len(rec) / self.samples_per_frame) \
            if len(rec) < len(ref) else int(len(ref) / self.samples_per_frame)
        if chunk > 0:
            print(f"force total to {chunk}")
            total = chunk
        if linear_path is not "":
            linear_wav = wave.open(linear_path, mode="wb")
            linear_wav.setnchannels(self.num_rec_channel)
            linear_wav.setsampwidth(2)
            linear_wav.setframerate(self.fs)
            linear_wav.setnframes(min(len(rec), len(ref)))
        if out_path is not "":
            out_wav = wave.open(out_path
            , mode="wb")
            out_wav.setnchannels(self.num_rec_channel)
            out_wav.setsampwidth(2)
            out_wav.setframerate(self.fs)
            out_wav.setnframes(min(len(rec), len(ref)))

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
        cdef webrtc_AudioBuffer *out_buffer = new webrtc_AudioBuffer(
            self.fs, self.num_rec_channel,
            self.fs, self.num_rec_channel,
            self.fs, self.num_rec_channel,
        )
        # print(f"ref buffer num_frames {ref_buffer[0].num_frames()}")
        # print(f"rec buffer num_frames {rec_buffer[0].num_frames()}")
        # print(f"linear buffer num_frames {linear_buffer[0].num_frames()}")
        # print(f"out buffer num_frames {out_buffer[0].num_frames()}")
        for current in tqdm(range(total)):
            start = current * self.samples_per_frame
            end = start + self.samples_per_frame

            ref_buffer[0].CopyFrom(ref_ptr + start, self.config[0])
            rec_buffer[0].CopyFrom(rec_ptr + start, self.config[0])
            self.process_chunk(ref_buffer, rec_buffer, linear_buffer, out_buffer)

            linear_buffer[0].CopyTo(self.config[0], linear_ptr + start)
            out_buffer[0].CopyTo(self.config[0], out_ptr + start)
            if linear_path is not "":
                linear_wav.writeframes(linear[start:end])
            if out_path is not "":
                out_wav.writeframes(out[start:end])

        del ref_buffer
        del rec_buffer
        del linear_buffer
        del out_buffer
        if linear_path is not "":
            linear_wav.close()
        if out_path is not "":
            out_wav.close()
        print("__call__ method executed")
        return linear, out

    def __dealloc__(self):
        if self.hp_filter != NULL:
            del self.hp_filter
        if self.config != NULL:
            del self.config
        print("__dealloc__ method executed")