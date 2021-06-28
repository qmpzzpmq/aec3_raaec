# distutils: language=c++

from tqdm import tqdm
import wave
import numpy as np
cimport numpy as cnp
from cython cimport view

from indirect_buffer.buffer_impl cimport IndirectMemory2D

from webrtc_aec3 cimport HighPassFilter as webrtc_HighPassFilter
from webrtc_aec3 cimport EchoControl as webrtc_EchoControl
from webrtc_aec3 cimport AudioBuffer as webrtc_AudioBuffer
from webrtc_aec3 cimport EchoCanceller3Factory as webrtc_EchoCanceller3Factory
from webrtc_aec3 cimport EchoCanceller3 as webrtc_EchoCanceller3
from webrtc_aec3 cimport EchoCanceller3Config as webrtc_EchoCanceller3Config
from webrtc_aec3 cimport StreamConfig as webrtc_StreamConfig

cdef void check_float2d_ptr(float** arr, size_t I, size_t J):
    cdef size_t i, j
    for i in range(I):
        for j in range(J):
            print(f"{i},{j}: {arr[i][j]}")

cdef void check_float2d(float[:, :] arr):
    cdef size_t i, j , I, J
    cdef float *ptr
    I = arr.shape[0]
    J = arr.shape[1]
    for i in range(I):
        for j in range(J):
            ptr = &arr[i][j]
            print(f"{i},{j}: {arr[i][j]}")

cdef void check_float2d_(float[:, :] arr):
    cdef size_t i, j , I, J
    J = arr.shape[0]
    I = arr.shape[1]
    for i in range(I): #frame
        for j in range(J): #channel
            print(f"{j},{i}: {arr[j][i]}")
 
cdef void check_float1d(float[:, :] arr):
    cdef size_t i, I
    I = arr.shape[0]
    for i in range(I):
        print(f"{i}: {arr[i]}")

cdef void check_floatpointer(float * arr, size_t I):
    cdef size_t i
    for i in range(I):
        print(f"{i}: {arr[i]}")

cdef vector[int16_t] arrayToVector(cnp.ndarray[cnp.int16_t, ndim=1] array):
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
        webrtc_StreamConfig *config
        readonly int fs
        readonly int out_fs
        readonly int num_ref_channel
        readonly int num_rec_channel
        readonly int samples_per_frame
        readonly int bits_per_sample

    def __cinit__(
            self, int num_ref_channel=1, int num_rec_channel=1,
            int fs=16000, int bits_per_sample=16,
            out_fs=None, pure_linear=False):
        self.num_ref_channel = num_ref_channel
        self.num_rec_channel = num_rec_channel
        self.fs = fs
        if out_fs == None:
            self.out_fs = fs
        self.bits_per_sample = bits_per_sample
        self.samples_per_frame = int(fs / 100)
        self.hp_filter = new webrtc_HighPassFilter(fs, num_rec_channel)
        cdef webrtc_EchoCanceller3Config aec3_config
        aec3_config.filter.export_linear_aec_output = True
        aec3_config.echo_model.model_reverb_in_nonlinear_mode = pure_linear
        self.config = new webrtc_StreamConfig(fs, num_rec_channel, False)
        cdef webrtc_EchoCanceller3Factory *aec3_factory = new webrtc_EchoCanceller3Factory(
            aec3_config)
        self.echo_controler = aec3_factory[0].Create(
            fs, num_ref_channel, num_rec_channel)

    cdef process_chunk(
            self,
            webrtc_AudioBuffer *ref_buffer,
            webrtc_AudioBuffer *rec_buffer,
            webrtc_AudioBuffer *linear_buffer,
        ):
        self.echo_controler.get().AnalyzeRender(ref_buffer)
        self.echo_controler.get().AnalyzeCapture(rec_buffer)
        self.hp_filter[0].Process(rec_buffer, True);
        self.echo_controler.get().SetAudioBufferDelay(0)
        self.echo_controler.get().ProcessCapture(rec_buffer, linear_buffer, False)

    def test_int16(
            self, cnp.ndarray[cnp.int16_t, ndim=1] ref,
            cnp.ndarray[cnp.int16_t, ndim=1] rec,
            str linear_path = "",
            str out_path = "",
            int chunk = -1,
        ):
        cdef cnp.ndarray[cnp.int16_t, ndim=1, mode = 'c'] linear = np.zeros_like(rec)
        cdef cnp.ndarray[cnp.int16_t, ndim=1, mode = 'c'] out = np.zeros_like(rec)
        cdef int start
        cdef int end
        cdef int16_t* ref_ptr = &ref[0]
        cdef int16_t* rec_ptr = &rec[0]
        cdef int16_t* linear_ptr = &linear[0]
        cdef int16_t* out_ptr = &out[0]

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
        print(f"ref buffer num_frames {ref_buffer[0].num_frames()}")
        print(f"rec buffer num_frames {rec_buffer[0].num_frames()}")
        print(f"linear buffer num_frames {linear_buffer[0].num_frames()}")
        for current in tqdm(range(total)):
            start = current * self.samples_per_frame
            end = start + self.samples_per_frame

            ref_buffer[0].CopyFrom(ref_ptr + start, self.config[0])
            rec_buffer[0].CopyFrom(rec_ptr + start, self.config[0])
            self.process_chunk(ref_buffer, rec_buffer, linear_buffer)

            linear_buffer[0].CopyTo(self.config[0], linear_ptr + start)
            rec_buffer[0].CopyTo(self.config[0], out_ptr + start)
            if linear_path is not "":
                linear_wav.writeframes(linear[start:end])
            if out_path is not "":
                out_wav.writeframes(out[start:end])

        del ref_buffer
        del rec_buffer
        del linear_buffer
        if linear_path is not "":
            linear_wav.close()
        if out_path is not "":
            out_wav.close()
        return linear, out

    def run_int16(
            self,
            cnp.ndarray[cnp.int16_t, ndim=1] ref,
            cnp.ndarray[cnp.int16_t, ndim=1] rec,
        ):
        cdef cnp.ndarray[cnp.int16_t, ndim=1, mode = 'c'] linear = np.zeros_like(rec)
        cdef cnp.ndarray[cnp.int16_t, ndim=1, mode = 'c'] out = np.zeros_like(rec)
        cdef int start
        cdef int end
        cdef int16_t* ref_ptr = &ref[0]
        cdef int16_t* rec_ptr = &rec[0]
        cdef int16_t* linear_ptr = &linear[0]
        cdef int16_t* out_ptr = &out[0]

        cdef int total = int(len(rec) / self.samples_per_frame) \
            if len(rec) < len(ref) else int(len(ref) / self.samples_per_frame)

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
        for current in range(total):
            start = current * self.samples_per_frame
            end = start + self.samples_per_frame

            ref_buffer[0].CopyFrom(ref_ptr + start, self.config[0])
            rec_buffer[0].CopyFrom(rec_ptr + start, self.config[0])
            self.process_chunk(ref_buffer, rec_buffer, linear_buffer)

            linear_buffer[0].CopyTo(self.config[0], linear_ptr + start)
            rec_buffer[0].CopyTo(self.config[0], out_ptr + start)

        del ref_buffer
        del rec_buffer
        del linear_buffer
        return linear, out

    cdef _run_float(
            self, 
            cnp.ndarray[cnp.float32_t, ndim=1, mode = 'c'] ref,
            cnp.ndarray[cnp.float32_t, ndim=1, mode = 'c'] rec,
        ):
        cdef cnp.ndarray[cnp.float32_t, ndim=2, mode = 'c'] ref_2d = ref.reshape([1,-1])
        cdef cnp.ndarray[cnp.float32_t, ndim=2, mode = 'c'] rec_2d = rec.reshape([1,-1])
        cdef cnp.ndarray[cnp.float32_t, ndim=2, mode = 'c'] linear = np.zeros_like(rec_2d, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=2, mode = 'c'] out = np.zeros_like(rec_2d, dtype=np.float32)
        cdef int start
        cdef int end

        cdef float [:, ::view.contiguous] ref_view = ref_2d
        cdef float [:, ::view.contiguous] rec_view = rec_2d
        cdef float [:, ::view.contiguous] linear_view = linear
        cdef float [:, ::view.contiguous] out_view = out

        cdef float * ref_ptr
        cdef float ** ref_ptr_ptr
        cdef float * rec_ptr
        cdef float ** rec_ptr_ptr
        cdef float * linear_ptr
        cdef float ** linear_ptr_ptr
        cdef float * out_ptr
        cdef float ** out_ptr_ptr

        cdef int total = int(len(rec) / self.samples_per_frame) \
            if len(rec) < len(ref) else int(len(ref) / self.samples_per_frame)

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
        for current in tqdm(range(total)):
            start = current * self.samples_per_frame
            end = start + self.samples_per_frame

            # print('1')
            ref_ptr = <float *>&ref_view[0, start]
            ref_ptr_ptr = <float **> &ref_ptr
            ref_buffer[0].CopyFrom(ref_ptr_ptr, self.config[0])

            # print('2')
            rec_ptr = <float *>&rec_view[0, start]
            rec_ptr_ptr = <float **> &rec_ptr
            rec_buffer[0].CopyFrom(rec_ptr_ptr, self.config[0])

            # print('3')
            self.process_chunk(ref_buffer, rec_buffer, linear_buffer)

            # print('4')
            linear_ptr = <float *>&linear_view[0, start]
            linear_ptr_ptr = <float **> &linear_ptr
            linear_buffer[0].CopyTo(self.config[0], linear_ptr_ptr)

            out_ptr = <float *>&out_view[0, start]
            out_ptr_ptr = <float **> &out_ptr
            rec_buffer[0].CopyTo(self.config[0], out_ptr_ptr)

        del ref_buffer
        del rec_buffer
        del linear_buffer
        return linear, out

    def run_float(self, 
        ref,
        rec,
    ):
        return self._run_float(ref,rec)

    cdef buffer_test_(
            self,
            #cnp.ndarray[cnp.float32_t, ndim=2, mode = 'c'] a,
            a,
        ):
        cdef webrtc_AudioBuffer *test_buffer = new webrtc_AudioBuffer(
            160, 1, 160, 1, 160,
        )
        print(f"input_num_frames_: {test_buffer[0].num_frames()}")
        print(f"a shape: {a.shape[0]} {a.shape[1]}")
        cdef float [:, ::view.contiguous] a_view = a
        cdef float * a_ptr = <float *>&a_view[0, 3]
        cdef float ** a_ptr_ptr = <float **> &a_ptr
        
        test_buffer[0].CopyFrom(a_ptr_ptr, self.config[0])
        # test_buffer[0].CopyFrom(<float **>a_ptr, self.config[0])
        return
        
    def buffer_test(self, a):
        print(f"a shape: {a.shape[0]} {a.shape[1]}")
        return self.buffer_test_(a)

    def __dealloc__(self):
        if self.hp_filter != NULL:
            del self.hp_filter
        if self.config != NULL:
            del self.config
