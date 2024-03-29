#include "api/audio/echo_canceller3_factory.h"
#include "api/audio/echo_canceller3_config.h"
#include "api/audio/audio_frame.h"
#include "modules/audio_processing/audio_buffer.h"
#include "modules/audio_processing/high_pass_filter.h"
#include "common_audio/wav_file.h"

#include "wavio/wavreader.h"
#include "wavio/wavwriter.h"

#include <iostream>

using namespace webrtc;
using namespace std;

void print_wav_information(const char* fn, int format, int channels, int sample_rate, int bits_per_sample, int length)
{
	cout << "=====================================" << endl
		<< fn << " information:" << endl
		<< "format: " << format << endl
		<< "channels: " << channels << endl
		<< "sample_rate: " << sample_rate << endl
		<< "bits_per_sample: " << bits_per_sample << endl
		<< "length: " << length << endl
		<< "total_samples: " << length / bits_per_sample * 8 << endl
		<< "======================================" << endl;
}

void print_progress(int current, int total)
{
	int percentage = current / static_cast<float>(total) * 100;
	static constexpr int p_bar_length = 50;
	int progress = percentage * p_bar_length / 100;
	cout << "        " << current << "/" << total << "    " << percentage << "%";
	cout << "|";
	for (int i = 0; i < progress; ++i)
		cout << "=";
	cout << ">";
	for (int i = progress; i < p_bar_length; ++i)
		cout << " ";
	cout << "|";
	cout <<"\r";
}

int main(int argc, char* argv[])
{
  if (argc != 4)
  {
    cerr << "usage: ./demo ref.wav rec.wav out.wav" << endl;
    return -1;
  }
  cout << "======================================" << endl
    << "ref file is: " << argv[1] << endl
    << "rec file is: " << argv[2] << endl
    << "out file is: " << argv[3] << endl
    << "======================================" << endl;
  
  ref_file = WavReader(argv[1])
  rec_file = WavReader(argv[2])
 
  size_t ref_num_samples = ref_file.num_samples();
  size_t ref_sample_rate = ref_file.sample_rate();
  size_t ref_channels = ref_file.num_channels();
  cout << "======================================" << endl
    "ref: " << endl
    "num_channel: " << ref_channels << endl
    "sample rate: " << ref_sample_rate << endl
    "samples: " << ref_samples << endl;

  size_t rec_num_samples = rec_file.num_samples();
  size_t rec_sample_rate = rec_file.sample_rate();
  size_t rec_channels = rec_file.num_channels();
  cout << "======================================" << endl
    "rec: " << endl
    "num_channel: " << rec_channels << endl
    "sample rate: " << rec_sample_rate << endl
    "samples: " << rec_samples << endl;
  
  EchoCanceller3Config aec_config;
  aec_config.filter.export_linear_aec_output = true;
  EchoCanceller3Factory aec_factory = EchoCanceller3Factory(aec_config);
  std::unique_ptr<EchoControl> echo_controler = aec_factory.Create(ref_sample_rate, ref_channels, rec_channels);
  std::unique_ptr<HighPassFilter> hp_filter = std::make_unique<HighPassFilter>(rec_sample_rate, rec_channels);
  
  StreamConfig config = StreamConfig(sample_rate, channels, false);
  
  std::unique_ptr<AudioBuffer> ref_audio = std::make_unique<AudioBuffer>(
  	config.sample_rate_hz(), config.num_channels(),
  	config.sample_rate_hz(), config.num_channels(),
  	config.sample_rate_hz(), config.num_channels());
  std::unique_ptr<AudioBuffer> aec_audio = std::make_unique<AudioBuffer>(
  	config.sample_rate_hz(), config.num_channels(),
  	config.sample_rate_hz(), config.num_channels(),
  	config.sample_rate_hz(), config.num_channels());
  constexpr int kLinearOutputRateHz = 16000;
  std::unique_ptr<AudioBuffer> aec_linear_audio = std::make_unique<AudioBuffer>(
  	kLinearOutputRateHz, config.num_channels(),
  	kLinearOutputRateHz, config.num_channels(),
  	kLinearOutputRateHz, config.num_channels());
  
  AudioFrame ref_frame, aec_frame;
  int a;
  
  void* h_out = wav_write_open(argv[3], rec_sample_rate, rec_bits_per_sample, rec_channels);
  void* h_linear_out = wav_write_open("linear.wav", kLinearOutputRateHz, rec_bits_per_sample, rec_channels);
  
  int samples_per_frame = sample_rate / 100;
  int bytes_per_frame = samples_per_frame * bits_per_sample / 8;
  int total = rec_samples < ref_samples ? rec_samples / samples_per_frame : rec_samples / samples_per_frame;
  
  int current = 0;
  unsigned char* ref_tmp = new unsigned char[bytes_per_frame];
  unsigned char* aec_tmp = new unsigned char[bytes_per_frame];
  cout << "processing audio frames ..." << endl;
  while (current++ < total) 
  {
  	print_progress(current, total);
  	wav_read_data(h_ref, ref_tmp, bytes_per_frame);
  	wav_read_data(h_rec, aec_tmp, bytes_per_frame);
  
  	ref_frame.UpdateFrame(0, reinterpret_cast<int16_t*>(ref_tmp), samples_per_frame, sample_rate, AudioFrame::kNormalSpeech, AudioFrame::kVadActive, 1);
  	aec_frame.UpdateFrame(0, reinterpret_cast<int16_t*>(aec_tmp), samples_per_frame, sample_rate, AudioFrame::kNormalSpeech, AudioFrame::kVadActive, 1);
  
  	ref_audio->CopyFrom(&ref_frame);
  	aec_audio->CopyFrom(&aec_frame);
  
  	ref_audio->SplitIntoFrequencyBands();
  	echo_controler->AnalyzeRender(ref_audio.get());
  	ref_audio->MergeFrequencyBands();
  	echo_controler->AnalyzeCapture(aec_audio.get());
  	aec_audio->SplitIntoFrequencyBands();
  	hp_filter->Process(aec_audio.get(), true);
  	echo_controler->SetAudioBufferDelay(0);
  	echo_controler->ProcessCapture(aec_audio.get(), aec_linear_audio.get(), false);
  	aec_audio->MergeFrequencyBands();
  
  	aec_audio->CopyTo(&aec_frame);
  	memcpy(aec_tmp, aec_frame.data(), bytes_per_frame);
  	wav_write_data(h_out, aec_tmp, bytes_per_frame);
  
  	aec_frame.UpdateFrame(0, nullptr, kLinearOutputRateHz / 100, kLinearOutputRateHz, AudioFrame::kNormalSpeech, AudioFrame::kVadActive, 1);
  	aec_linear_audio->CopyTo(&aec_frame);
  	memcpy(aec_tmp, aec_frame.data(), 320);
  	wav_write_data(h_linear_out, aec_tmp, 320);
  }
  
  delete[] ref_tmp;
  delete[] aec_tmp;
  
  wav_read_close(h_ref);
  wav_read_close(h_rec);
  wav_write_close(h_out);
  wav_write_close(h_linear_out);
  
  return 0;
}
