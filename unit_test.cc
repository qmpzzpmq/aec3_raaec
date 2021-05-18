#include "api/audio/echo_canceller3_factory.h"
#include "api/audio/echo_canceller3_config.h"
#include "modules/audio_processing/audio_buffer.h"
#include "modules/audio_processing/high_pass_filter.h"

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
  if (argc != 2)
  {
    cerr << "unit_tset: ./main <input.wav>\n";
    return -1;
  }
  int ref_format, ref_channels, ref_sample_rate, ref_bits_per_sample;
  unsigned int ref_data_length;

  cout << "======================================" << endl
    << "In file is: " << argv[1] << endl;
  
  void* h_ref = wav_read_open(argv[1]);
  int res = wav_get_header(h_ref, &ref_format, &ref_channels, &ref_sample_rate, &ref_bits_per_sample, &ref_data_length);
  if (!res)
  {
    cerr << "get ref header error: " << res << endl;
    return -1;
  }  
  int ref_samples = ref_data_length * 8 / ref_bits_per_sample;
 
  print_wav_information(argv[1], ref_format, ref_channels, ref_sample_rate, ref_bits_per_sample, ref_data_length);
  int current = 0;
  int samples_per_frame = ref_sample_rate / 100;
  int total = ref_samples / (samples_per_frame);
  int bytes_per_frame = samples_per_frame * ref_bits_per_sample / 8;
  unsigned char* ref_tmp = new unsigned char[bytes_per_frame];
  while (current++ < total) 
  {
    cout << "Processing: " << current << "/" << total << endl;
    wav_read_data(h_ref, ref_tmp, bytes_per_frame);
  }

}
