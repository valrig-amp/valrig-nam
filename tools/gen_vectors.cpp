// Reference-vector generator for valverig-nam: runs a .nam model through the C++
// reference over a deterministic signal and a block schedule, and writes the
// input, the output and a JSON sidecar. Raw little-endian f64 blobs, so the
// Rust test suite can compare bit for bit.
//
//   gen_vectors <model.nam> <out_prefix> <num_samples> <seed> <schedule> [--no-prewarm] [--slim V]
//   gen_vectors --spread <a.out.f64> <b.out.f64> ...
//
// The input is uniform in [-0.5, 0.5); every value has a 24-bit mantissa so it
// survives the double/float boundary unchanged.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <iostream>
#include <string>
#include <vector>

#include "NAM/dsp.h"
#include "NAM/get_dsp.h"
#include "NAM/slimmable.h"

// Deterministic PRNG (splitmix64) -> uniform in [-1, 1) as *exactly representable* f32,
// so the input signal is identical in C++ and Rust with zero parsing ambiguity.
static uint64_t sm_state;
static uint64_t sm_next()
{
  uint64_t z = (sm_state += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}
static float sm_uniform()
{
  // 24-bit mantissa -> exactly representable in f32
  const uint32_t bits = (uint32_t)(sm_next() >> 40); // 24 bits
  const float u = (float)bits / 8388608.0f;          // [0, 2)
  return u - 1.0f;                                   // [-1, 1)
}

static void write_f64(std::ofstream& o, const std::vector<double>& v)
{
  o.write(reinterpret_cast<const char*>(v.data()), (std::streamsize)(v.size() * sizeof(double)));
}

// --spread <file0.f64> <file1.f64> ...
// Prints the largest disagreement among the given outputs, relative to the
// peak of the first. Used to measure how far the reference's own supported
// build configurations drift from each other.
static int spread_mode(int argc, char** argv)
{
  std::vector<std::vector<double>> sets;
  for (int i = 2; i < argc; i++)
  {
    std::ifstream f(argv[i], std::ios::binary);
    if (!f) { std::cerr << "cannot open " << argv[i] << "\n"; return 1; }
    std::vector<double> v;
    double d;
    while (f.read(reinterpret_cast<char*>(&d), sizeof(double))) v.push_back(d);
    sets.push_back(std::move(v));
  }
  if (sets.size() < 2) { std::cerr << "need at least two files\n"; return 1; }
  double peak = 0.0;
  for (double x : sets[0]) peak = std::max(peak, std::abs(x));
  if (peak == 0.0) peak = 1.0;
  double worst = 0.0;
  for (size_t a = 0; a < sets.size(); a++)
    for (size_t b = a + 1; b < sets.size(); b++)
    {
      const size_t n = std::min(sets[a].size(), sets[b].size());
      for (size_t i = 0; i < n; i++) worst = std::max(worst, std::abs(sets[a][i] - sets[b][i]));
    }
  std::printf("%.6e\n", worst / peak);
  return 0;
}

int main(int argc, char** argv)
{
  if (argc >= 2 && std::string(argv[1]) == "--spread") return spread_mode(argc, argv);
  if (argc < 6)
  {
    std::cerr << "usage: gen_vectors <model.nam> <out_prefix> <num_samples> <seed> <schedule>"
                 " [--no-prewarm] [--slim V]\n"
                 "  schedule: comma-separated block sizes, cycled (e.g. \"64\" or \"1,7,64,3\")\n";
    return 1;
  }
  const std::string modelPath = argv[1];
  const std::string prefix = argv[2];
  const int numSamples = std::atoi(argv[3]);
  sm_state = (uint64_t)std::strtoull(argv[4], nullptr, 10);

  std::vector<int> schedule;
  {
    std::string s(argv[5]);
    size_t pos = 0;
    while (pos <= s.size())
    {
      size_t c = s.find(',', pos);
      if (c == std::string::npos) c = s.size();
      if (c > pos) schedule.push_back(std::atoi(s.substr(pos, c - pos).c_str()));
      pos = c + 1;
    }
  }
  if (schedule.empty()) { std::cerr << "empty schedule\n"; return 1; }

  bool prewarm = true;
  bool hasSlim = false;
  double slimValue = 0.0;
  const float amp = 0.5f;
  for (int i = 6; i < argc; i++)
  {
    std::string a(argv[i]);
    if (a == "--no-prewarm") prewarm = false;
    else if (a == "--slim" && i + 1 < argc) { hasSlim = true; slimValue = std::atof(argv[++i]); }
    else { std::cerr << "unknown arg " << a << "\n"; return 1; }
  }

  int maxBlock = 0;
  for (int b : schedule) if (b > maxBlock) maxBlock = b;

  auto model = nam::get_dsp(std::filesystem::path(modelPath));
  if (!model) { std::cerr << "failed to load\n"; return 1; }
  model->SetPrewarmOnReset(prewarm);
  if (hasSlim)
  {
    auto* slim = dynamic_cast<nam::SlimmableModel*>(model.get());
    if (!slim) { std::cerr << "not slimmable\n"; return 1; }
    slim->SetSlimmableSize(slimValue);
  }

  const int inCh = model->NumInputChannels();
  const int outCh = model->NumOutputChannels();
  double sr = model->GetExpectedSampleRate();
  if (!(sr > 0)) sr = 48000.0;
  model->Reset(sr, maxBlock);

  // Deterministic input, generated once, shared with Rust via the .in.f64 blob.
  std::vector<std::vector<double>> inAll((size_t)inCh, std::vector<double>((size_t)numSamples));
  for (int s = 0; s < numSamples; s++)
    for (int c = 0; c < inCh; c++)
      inAll[(size_t)c][(size_t)s] = (double)(amp * sm_uniform());

  std::vector<std::vector<double>> inBuf((size_t)inCh, std::vector<double>((size_t)maxBlock, 0.0));
  std::vector<std::vector<double>> outBuf((size_t)outCh, std::vector<double>((size_t)maxBlock, 0.0));
  std::vector<double*> inPtr((size_t)inCh), outPtr((size_t)outCh);
  for (int c = 0; c < inCh; c++) inPtr[(size_t)c] = inBuf[(size_t)c].data();
  for (int c = 0; c < outCh; c++) outPtr[(size_t)c] = outBuf[(size_t)c].data();

  std::vector<std::vector<double>> outAll((size_t)outCh, std::vector<double>((size_t)numSamples, 0.0));

  int pos = 0, si = 0;
  while (pos < numSamples)
  {
    int n = schedule[(size_t)(si++ % schedule.size())];
    if (n > numSamples - pos) n = numSamples - pos;
    if (n <= 0) break;
    for (int c = 0; c < inCh; c++)
      for (int i = 0; i < n; i++) inBuf[(size_t)c][(size_t)i] = inAll[(size_t)c][(size_t)(pos + i)];
    model->process(inPtr.data(), outPtr.data(), n);
    for (int c = 0; c < outCh; c++)
      for (int i = 0; i < n; i++) outAll[(size_t)c][(size_t)(pos + i)] = outBuf[(size_t)c][(size_t)i];
    pos += n;
  }

  {
    std::ofstream fi(prefix + ".in.f64", std::ios::binary);
    for (int c = 0; c < inCh; c++) write_f64(fi, inAll[(size_t)c]);
  }
  {
    std::ofstream fo(prefix + ".out.f64", std::ios::binary);
    for (int c = 0; c < outCh; c++) write_f64(fo, outAll[(size_t)c]);
  }
  {
    std::ofstream fm(prefix + ".meta.json");
    fm << "{\"in_channels\":" << inCh << ",\"out_channels\":" << outCh << ",\"num_samples\":" << numSamples
       << ",\"sample_rate\":" << sr << ",\"prewarm\":" << (prewarm ? "true" : "false")
       << ",\"prewarm_samples\":" << model->GetPrewarmSamples() << ",\"max_block\":" << maxBlock
       << ",\"has_loudness\":" << (model->HasLoudness() ? "true" : "false");
    if (model->HasLoudness()) fm << ",\"loudness\":" << model->GetLoudness();
    fm << ",\"schedule\":[";
    for (size_t i = 0; i < schedule.size(); i++) fm << (i ? "," : "") << schedule[i];
    fm << "]}\n";
  }
  std::cerr << "wrote " << prefix << ".{in,out}.f64 (" << numSamples << " samples, "
            << inCh << "->" << outCh << " ch)\n";
  return 0;
}
