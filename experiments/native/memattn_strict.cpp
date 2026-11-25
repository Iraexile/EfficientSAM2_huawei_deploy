#include <cstdio>
#include <fstream>
#include <gpu.h>
#include <net.h>
int PreflightMemattnStrict(const char *base) {
  ncnn::create_gpu_instance();
  if (ncnn::get_gpu_count() < 1) {
    printf("K01 no_gpu\n");
    return 2;
  }
  ncnn::Net net;
  net.opt.use_vulkan_compute = true;
  net.opt.use_packing_layout = false;
  net.opt.use_fp16_packed = net.opt.use_fp16_storage =
      net.opt.use_fp16_arithmetic = net.opt.use_fp16_uniform = false;
  net.set_vulkan_device(0);
  std::string path(base);
  int rc = net.load_param((path + "/k01.param").c_str());
  if (rc) {
    printf("K01 load_param=%d\n", rc);
    return 3;
  }
  int unsupported = 0;
  for (size_t i = 0; i < net.layers().size(); i++) {
    auto *l = net.layers()[i];
    bool metadata = l->type == "Input" || l->type == "Split";
    printf("K01_MAP index=%zu type=%s name=%s vulkan=%d metadata=%d\n", i,
           l->type.c_str(), l->name.c_str(), int(l->support_vulkan),
           int(metadata));
    if (!metadata && !l->support_vulkan)
      unsupported++;
  }
  printf("K01_PREFLIGHT layers=%zu unsupported=%d inference=NOT_RUN "
         "cpu_fallback=FORBIDDEN\n",
         net.layers().size(), unsupported);
  return unsupported ? 4 : 0;
}

#include "k01_rope_layout.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <command.h>
#include <map>
#include <sstream>
#include <stdexcept>
using KClock = std::chrono::steady_clock;
static double KMs(KClock::time_point t) {
  return std::chrono::duration<double, std::milli>(KClock::now() - t).count();
}
class KAlloc : public ncnn::VkBlobAllocator {
public:
  KAlloc(const ncnn::VulkanDevice *d) : VkBlobAllocator(d) {}
  std::map<ncnn::VkBufferMemory *, size_t> sizes;
  size_t live = 0, peak = 0;
  ncnn::VkBufferMemory *fastMalloc(size_t n) override {
    auto *p = VkBlobAllocator::fastMalloc(n);
    if (p) {
      sizes[p] = n;
      live += n;
      peak = std::max(peak, live);
    }
    return p;
  }
  void fastFree(ncnn::VkBufferMemory *p) override {
    auto it = sizes.find(p);
    if (it != sizes.end()) {
      live -= it->second;
      sizes.erase(it);
    }
    VkBlobAllocator::fastFree(p);
  }
};
struct KRef {
  size_t n;
  std::string pc, onnx;
};
static bool KCompare(const std::vector<float> &a, const std::string &path,
                     const std::string &label) {
  std::ifstream f(path, std::ios::binary);
  std::vector<float> b(a.size());
  if (!f.read(reinterpret_cast<char *>(b.data()), b.size() * 4))
    throw std::runtime_error("ref_read " + path);
  double mx = 0, ss = 0;
  size_t bad = 0, nf = 0;
  for (size_t i = 0; i < a.size(); i++) {
    if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
      nf++;
      continue;
    }
    double d = std::abs(double(a[i]) - b[i]);
    mx = std::max(mx, d);
    ss += d * d;
    if (d > 1e-4 + 1e-4 * std::abs(double(b[i])))
      bad++;
  }
  printf("K01_COMPARE %s maxabs=%.9g rmse=%.9g violations=%zu total=%zu "
         "nonfinite=%zu\n",
         label.c_str(), mx, std::sqrt(ss / a.size()), bad, a.size(), nf);
  return !bad && !nf;
}
int RunMemattnStrict(const char *directory) {
  std::string base(directory);
  int result = 0;
  ncnn::create_gpu_instance();
  if (ncnn::get_gpu_count() < 1)
    return 2;
  auto *dev = ncnn::get_gpu_device(0);
  auto &gpu = ncnn::get_gpu_info(0);
  printf("K01_BASE=%s\n",base.c_str());
  printf("K01_DEVICE name=%s driver=%u api=%u\n", gpu.device_name(),
         gpu.driver_version(), gpu.api_version());
  KAlloc alloc(dev);
  ncnn::VkStagingAllocator staging(dev);
  try {
    auto load = KClock::now();
    ncnn::Net net;
    net.opt.use_vulkan_compute = true;
    net.opt.use_packing_layout = false;
    net.opt.use_fp16_packed = net.opt.use_fp16_storage =
        net.opt.use_fp16_arithmetic = net.opt.use_fp16_uniform = false;
    net.opt.blob_vkallocator = &alloc;
    net.opt.workspace_vkallocator = &alloc;
    net.opt.staging_vkallocator = &staging;
    net.set_vulkan_device(0);
    if (net.load_param((base + "/k01.param").c_str()))
      throw std::runtime_error("load_param");
    for (size_t i = 0; i < net.layers().size(); i++) {
      auto *l = net.layers()[i];
      bool meta = l->type == "Input" || l->type == "Split";
      printf("K01_MAP index=%zu type=%s name=%s vulkan=%d metadata=%d\n", i,
             l->type.c_str(), l->name.c_str(), int(l->support_vulkan),
             int(meta));
      if (!meta && !l->support_vulkan)
        throw std::runtime_error("GPU_UNSUPPORTED " + l->name);
    }
    if (net.load_model((base + "/k01.bin").c_str()))
      throw std::runtime_error("load_model");
    printf("K01_LOAD ms=%.6f\n", KMs(load));
    K01RoPELayout rope(dev, net.opt);
    std::map<std::string, KRef> refs;
    std::ifstream mf(base + "/refs-full.txt");
    if (!mf) mf.open(base + "/refs.txt");
    std::string key, p, o;
    size_t n;
    while (mf >> key >> n >> p >> o)
      refs[key] = {n, p, o};
    if (refs.size() < 22)
      throw std::runtime_error("refs_count");
    std::map<std::string, std::vector<int>> shapes;
    std::ifstream shapeFile(base + "/shapes.txt");
    std::string sn;
    int sd, sw, sh, sz, sc;
    while (shapeFile >> sn >> sd >> sw >> sh >> sz >> sc)
      shapes[sn] = {sd, sw, sh, sz, sc};
    std::vector<ncnn::VkMat> blobs(net.blobs().size());
    std::vector<int> uses(blobs.size());
    for (auto *l : net.layers())
      for (int b : l->bottoms)
        uses[b]++;
    const char *names[] = {"curr", "curr_pos", "memory", "memory_pos"};
    std::vector<ncnn::Mat> host;
    auto up = KClock::now();
    ncnn::VkCompute upload(dev);
    size_t uploaded = 0;
    for (int i = 0; i < 4; i++) {
      int w = i < 2 ? 256 : 64, c = i < 2 ? 4096 : 4100;
      ncnn::Mat m(w, 1, c);
      std::ifstream f(base + "/in_" + names[i] + ".bin", std::ios::binary);
      for (int j = 0; j < c; j++)
        if (!f.read(static_cast<char *>(m.channel(j).data), w * 4))
          throw std::runtime_error("input_read");
      host.push_back(m);
      uploaded += size_t(w) * c * 4;
      for (size_t j = 0; j < net.blobs().size(); j++)
        if (net.blobs()[j].name == "in" + std::to_string(i))
          upload.record_upload(host.back(), blobs[j], net.opt);
    }
    std::vector<ncnn::VkMat> uploadedPacked = blobs;
    for (auto &b : blobs)
      if (!b.empty() && b.elempack != 1) {
        ncnn::VkMat plain;
        dev->convert_packing(b, plain, 1, 1, upload, net.opt);
        b = plain;
      }
    if (upload.submit_and_wait())
      throw std::runtime_error("upload_submit");
    printf("K01_UPLOAD ms=%.6f bytes=%zu includes_file_read=1\n", KMs(up),
           uploaded);
    host.clear();
    auto graph = KClock::now();
    double encTotal = 0, waitTotal = 0, readTotal = 0, compareTotal = 0;
    size_t checkpoints = 0;
    for (size_t i = 0; i < net.layers().size(); i++) {
      auto *l = net.layers()[i];
      if (l->type == "Input")
        continue;
      auto start = KClock::now();
      double enc = 0, wait = 0;
      if (l->type == "Split") {
        for (int t : l->tops)
          blobs[t] = blobs[l->bottoms[0]];
      } else if (l->type == "Reshape") {
        auto it = shapes.find(l->name);
        if (it == shapes.end())
          throw std::runtime_error("missing_static_shape");
        auto shape = it->second;
        auto m = blobs[l->bottoms[0]];
        size_t oldplane = size_t(m.w) * m.h * m.d,
               newplane = size_t(shape[1]) * shape[2] * shape[3];
        if (m.elempack != 1 || m.elemsize != 4 || oldplane != m.cstep ||
            oldplane * m.c != newplane * shape[4] || newplane % 4) {
          printf("K01_LAYOUT name=%s dims=%d w=%d h=%d d=%d c=%d cstep=%zu "
                 "elem=%zu pack=%d newplane=%zu newc=%d\n",
                 l->name.c_str(), m.dims, m.w, m.h, m.d, m.c, m.cstep,
                 m.elemsize, m.elempack, newplane, shape[4]);
          throw std::runtime_error("reshape_noncontiguous " + l->name);
        }
        m.dims = shape[0];
        m.w = shape[1];
        m.h = shape[2];
        m.d = shape[3];
        m.c = shape[4];
        m.cstep = newplane;
        blobs[l->tops[0]] = m;
      } else {
        ncnn::VkCompute cmd(dev);
        std::vector<ncnn::VkMat> ins, outs(l->tops.size());
        for (int b : l->bottoms) {
          if (blobs[b].empty())
            throw std::runtime_error("missing " + l->name);
          ins.push_back(blobs[b]);
        }
        int rc;
        if (l->type == "Slice" && l->name.rfind("unbind_", 0) == 0) {
          printf("K01_CUSTOM_SPLIT %s\n", l->name.c_str());
          rc = rope.split(ins[0], outs, cmd, net.opt);
        } else if (l->type == "Concat" && ins.size() == 2 && ins[0].dims == 4 &&
                   ins[0].w == 1) {
          printf("K01_CUSTOM_JOIN %s\n", l->name.c_str());
          rc = rope.join(ins, outs[0], cmd, net.opt);
        } else if (l->one_blob_only) {
          ncnn::VkMat empty;
          rc = l->forward(ins.empty() ? empty : ins[0], outs[0], cmd, net.opt);
        } else
          rc = l->forward(ins, outs, cmd, net.opt);
        if (rc)
          throw std::runtime_error("vk_forward " + l->name +
                                   " rc=" + std::to_string(rc));
        std::vector<ncnn::VkMat> packed_outputs = outs;
        for (size_t oi = 0; oi < outs.size(); oi++)
          if (outs[oi].elempack != 1) {
            printf("K01_GPU_UNPACK layer=%s pack=%d\n", l->name.c_str(),
                   outs[oi].elempack);
            ncnn::VkMat plain;
            dev->convert_packing(outs[oi], plain, 1, 1, cmd, net.opt);
            outs[oi] = plain;
          }
        enc = KMs(start);
        auto sub = KClock::now();
        rc = cmd.submit_and_wait();
        wait = KMs(sub);
        if (rc)
          throw std::runtime_error("vk_submit " + l->name +
                                   " rc=" + std::to_string(rc));
        for (size_t j = 0; j < outs.size(); j++) {
          if (outs[j].empty())
            throw std::runtime_error("empty " + l->name);
          blobs[l->tops[j]] = outs[j];
        }
      }
      encTotal += enc;
      waitTotal += wait;
      printf(
          "K01_OP index=%zu name=%s type=%s encode_ms=%.6f submit_wait_ms=%.6f "
          "live_requested=%zu peak_requested=%zu path=%s\n",
          i, l->name.c_str(), l->type.c_str(), enc, wait, alloc.live,
          alloc.peak,
          (l->type == "Split" || l->type == "Reshape")
              ? "GPU_BUFFER_METADATA_ALIAS"
              : "VULKAN_FORWARD");
      for (int top : l->tops) {
        const auto &blob = net.blobs()[top].name;
        auto r = refs.find(blob);
        if (r == refs.end())
          continue;
        ncnn::Mat m;
        auto t = KClock::now();
        ncnn::VkCompute read(dev);
        read.record_download(blobs[top], m, net.opt);
        if (read.submit_and_wait())
          throw std::runtime_error("read_submit " + blob);
        double readms = KMs(t);
        readTotal += readms;
        t = KClock::now();
        if (m.elempack != 1 || m.elemsize != 4)
          throw std::runtime_error("read_type");
        std::vector<float> a;
        size_t plane = size_t(m.w) * m.h * m.d;
        for (int c = 0; c < m.c; c++) {
          const float *v = m.channel(c);
          a.insert(a.end(), v, v + plane);
        }
        if (a.size() != r->second.n)
          throw std::runtime_error("read_shape " + blob);
        if (!KCompare(a, base + "/" + r->second.pc, l->name + "/PC_NCNN"))
          result = 10;
        if (r->second.pc != r->second.onnx && !KCompare(a, base + "/" + r->second.onnx,
                      l->name + (r->second.pc == r->second.onnx ? "/PC_AUX"
                                                                : "/ONNX")))
          result = 10;
        if (blob == "out0" &&
            !KCompare(a, base + "/golden_final.bin", "FINAL/PYTORCH_GOLDEN"))
          result = 10;
        std::ofstream f(base + "/gpu_" + blob + ".bin", std::ios::binary);
        f.write(reinterpret_cast<const char *>(a.data()), a.size() * 4);
        compareTotal += KMs(t);
        checkpoints++;
        printf("K01_READBACK blob=%s bytes=%zu ms=%.6f\n", blob.c_str(),
               a.size() * 4, readms);
      }
      for (int b : l->bottoms)
        if (--uses[b] == 0)
          blobs[b].release();
    }
    std::ifstream proc("/proc/self/status");
    std::string line;
    while (std::getline(proc, line))
      if (line.rfind("VmRSS:", 0) == 0 || line.rfind("VmHWM:", 0) == 0)
        printf("K01_PROCESS %s\n", line.c_str());
    printf("K01_SUMMARY layers=%zu checkpoints=%zu encode_ms=%.6f "
           "submit_wait_ms=%.6f readback_ms=%.6f compare_save_ms=%.6f "
           "diagnostic_wall_ms=%.6f peak_requested_bytes=%zu "
           "cpu_forward_calls=0 result=%d\n",
           net.layers().size(), checkpoints, encTotal, waitTotal, readTotal,
           compareTotal, KMs(graph), alloc.peak, result);
    if (checkpoints != refs.size())
      throw std::runtime_error("incomplete");
  } catch (const std::exception &e) {
    printf("K01_FAILURE reason=%s\n", e.what());
    result = 20;
  }
  printf("K01_EXIT=%d\n", result);
  return result;
}
