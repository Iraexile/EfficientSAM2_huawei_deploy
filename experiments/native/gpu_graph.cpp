#include <net.h>
#include <gpu.h>
#include <fstream>
#include <cstdio>
#include "suite_rope_layout.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <command.h>
#include <map>
#include <sstream>
#include <stdexcept>
using SuiteClock = std::chrono::steady_clock;
static double SuiteMs(SuiteClock::time_point t) {
  return std::chrono::duration<double, std::milli>(SuiteClock::now() - t).count();
}
class SuiteAlloc : public ncnn::VkBlobAllocator {
public:
  SuiteAlloc(const ncnn::VulkanDevice *d) : VkBlobAllocator(d) {}
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
struct SuiteRef {
  size_t n;
  std::string pc, onnx;
};
static bool SuiteCompare(const std::vector<float> &a, const std::string &path,
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
  printf("GRAPH_COMPARE %s maxabs=%.9g rmse=%.9g violations=%zu total=%zu "
         "nonfinite=%zu\n",
         label.c_str(), mx, std::sqrt(ss / a.size()), bad, a.size(), nf);
  return !bad && !nf;
}
int RunGpuGraph(const char *directory) {
  std::string base(directory);
  int result = 0;
  ncnn::create_gpu_instance();
  if (ncnn::get_gpu_count() < 1)
    return 2;
  auto *dev = ncnn::get_gpu_device(0);
  auto &gpu = ncnn::get_gpu_info(0);
  printf("GRAPH_BASE=%s\n",base.c_str());
  printf("GRAPH_DEVICE name=%s driver=%u api=%u\n", gpu.device_name(),
         gpu.driver_version(), gpu.api_version());
  SuiteAlloc alloc(dev);
  ncnn::VkStagingAllocator staging(dev);
  try {
    auto load = SuiteClock::now();
    ncnn::Net net;
    net.opt.use_vulkan_compute = true;
    // Keep pack1 + no packing_layout for all graphs, including memenc_*.
    // The previous memenc-only packing=true + DepthWise pack4 bypass matched the
    // crash workaround but diverged from PC prepare (packing=false) and produced
    // near-total maskmem_features errors while maskmem_pos_enc stayed exact.
    net.opt.use_packing_layout = false;
    net.opt.use_fp16_packed = net.opt.use_fp16_storage =
        net.opt.use_fp16_arithmetic = net.opt.use_fp16_uniform = false;
    net.opt.blob_vkallocator = &alloc;
    net.opt.workspace_vkallocator = &alloc;
    net.opt.staging_vkallocator = &staging;
    net.set_vulkan_device(0);
    if (net.load_param((base + "/model.param").c_str()))
      throw std::runtime_error("load_param");
    int unsupported=0;
    for (size_t i = 0; i < net.layers().size(); i++) {
      auto *l = net.layers()[i];
      bool meta = l->type == "Input" || l->type == "Split";
      printf("GRAPH_MAP index=%zu type=%s name=%s vulkan=%d metadata=%d\n", i,
             l->type.c_str(), l->name.c_str(), int(l->support_vulkan),
             int(meta));
      if (!meta && !l->support_vulkan)
        unsupported++;
    }
    if(unsupported)throw std::runtime_error("GPU_UNSUPPORTED_COUNT "+std::to_string(unsupported));
    if (net.load_model((base + "/model.bin").c_str()))
      throw std::runtime_error("load_model");
    printf("GRAPH_LOAD ms=%.6f\n", SuiteMs(load));
    SuiteRoPELayout rope(dev, net.opt);
    std::map<std::string, SuiteRef> refs;
    std::ifstream mf(base + "/refs.txt");
    std::string key, p, o;
    size_t n;
    while (mf >> key >> n >> p >> o)
      refs[key] = {n, p, o};
    if (refs.empty())
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

    std::vector<ncnn::Mat> host;
    auto up=SuiteClock::now();ncnn::VkCompute upload(dev);size_t uploaded=0;
    std::ifstream inputs(base+"/inputs.txt");std::string inputName,dtype,file;int dims,w,h,d,c,batch;
    while(inputs>>inputName>>dims>>w>>h>>d>>c>>batch>>dtype>>file){
      printf("GRAPH_INPUT name=%s dims=%d w=%d h=%d d=%d c=%d batch=%d dtype=%s\n",inputName.c_str(),dims,w,h,d,c,batch,dtype.c_str());
      if(dtype!="float32"||batch!=1)throw std::runtime_error("UNSUPPORTED_INPUT_DTYPE_OR_BATCH "+inputName+" "+dtype);
      ncnn::Mat m;if(dims==1)m.create(w,(size_t)4u);else if(dims==2)m.create(w,h,(size_t)4u);else if(dims==3)m.create(w,h,c,(size_t)4u);else if(dims==4)m.create(w,h,d,c,(size_t)4u);else throw std::runtime_error("input_rank");
      std::ifstream f(base+"/"+file,std::ios::binary);size_t plane=size_t(w)*h*d;
      for(int ch=0;ch<c;ch++)if(!f.read(static_cast<char*>(m.channel(ch).data),plane*4))throw std::runtime_error("input_read "+inputName);
      host.push_back(m);uploaded+=plane*c*4;bool found=false;
      for(size_t j=0;j<net.blobs().size();j++)if(net.blobs()[j].name==inputName){upload.record_upload(host.back(),blobs[j],net.opt);found=true;}
      if(!found)throw std::runtime_error("input_name "+inputName);
    }
    if(host.empty())throw std::runtime_error("input_manifest_empty");
    std::vector<ncnn::VkMat> uploadedPacked = blobs;
    for (auto &b : blobs)
      if (!b.empty() && b.elempack != 1) {
        ncnn::VkMat plain;
        dev->convert_packing(b, plain, 1, 1, upload, net.opt);
        b = plain;
      }
    if (upload.submit_and_wait())
      throw std::runtime_error("upload_submit");
    printf("GRAPH_UPLOAD ms=%.6f bytes=%zu includes_file_read=1\n", SuiteMs(up),
           uploaded);
    host.clear();
    auto graph = SuiteClock::now();
    double encTotal = 0, waitTotal = 0, readTotal = 0, compareTotal = 0;
    size_t checkpoints = 0;
    for (size_t i = 0; i < net.layers().size(); i++) {
      auto *l = net.layers()[i];
      if (l->type == "Input")
        continue;
      auto start = SuiteClock::now();
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
        if (m.elempack != 1 || m.elemsize != 4 || (m.c != 1 && oldplane != m.cstep) ||
            oldplane * m.c != newplane * shape[4] || (shape[4] != 1 && newplane % 4)) {
          printf("GRAPH_LAYOUT name=%s dims=%d w=%d h=%d d=%d c=%d cstep=%zu "
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
        m.cstep = (newplane+3)/4*4;
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
          printf("GRAPH_CUSTOM_SPLIT %s\n", l->name.c_str());
          rc = rope.split(ins[0], outs, cmd, net.opt);
        } else if (l->type == "Concat" && ins.size() == 2 && ins[0].dims == 4 &&
                   ins[0].w == 1) {
          printf("GRAPH_CUSTOM_JOIN %s\n", l->name.c_str());
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
            printf("GRAPH_GPU_UNPACK layer=%s pack=%d\n", l->name.c_str(),
                   outs[oi].elempack);
            ncnn::VkMat plain;
            dev->convert_packing(outs[oi], plain, 1, 1, cmd, net.opt);
            outs[oi] = plain;
          }
        enc = SuiteMs(start);
        auto sub = SuiteClock::now();
        rc = cmd.submit_and_wait();
        wait = SuiteMs(sub);
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
          "GRAPH_OP index=%zu name=%s type=%s encode_ms=%.6f submit_wait_ms=%.6f "
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
        auto t = SuiteClock::now();
        ncnn::VkCompute read(dev);
        read.record_download(blobs[top], m, net.opt);
        if (read.submit_and_wait())
          throw std::runtime_error("read_submit " + blob);
        double readms = SuiteMs(t);
        readTotal += readms;
        t = SuiteClock::now();
        printf("GRAPH_READ_TYPE blob=%s dims=%d pack=%d elemsize=%zu\n",
               blob.c_str(), m.dims, m.elempack, m.elemsize);
        if (m.elempack != 1) {
          ncnn::Mat plain;
          ncnn::convert_packing(m, plain, 1, net.opt);
          m = plain;
          printf("GRAPH_READ_LAYOUT_CONVERT blob=%s pack=%d elemsize=%zu\n",
                 blob.c_str(), m.elempack, m.elemsize);
        }
        if (m.elempack != 1 || m.elemsize != 4)
          throw std::runtime_error("read_type " + blob + " pack=" +
                                   std::to_string(m.elempack) + " size=" +
                                   std::to_string(m.elemsize));
        std::vector<float> a;
        size_t plane = size_t(m.w) * m.h * m.d;
        for (int c = 0; c < m.c; c++) {
          const float *v = m.channel(c);
          a.insert(a.end(), v, v + plane);
        }
        if (a.size() != r->second.n)
          throw std::runtime_error("read_shape " + blob);
        if (!SuiteCompare(a, base + "/" + r->second.pc, l->name + "/ONNX_REFERENCE"))
          result = 10;
        if (r->second.pc != r->second.onnx && !SuiteCompare(a, base + "/" + r->second.onnx,
                      l->name + (r->second.pc == r->second.onnx ? "/PC_AUX"
                                                                : "/ONNX")))
          result = 10;
        std::ofstream f(base + "/gpu_" + blob + ".bin", std::ios::binary);
        f.write(reinterpret_cast<const char *>(a.data()), a.size() * 4);
        compareTotal += SuiteMs(t);
        checkpoints++;
        printf("GRAPH_READBACK blob=%s bytes=%zu ms=%.6f\n", blob.c_str(),
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
        printf("GRAPH_PROCESS %s\n", line.c_str());
    printf("GRAPH_SUMMARY layers=%zu checkpoints=%zu encode_ms=%.6f "
           "submit_wait_ms=%.6f readback_ms=%.6f compare_save_ms=%.6f "
           "diagnostic_wall_ms=%.6f peak_requested_bytes=%zu "
           "cpu_forward_calls=0 result=%d\n",
           net.layers().size(), checkpoints, encTotal, waitTotal, readTotal,
           compareTotal, SuiteMs(graph), alloc.peak, result);
    if (checkpoints != refs.size())
      throw std::runtime_error("incomplete");
  } catch (const std::exception &e) {
    printf("GRAPH_FAILURE reason=%s\n", e.what());
    result = 20;
  }
  printf("GRAPH_EXIT=%d\n", result);
  return result;
}
