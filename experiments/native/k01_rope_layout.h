// Fixed FP32 RoPE pair layout kernels. No attention arithmetic is changed.
#include <pipeline.h>
class K01RoPELayout {
  ncnn::Pipeline split_, join_;

public:
  K01RoPELayout(const ncnn::VulkanDevice *dev, const ncnn::Option &opt)
      : split_(dev), join_(dev) {
    const char *splitSource = R"(#version 450

layout(binding=0) readonly buffer X {float x[];};
layout(binding=1) writeonly buffer A {float a[];};
layout(binding=2) writeonly buffer B {float b[];};

void main(){uint i=gl_GlobalInvocationID.x;if(i>=524288u)return;a[i]=x[2*i];b[i]=x[2*i+1];}
)";
    const char *joinSource = R"(#version 450

layout(binding=0) readonly buffer A {float a[];};
layout(binding=1) readonly buffer B {float b[];};
layout(binding=2) writeonly buffer X {float x[];};

void main(){uint i=gl_GlobalInvocationID.x;if(i>=524288u)return;x[2*i]=a[i];x[2*i+1]=b[i];}
)";
    std::vector<uint32_t> spv;
    split_.set_local_size_xyz(256, 1, 1);
    join_.set_local_size_xyz(256, 1, 1);
    if (ncnn::compile_spirv_module(splitSource, opt, spv) ||
        split_.create(spv.data(), spv.size() * 4, {}))
      throw std::runtime_error("rope_split_pipeline");
    spv.clear();
    if (ncnn::compile_spirv_module(joinSource, opt, spv) ||
        join_.create(spv.data(), spv.size() * 4, {}))
      throw std::runtime_error("rope_join_pipeline");
  }
  int split(const ncnn::VkMat &x, std::vector<ncnn::VkMat> &out,
            ncnn::VkCompute &cmd, const ncnn::Option &opt) {
    if (x.h != 128 || x.d != 4096 || x.dims != 4 || x.w != 2 || x.c != 1 || x.elempack != 1 ||
        x.elemsize != 4)
      return -301;
    out[0].create(1, x.h, x.d, x.c, 4u, 1, opt.blob_vkallocator);
    out[1].create(1, x.h, x.d, x.c, 4u, 1, opt.blob_vkallocator);
    ncnn::VkMat dispatcher;
    dispatcher.w = x.h * x.d;
    dispatcher.h = dispatcher.d = dispatcher.c = 1;
    dispatcher.dims = 1;
    std::vector<ncnn::vk_constant_type> p;
    cmd.record_pipeline(&split_, {x, out[0], out[1]}, p, dispatcher);
    return 0;
  }
  int join(const std::vector<ncnn::VkMat> &in, ncnn::VkMat &out,
           ncnn::VkCompute &cmd, const ncnn::Option &opt) {
    if (in.size() != 2 || in[0].h != 128 || in[0].d != 4096 || in[0].dims != 4 || in[0].w != 1 || in[0].c != 1 ||
        in[0].elempack != 1 || in[1].elempack != 1 || in[0].h != in[1].h ||
        in[0].d != in[1].d)
      return -302;
    out.create(2, in[0].h, in[0].d, 1, 4u, 1, opt.blob_vkallocator);
    ncnn::VkMat dispatcher;
    dispatcher.w = in[0].h * in[0].d;
    dispatcher.h = dispatcher.d = dispatcher.c = 1;
    dispatcher.dims = 1;
    std::vector<ncnn::vk_constant_type> p;
    cmd.record_pipeline(&join_, {in[0], in[1], out}, p, dispatcher);
    return 0;
  }
};
