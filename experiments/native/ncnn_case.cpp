#include <net.h>
#include <gpu.h>
#include <cstdio>
#include <fstream>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <string>

int main(int argc,char** argv) {
    setvbuf(stdout,nullptr,_IONBF,0);
    if(argc!=6) {fprintf(stderr,"usage: ncnn_case cpu|vulkan param bin memattn_004_dir output.bin\n");return 2;}
    bool gpu=std::string(argv[1])=="vulkan";
    if(!gpu && std::string(argv[1])!="cpu")return 2;
    ncnn::create_gpu_instance();
    if(gpu && ncnn::get_gpu_count()==0) return 3;
    int result=0;
    {
        ncnn::Net net;
        net.opt.num_threads=4;net.opt.use_vulkan_compute=gpu;
        net.opt.use_fp16_packed=net.opt.use_fp16_storage=net.opt.use_fp16_arithmetic=false;
        net.opt.use_fp16_uniform=false;net.opt.use_packing_layout=false;
        if(gpu) net.set_vulkan_device(0);
        auto start=std::chrono::steady_clock::now();
        if(net.load_param(argv[2]) || net.load_model(argv[3]))return 4;
        printf("stage=load mode=%s ms=%.3f ncnn=20261020\n",argv[1],std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
        auto ex=net.create_extractor();
        const char* names[]={"curr","curr_pos","memory","memory_pos"};
        for(int i=0;i<4;i++) {
            int width=i<2?256:64,channels=i<2?4096:4100;
            ncnn::Mat m(width,1,channels);
            std::ifstream f(std::string(argv[4])+"/in_"+names[i]+".bin",std::ios::binary);
            for(int c=0;c<channels;c++) if(!f.read(static_cast<char*>(m.channel(c).data),width*4))return 5;
            if(ex.input(("in"+std::to_string(i)).c_str(),m))return 6;
        }
        ncnn::Mat out;start=std::chrono::steady_clock::now();
        int rc=ex.extract("out0",out);
        printf("stage=extract status=%d mode=%s wall_ms=%.3f hardware_execution=%s\n",rc,argv[1],std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count(),gpu?"MIXED_OR_GPU_UNPROVEN":"PC_OR_DEVICE_CPU");
        if(rc || out.w!=64 || out.h!=64 || out.c!=256 || out.elempack!=1)return 7;
        std::ofstream saved(argv[5],std::ios::binary);
        std::ifstream golden(std::string(argv[4])+"/out_pix_feat_with_mem.bin",std::ios::binary);
        double worst=0;size_t bad=0,nonfinite=0;
        for(int c=0;c<256;c++) {
            const float* p=out.channel(c);
            if(!saved.write(reinterpret_cast<const char*>(p),4096*4))return 8;
            for(int i=0;i<4096;i++) {
                float ref;if(!golden.read(reinterpret_cast<char*>(&ref),4))return 9;
                if(!std::isfinite(p[i])||!std::isfinite(ref)) {nonfinite++;continue;}
                double d=std::abs(double(p[i])-ref);worst=std::max(worst,d);
                if(d>1e-4+1e-4*std::abs(double(ref)))bad++;
            }
        }
        printf("compare=golden maxabs=%.9g violations=%zu total=1048576 nonfinite=%zu\n",worst,bad,nonfinite);
        if(bad||nonfinite)result=10;
    }
    ncnn::destroy_gpu_instance();return result;
}
