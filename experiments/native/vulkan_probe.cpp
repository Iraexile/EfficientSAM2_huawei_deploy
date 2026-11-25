// Explicit Vulkan-only layers: no Net/extractor CPU fallback path.
#include <gpu.h>
#include <layer.h>
#include <command.h>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <string>

static void check(int rc, const char* stage) {
    if (rc) throw std::runtime_error(std::string(stage)+" rc="+std::to_string(rc));
}
static ncnn::Layer* layer(const char* type, ncnn::VulkanDevice* dev,
                          const ncnn::Option& opt, const ncnn::ParamDict& p) {
    auto* l=ncnn::create_layer_vulkan(type);
    if (!l || !l->support_vulkan) throw std::runtime_error(std::string(type)+" Vulkan unavailable");
    l->vkdev=dev;
    check(l->load_param(p), "load_param");
    check(l->create_pipeline(opt), "create_pipeline");
    return l;
}
int main() {
    setvbuf(stdout,nullptr,_IONBF,0);
    printf("probe=ncnn-explicit-vulkan ncnn=20261020 precision=fp32\n");
    ncnn::create_gpu_instance();
    int count=ncnn::get_gpu_count();
    printf("stage=enumerate gpu_count=%d\n",count);
    if(count<=0) return 2;
    auto* dev=ncnn::get_gpu_device(0);
    const auto& info=ncnn::get_gpu_info(0);
    printf("device=%s vendor=%u device_id=%u driver=%u api=%u compute_family=%u queues=%u\n",
        info.device_name(),info.vendor_id(),info.device_id(),info.driver_version(),info.api_version(),
        info.compute_queue_family_index(),info.compute_queue_count());
    if(!dev || !info.compute_queue_count()) return 3;
    int result=0;
    {
        ncnn::Option opt;
        opt.use_vulkan_compute=true;
        opt.use_fp16_packed=opt.use_fp16_storage=opt.use_fp16_arithmetic=false;
        opt.use_fp16_uniform=false;
        opt.use_packing_layout=false;
        opt.blob_vkallocator=dev->acquire_blob_allocator();
        opt.workspace_vkallocator=opt.blob_vkallocator;
        opt.staging_vkallocator=dev->acquire_staging_allocator();
        try {
            constexpr int N=16,D=16;
            ncnn::Mat q(D,N), kt(N,D), v(D,N);
            for(int i=0;i<N;i++) for(int j=0;j<D;j++) {
                q.row(i)[j]=float((i*7+j*3)%19-9)/32;
                kt.row(j)[i]=float((i*5+j*2)%17-8)/32;
                v.row(i)[j]=float((i*3+j*7)%23-11)/32;
            }
            ncnn::ParamDict mp; mp.set(0,1.f); mp.set(1,1.f); mp.set(2,0); mp.set(3,0);
            ncnn::ParamDict sp; sp.set(0,1); sp.set(1,1); // modern Softmax axis convention
            printf("matrix_backend=Gemm_vulkan alpha=1 beta=1 transA=0 transB=0\n");
            auto* mm=layer("Gemm",dev,opt,mp);
            auto* sm=layer("Softmax",dev,opt,sp);
            ncnn::VkMat a,b,c;
            std::vector<ncnn::VkMat> scores(1), output(1);
            ncnn::Mat score_cpu,out_cpu;
            auto start=std::chrono::steady_clock::now();
            ncnn::VkCompute cmd(dev);
            cmd.record_upload(q,a,opt); cmd.record_upload(kt,b,opt); cmd.record_upload(v,c,opt);
            check(mm->forward({a,b},scores,cmd,opt),"matmul_dispatch");
            cmd.record_download(scores[0],score_cpu,opt);
            // Submit separately so MatMul has an independent numerical check.
            check(cmd.submit_and_wait(),"matmul_submit_and_wait");
            for(int i=0;i<N;i++) {printf("GPU_MATMUL_ROW %d",i);for(int j=0;j<N;j++)printf(" %.9g",score_cpu.row(i)[j]);printf("\n");}
            if(score_cpu.dims!=2 || score_cpu.w!=N || score_cpu.h!=N || score_cpu.elempack!=1)
                throw std::runtime_error("MatMul shape/packing mismatch");
            double mmerr=0;
            std::vector<double> ref(N*N), expected(N*D);
            for(int i=0;i<N;i++) for(int j=0;j<N;j++) {
                double sum=0; for(int k=0;k<D;k++) sum+=double(q.row(i)[k])*kt.row(k)[j];
                ref[i*N+j]=sum;
                if(!std::isfinite(score_cpu.row(i)[j])) throw std::runtime_error("nonfinite MatMul output");
                mmerr=std::max(mmerr,std::abs(sum-score_cpu.row(i)[j]));
            }
            ncnn::VkCompute attn(dev);
            check(sm->forward_inplace(scores[0],attn,opt),"softmax_dispatch");
            check(mm->forward({scores[0],c},output,attn,opt),"attention_av_dispatch");
            attn.record_download(output[0],out_cpu,opt);
            check(attn.submit_and_wait(),"attention_submit_and_wait");
            for(int i=0;i<N;i++) {printf("GPU_ATTENTION_ROW %d",i);for(int j=0;j<D;j++)printf(" %.9g",out_cpu.row(i)[j]);printf("\n");}
            if(out_cpu.dims!=2 || out_cpu.w!=D || out_cpu.h!=N || out_cpu.elempack!=1)
                throw std::runtime_error("Attention shape/packing mismatch");
            double err=0;
            for(int i=0;i<N;i++) {
                double mx=ref[i*N],den=0;
                for(int j=0;j<N;j++) mx=std::max(mx,ref[i*N+j]);
                for(int j=0;j<N;j++) den+=std::exp(ref[i*N+j]-mx);
                for(int k=0;k<D;k++) {
                    double sum=0; for(int j=0;j<N;j++) sum+=std::exp(ref[i*N+j]-mx)/den*v.row(j)[k];
                    if(!std::isfinite(out_cpu.row(i)[k])) throw std::runtime_error("nonfinite Attention output");
                    err=std::max(err,std::abs(sum-out_cpu.row(i)[k]));
                }
            }
            double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            printf("stage=completed explicit_vkmat_dispatch=1 submissions=2 matmul_maxabs=%.9g attention_maxabs=%.9g cold_with_transfers_ms=%.3f\n",mmerr,err,ms);
            if(!std::isfinite(mmerr)||!std::isfinite(err)||mmerr>1e-4||err>1e-4) result=4;
            sm->destroy_pipeline(opt); mm->destroy_pipeline(opt); delete sm; delete mm;
        } catch(const std::exception& e) {fprintf(stderr,"stage=failed %s\n",e.what());result=5;}
        dev->reclaim_blob_allocator(opt.blob_vkallocator);
        dev->reclaim_staging_allocator(opt.staging_vkallocator);
    }
    ncnn::destroy_gpu_instance();
    printf("exit_code=%d scope=minimal_unscaled_attention_not_SAM2\n",result);
    return result;
}
