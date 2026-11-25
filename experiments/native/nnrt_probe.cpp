#include <mindspore/context.h>
#include <mindspore/model.h>
#include <mindspore/tensor.h>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <fstream>
#include <sstream>
#include <map>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

int main(int argc,char** argv) {
    setvbuf(stdout,nullptr,_IONBF,0);
    if(argc!=3 && argc!=5) {fprintf(stderr,"usage: nnrt_probe cpu|nnrt|accelerator model.ms [case_dir output_prefix]\n");return 2;}
    if(strcmp(argv[1],"cpu") && strcmp(argv[1],"nnrt") && strcmp(argv[1],"accelerator")) return 2;
    size_t n=0; auto* descs=OH_AI_GetAllNNRTDeviceDescs(&n);
    printf("stage=enumerate nnrt_devices=%zu\n",n);
    for(size_t i=0;i<n;i++) {
        auto* d=OH_AI_GetElementOfNNRTDeviceDescs(descs,i);
        printf("device[%zu]=%s\n",i,OH_AI_GetNameFromNNRTDeviceDesc(d));
    }
    OH_AI_DestroyAllNNRTDeviceDescs(&descs);
    auto ctx=OH_AI_ContextCreate(); if(!ctx) return 3;
    OH_AI_ContextSetThreadNum(ctx,4);
    if(strcmp(argv[1],"cpu")) {
        auto accel=strcmp(argv[1],"accelerator")==0
            ? OH_AI_CreateNNRTDeviceInfoByType(OH_AI_NNRTDEVICE_ACCELERATOR)
            : OH_AI_DeviceInfoCreate(OH_AI_DEVICETYPE_NNRT);
        if(!accel) {printf("stage=context accelerator_null=1\n");return 4;}
        OH_AI_ContextAddDeviceInfo(ctx,accel);
    }
    auto cpu=OH_AI_DeviceInfoCreate(OH_AI_DEVICETYPE_CPU);
    if(!cpu) return 5;
    OH_AI_DeviceInfoSetEnableFP16(cpu,false); OH_AI_ContextAddDeviceInfo(ctx,cpu);
    auto model=OH_AI_ModelCreate(); if(!model) return 6;
    auto start=std::chrono::steady_clock::now();
    auto status=OH_AI_ModelBuildFromFile(model,argv[2],OH_AI_MODELTYPE_MINDIR,ctx);
    printf("stage=build mode=%s model=%s status=%d ms=%.3f\n",argv[1],argv[2],int(status),
        std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
    int exitcode=status==OH_AI_STATUS_SUCCESS?0:7;
    if(status==OH_AI_STATUS_SUCCESS) {
        auto inputs=OH_AI_ModelGetInputs(model);
        for(size_t i=0;i<inputs.handle_num;i++) {
            auto t=inputs.handle_list[i]; size_t ndim=0; const auto* dims=OH_AI_TensorGetShape(t,&ndim);
            printf("input=%s dtype=%d bytes=%zu shape=",OH_AI_TensorGetName(t),int(OH_AI_TensorGetDataType(t)),OH_AI_TensorGetDataSize(t));
            for(size_t j=0;j<ndim;j++) printf("%lld,",(long long)dims[j]);
            printf("\n");
        }
        if(argc==5) {
            std::map<std::string,std::string> files;
            std::ifstream manifest(std::string(argv[3])+"/case.txt");
            std::string line;
            while(std::getline(manifest,line)) {
                std::istringstream s(line); std::string tag,key,dtype,file; int rank; s>>tag;
                if(tag!="tensor") continue;
                s>>key>>dtype>>rank; long long dim; for(int j=0;j<rank;j++) s>>dim;
                s>>file; files[key]=std::string(argv[3])+"/"+file;
            }
            bool valid=bool(manifest.eof()) && !files.empty();
            for(size_t i=0;i<inputs.handle_num;i++) {
                auto t=inputs.handle_list[i]; std::string key="in_";key+=OH_AI_TensorGetName(t);
                std::ifstream f(files[key],std::ios::binary);
                auto bytes=OH_AI_TensorGetDataSize(t);
                void* data=OH_AI_TensorGetMutableData(t);
                if(!data || !f.read(static_cast<char*>(data),bytes)) {valid=false;printf("stage=input_read failed=%s\n",key.c_str());}
            }
            if(valid) {
                auto outputs=OH_AI_ModelGetOutputs(model);
                start=std::chrono::steady_clock::now();
                status=OH_AI_ModelPredict(model,inputs,&outputs,nullptr,nullptr);
                printf("stage=predict status=%d ms=%.3f\n",int(status),std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
                if(status!=OH_AI_STATUS_SUCCESS) exitcode=9;
                else for(size_t i=0;i<outputs.handle_num;i++) {
                    auto t=outputs.handle_list[i]; std::string name=OH_AI_TensorGetName(t);
                    // Index avoids model-provided names becoming filesystem paths.
                    auto path=std::string(argv[4])+".out"+std::to_string(i)+".bin";
                    size_t bytes=OH_AI_TensorGetDataSize(t); const void* data=OH_AI_TensorGetData(t);
                    std::ofstream f(path,std::ios::binary);
                    if(!data || !f.write(static_cast<const char*>(data),bytes)) {exitcode=10;continue;}
                    printf("output=%s dtype=%d bytes=%zu saved=%s\n",name.c_str(),int(OH_AI_TensorGetDataType(t)),bytes,path.c_str());
                    auto it=files.find("out_"+name);
                    if(it==files.end() && outputs.handle_num==1) {
                        for(auto j=files.begin();j!=files.end();++j) if(j->first.rfind("out_",0)==0) {it=j;break;}
                    }
                    if(it!=files.end() && OH_AI_TensorGetDataType(t)==OH_AI_DATATYPE_NUMBERTYPE_FLOAT32) {
                        std::vector<float> golden(bytes/4); std::ifstream g(it->second,std::ios::binary);
                        if(!g.read(reinterpret_cast<char*>(golden.data()),bytes)) {exitcode=11;continue;}
                        const auto* values=static_cast<const float*>(data);size_t bad=0,nonfinite=0;double worst=0;
                        for(size_t j=0;j<golden.size();j++) {
                            if(!std::isfinite(values[j])||!std::isfinite(golden[j])) {nonfinite++;continue;}
                            double d=std::abs(double(values[j])-golden[j]); worst=std::max(worst,d);
                            if(d>1e-4+1e-4*std::abs(double(golden[j]))) bad++;
                        }
                        printf("compare=%s maxabs=%.9g violations=%zu total=%zu nonfinite=%zu atol=1e-4 rtol=1e-4\n",name.c_str(),worst,bad,golden.size(),nonfinite);
                        if(bad||nonfinite) exitcode=12;
                    }
                }
            } else exitcode=8;
        }
    }
    printf("hardware_execution=UNPROVEN scope=build_predict_diagnostic exit_code=%d\n",exitcode);
    OH_AI_ModelDestroy(&model); OH_AI_ContextDestroy(&ctx);
    return exitcode;
}
