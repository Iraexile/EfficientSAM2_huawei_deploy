#include <napi/native_api.h>
#include <hilog/log.h>
#include <unistd.h>
#include <fstream>
#include <sstream>
#define main RunGpuProbe
#include "../../../../experiments/native/vulkan_probe.cpp"
#undef main
#include "../../../../experiments/native/memattn_strict.cpp"
#include "../../../../experiments/native/gpu_graph.cpp"
static napi_value Run(napi_env env,napi_callback_info info) {
    size_t argc=1;napi_value argv[1];napi_get_cb_info(env,info,&argc,argv,nullptr,nullptr);
    char path[1024]={0};size_t len=0;napi_get_value_string_utf8(env,argv[0],path,sizeof(path),&len);
    fflush(stdout);fflush(stderr);int out=dup(1),err=dup(2);
    FILE* file=fopen(path,"w");
    if(!file){napi_throw_error(env,nullptr,"Cannot open GPU probe log");return nullptr;}
    std::string stderrPath=std::string(path)+".stderr";
    FILE* errorFile=fopen(stderrPath.c_str(),"w");
    if(!errorFile){fclose(file);napi_throw_error(env,nullptr,"Cannot open GPU stderr log");return nullptr;}
    dup2(fileno(file),1);dup2(fileno(errorFile),2);
    std::string filePath(path);std::string base=filePath.substr(0,filePath.find_last_of('/'));
    int rc=filePath.find("graph-probe.txt")!=std::string::npos?RunGpuGraph(base.c_str()):filePath.find("k01-probe.txt")!=std::string::npos?RunMemattnStrict(base.c_str()):RunGpuProbe();printf("APP_PROBE_EXIT=%d\n",rc);fflush(stdout);fflush(stderr);
    dup2(out,1);dup2(err,2);close(out);close(err);fclose(file);fclose(errorFile);
    std::ifstream f(path);std::stringstream s;s<<f.rdbuf();std::string text=s.str(),line;
    std::istringstream lines(text);while(std::getline(lines,line))OH_LOG_Print(LOG_APP,LOG_INFO,0x1234,"SAM2GPU","%{public}s",line.c_str());
    napi_value result;napi_create_string_utf8(env,text.c_str(),text.size(),&result);return result;
}
static napi_value Init(napi_env env,napi_value exports){napi_property_descriptor d={"run",nullptr,Run,nullptr,nullptr,nullptr,napi_default,nullptr};napi_define_properties(env,exports,1,&d);return exports;}
static napi_module module={1,0,nullptr,Init,"gpu_probe",nullptr,{0}};
extern "C" __attribute__((constructor)) void RegisterGpu(){napi_module_register(&module);}
