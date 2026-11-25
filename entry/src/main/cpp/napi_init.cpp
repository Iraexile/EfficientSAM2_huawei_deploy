#include "napi/native_api.h"

#include <string>

#include "mslite_runner.h"

namespace {

std::string GetStr(napi_env env, napi_value v) {
    size_t len = 0;
    napi_get_value_string_utf8(env, v, nullptr, 0, &len);
    std::string s(len, '\0');
    napi_get_value_string_utf8(env, v, &s[0], len + 1, &len);
    return s;
}

napi_value MakeStr(napi_env env, const std::string &s) {
    napi_value out;
    napi_create_string_utf8(env, s.c_str(), s.size(), &out);
    return out;
}

// runCase(modelPath, caseDir, outDir, threads, device?) -> status string
napi_value RunCase(napi_env env, napi_callback_info info) {
    size_t argc = 5;
    napi_value args[5] = {nullptr};
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    std::string modelPath = GetStr(env, args[0]);
    std::string caseDir = GetStr(env, args[1]);
    std::string outDir = GetStr(env, args[2]);
    int32_t threads = 4;
    napi_get_value_int32(env, args[3], &threads);
    std::string device = argc > 4 && args[4] ? GetStr(env, args[4]) : "cpu";
    double ms = -1;
    std::string r = sam2::RunCase(modelPath, caseDir, outDir, threads, &ms, device);
    if (ms >= 0) r += " " + std::to_string(ms) + "ms";
    return MakeStr(env, r);
}

// runAllCases(modelDir, vectorDir, outDir, threads, device?) -> summary string
napi_value RunAllCases(napi_env env, napi_callback_info info) {
    size_t argc = 5;
    napi_value args[5] = {nullptr};
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    std::string modelDir = GetStr(env, args[0]);
    std::string vectorDir = GetStr(env, args[1]);
    std::string outDir = GetStr(env, args[2]);
    int32_t threads = 4;
    napi_get_value_int32(env, args[3], &threads);
    std::string device = argc > 4 && args[4] ? GetStr(env, args[4]) : "cpu";
    return MakeStr(env, sam2::RunAllCases(modelDir, vectorDir, outDir, threads, device));
}

// runPipeline(modelDir, vectorDir, outDir, threads, device?) -> summary string
napi_value RunPipeline(napi_env env, napi_callback_info info) {
    size_t argc = 5;
    napi_value args[5] = {nullptr};
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    std::string modelDir = GetStr(env, args[0]);
    std::string vectorDir = GetStr(env, args[1]);
    std::string outDir = GetStr(env, args[2]);
    int32_t threads = 4;
    napi_get_value_int32(env, args[3], &threads);
    std::string device = argc > 4 && args[4] ? GetStr(env, args[4]) : "cpu";
    return MakeStr(env, sam2::RunPipeline(modelDir, vectorDir, outDir, threads, device));
}

// runNnrtProbe(modelDir, vectorDir, outDir) -> multi-line probe results
napi_value RunNnrtProbe(napi_env env, napi_callback_info info) {
    size_t argc = 3;
    napi_value args[3] = {nullptr};
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    return MakeStr(env, sam2::RunNnrtProbe(GetStr(env, args[0]), GetStr(env, args[1]),
                                           GetStr(env, args[2])));
}

}  // namespace

EXTERN_C_START
static napi_value Init(napi_env env, napi_value exports) {
    napi_property_descriptor desc[] = {
        {"runCase", nullptr, RunCase, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"runAllCases", nullptr, RunAllCases, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"runPipeline", nullptr, RunPipeline, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"runNnrtProbe", nullptr, RunNnrtProbe, nullptr, nullptr, nullptr, napi_default, nullptr},
    };
    napi_define_properties(env, exports, sizeof(desc) / sizeof(desc[0]), desc);
    return exports;
}
EXTERN_C_END

static napi_module sam2Module = {
    .nm_version = 1,
    .nm_flags = 0,
    .nm_filename = nullptr,
    .nm_register_func = Init,
    .nm_modname = "entry",
    .nm_priv = ((void *)0),
    .reserved = {0},
};

extern "C" __attribute__((constructor)) void RegisterEntryModule(void) {
    napi_module_register(&sam2Module);
}
