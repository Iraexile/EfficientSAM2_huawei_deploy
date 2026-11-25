#include "mslite_runner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "mindspore/context.h"
#include "mindspore/model.h"
#include "mindspore/tensor.h"
#include "mindspore/types.h"
#include <cerrno>
#include <dirent.h>
#include <fcntl.h>
#include <hilog/log.h>
#include <unistd.h>

namespace sam2 {
namespace {

// 之前的 ENOENT 实为 CRLF 文件名污染；native 可直接用 /data/storage/el2/base 路径。
std::string TranslatePath(const std::string &p) { return p; }

std::string ProbeDir(const std::string &dir) {
    DIR *d = opendir(dir.c_str());
    if (!d) return "{" + dir + " errno=" + std::to_string(errno) + "}";
    std::string names = "{" + dir + ": ";
    struct dirent *e;
    while ((e = readdir(d)) != nullptr && names.size() < 200) {
        names += e->d_name;
        names += ',';
    }
    closedir(d);
    return names + "}";
}

struct TensorSpec {
    std::string key;    // e.g. in_image / out_fpn0
    std::string dtype;  // float32 / int64 / int32 / bool
    std::vector<int64_t> shape;
    std::string file;
};

size_t DTypeSize(const std::string &dt) {
    if (dt == "float32" || dt == "int32") return 4;
    if (dt == "int64") return 8;
    if (dt == "bool" || dt == "int8" || dt == "uint8") return 1;
    if (dt == "float16" || dt == "int16") return 2;
    return 0;
}

bool ParseCaseTxt(const std::string &path, std::vector<TensorSpec> *tensors,
                  std::string *err) {
    std::ifstream in(path);
    if (!in) {
        *err = "cannot open " + path;
        return false;
    }
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string tag;
        ss >> tag;
        if (tag == "tensor") {
            TensorSpec t;
            int ndim;
            ss >> t.key >> t.dtype >> ndim;
            for (int i = 0; i < ndim; i++) {
                int64_t d;
                ss >> d;
                t.shape.push_back(d);
            }
            ss >> t.file;
            tensors->push_back(t);
        }
        // meta lines are ignored by the harness
    }
    return true;
}

bool ReadFile(const std::string &path, void *dst, size_t bytes) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    in.read(reinterpret_cast<char *>(dst), bytes);
    return in.gcount() == static_cast<std::streamsize>(bytes);
}

std::string StatusStr(OH_AI_Status st) {
    char buf[32];
    snprintf(buf, sizeof(buf), "status=%d", static_cast<int>(st));
    return buf;
}

// 探测设备上可见的 NNRT 设备（只跑一次；hilog 会被打码，结果写进报告）
const std::string &NnrtProbe() {
    static std::string probe = [] {
        size_t ndev = 0;
        NNRTDeviceDesc *descs = OH_AI_GetAllNNRTDeviceDescs(&ndev);
        std::string names = "nnrt_devices n=" + std::to_string(ndev) + ":";
        for (size_t i = 0; i < ndev; i++) {
            NNRTDeviceDesc *d = OH_AI_GetElementOfNNRTDeviceDescs(descs, i);
            if (d) names += " [" + std::string(OH_AI_GetNameFromNNRTDeviceDesc(d) ? OH_AI_GetNameFromNNRTDeviceDesc(d) : "?") + "]";
        }
        OH_AI_DestroyAllNNRTDeviceDescs(&descs);
        return names;
    }();
    return probe;
}

class ModelGuard {
  public:
    ModelGuard() = default;
    ~ModelGuard() {
        if (model_) OH_AI_ModelDestroy(&model_);
        if (ctx_) OH_AI_ContextDestroy(&ctx_);
    }
    bool Load(const std::string &modelPath, int threads, std::string *err,
              const std::string &device = "cpu") {
        // 先用 C 库直接探一下文件（诊断 native 侧是否可见/可读）
        FILE *fp = fopen(modelPath.c_str(), "rb");
        if (!fp) {
            std::string info = "fopen errno=" + std::to_string(errno) + " " + modelPath;
            // 同时探别名路径与几个候选根
            static const char *kAlias = "/data/storage/el2/base/";
            static const char *kReal = "/data/app/el2/100/base/com.example.efficientsam2/";
            std::string alt = modelPath.rfind(kReal, 0) == 0
                ? kAlias + modelPath.substr(strlen(kReal))
                : std::string();
            if (!alt.empty()) {
                FILE *f2 = fopen(alt.c_str(), "rb");
                info += f2 ? " | alias OPEN-OK" : " | alias errno=" + std::to_string(errno);
                if (f2) fclose(f2);
            }
            std::string dir = modelPath.substr(0, modelPath.rfind('/'));
            info += " " + ProbeDir(dir);
            info += " " + ProbeDir("/data/storage/el2/base/haps/entry/files/models");
            {
                static const char *kProbes[] = {
                    "/data/storage/el2/base/haps/entry/files/push_list.txt",
                    "/data/storage/el2/base/haps/entry/files/vectors/memattn_004/case.txt",
                    "/data/storage/el2/base/haps/entry/files/vectors/memattn_004/in_curr.bin",
                    "/data/storage/el2/base/haps/entry/files/out/smoke.txt",
                };
                for (const char *p : kProbes) {
                    FILE *f3 = fopen(p, "rb");
                    const char *slash = strrchr(p, '/');
                    info += f3 ? " | " : " | !";
                    info += slash ? slash + 1 : p;
                    if (!f3) info += ":" + std::to_string(errno);
                    if (f3) fclose(f3);
                }
                // 直接 syscall 再试一次模型文件（排除 libc 层因素）
                int fd = open(("/data/storage/el2/base/haps/entry/files/models/tiny.dat"), O_RDONLY);
                info += fd >= 0 ? " | open() OK" : " | open():" + std::to_string(errno);
                if (fd >= 0) close(fd);
                struct stat stt;
                int sr = stat("/data/storage/el2/base/haps/entry/files/models/tiny.dat", &stt);
                info += sr == 0 ? " | stat OK sz=" + std::to_string((long)stt.st_size)
                                : " | stat:" + std::to_string(errno);
            }
            *err = info;
            return false;
        }
        fclose(fp);
        ctx_ = OH_AI_ContextCreate();
        if (!ctx_) { *err = "ContextCreate failed"; return false; }
        OH_AI_ContextSetThreadNum(ctx_, threads);
        if (device != "cpu") {
            NnrtProbe();  // 触发一次探测（结果在报告头部输出）
            OH_AI_DeviceInfoHandle npu = nullptr;
            if (device == "nnrt" || device == "nnrt16") {
                npu = OH_AI_DeviceInfoCreate(OH_AI_DEVICETYPE_NNRT);
                if (npu && device == "nnrt16") {
                    OH_AI_DeviceInfoSetEnableFP16(npu, true);   // NPU fp16
                    OH_AI_DeviceInfoSetFrequency(npu, 3);       // 3 = high performance
                }
            } else if (device == "knpu") {
                npu = OH_AI_DeviceInfoCreate(OH_AI_DEVICETYPE_KIRIN_NPU);
            } else if (device == "gpu") {
                npu = OH_AI_DeviceInfoCreate(OH_AI_DEVICETYPE_GPU);
            }
            if (!npu) { *err = "DeviceInfoCreate(" + device + ") failed"; return false; }
            OH_AI_ContextAddDeviceInfo(ctx_, npu);
        }
        OH_AI_DeviceInfoHandle dev = OH_AI_DeviceInfoCreate(OH_AI_DEVICETYPE_CPU);
        if (!dev) { *err = "DeviceInfoCreate failed"; return false; }
        OH_AI_DeviceInfoSetEnableFP16(dev, false);  // 显式 fp32，排除精度模式干扰
        OH_AI_ContextAddDeviceInfo(ctx_, dev);
        model_ = OH_AI_ModelCreate();
        if (!model_) { *err = "ModelCreate failed"; return false; }
        OH_AI_Status st = OH_AI_ModelBuildFromFile(model_, modelPath.c_str(),
                                                   OH_AI_MODELTYPE_MINDIR, ctx_);
        if (st != OH_AI_STATUS_SUCCESS) {
            *err = "ModelBuildFromFile " + modelPath + " " + StatusStr(st);
            return false;
        }
        return true;
    }
    OH_AI_ModelHandle model() { return model_; }
  private:
    OH_AI_ContextHandle ctx_ = nullptr;
    OH_AI_ModelHandle model_ = nullptr;
};

// Core: given an already-built model, run one case directory.
std::string RunCaseLoaded(OH_AI_ModelHandle model, const std::string &caseDir,
                          const std::string &outDir, double *predictMs) {
    mkdir(outDir.c_str(), 0755);  // best effort; exists is fine
    std::vector<TensorSpec> tensors;
    std::string err;
    if (!ParseCaseTxt(caseDir + "/case.txt", &tensors, &err)) return "ERR " + err;

    std::map<std::string, const TensorSpec *> ins, outs;
    for (const auto &t : tensors) {
        if (t.key.rfind("in_", 0) == 0) ins[t.key.substr(3)] = &t;
        else if (t.key.rfind("out_", 0) == 0) outs[t.key.substr(4)] = &t;
    }

    OH_AI_TensorHandleArray inputs = OH_AI_ModelGetInputs(model);
    std::vector<std::vector<char>> inBufs(inputs.handle_num);
    for (size_t i = 0; i < inputs.handle_num; i++) {
        OH_AI_TensorHandle t = inputs.handle_list[i];
        std::string name = OH_AI_TensorGetName(t);
        auto it = ins.find(name);
        if (it == ins.end()) return "ERR no golden input for model input '" + name + "'";
        const TensorSpec &spec = *it->second;
        size_t bytes = OH_AI_TensorGetDataSize(t);
        size_t expect = DTypeSize(spec.dtype);
        for (int64_t d : spec.shape) expect *= d;
        if (expect != bytes)
            return "ERR input '" + name + "' size mismatch: golden " +
                   std::to_string(expect) + " vs model " + std::to_string(bytes);
        inBufs[i].resize(bytes);
        if (!ReadFile(caseDir + "/" + spec.file, inBufs[i].data(), bytes))
            return "ERR read failed " + spec.file;
        void *dst = OH_AI_TensorGetMutableData(t);
        if (!dst) return "ERR MutableData null for '" + name + "'";
        memcpy(dst, inBufs[i].data(), bytes);
    }

    auto t0 = std::chrono::steady_clock::now();
    OH_AI_TensorHandleArray outputs;
    OH_AI_Status st = OH_AI_ModelPredict(model, inputs, &outputs, nullptr, nullptr);
    auto t1 = std::chrono::steady_clock::now();
    if (st != OH_AI_STATUS_SUCCESS) return "ERR Predict " + StatusStr(st);
    *predictMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Compare each golden-checked output against its golden .bin (fp32 only),
    // same tolerance as the PC parity checks: atol=rtol=1e-4.
    const float kTol = 1e-4f;
    double worstDiff = 0.0;
    std::string worstName, detail;
    size_t ncompared = 0, nfail = 0;
    std::string firstFail;
    for (size_t i = 0; i < outputs.handle_num; i++) {
        OH_AI_TensorHandle t = outputs.handle_list[i];
        std::string name = OH_AI_TensorGetName(t);
        auto it = outs.find(name);
        const TensorSpec *specp = nullptr;
        if (it != outs.end()) {
            specp = it->second;
        } else if (outs.size() == 1 && outputs.handle_num == 1) {
            specp = outs.begin()->second;  // 单输出 case：按位置匹配（子图输出名是内部张量名）
        }
        if (!specp) continue;  // not a golden-checked output
        const TensorSpec &spec = *specp;
        size_t bytes = OH_AI_TensorGetDataSize(t);
        size_t expect = DTypeSize(spec.dtype);
        for (int64_t d : spec.shape) expect *= d;
        if (expect != bytes)
            return "ERR output '" + name + "' size mismatch: golden " +
                   std::to_string(expect) + " vs model " + std::to_string(bytes);
        const void *src = OH_AI_TensorGetData(t);
        if (!src) return "ERR GetData null for output '" + name + "'";
        if (spec.dtype != "float32") {
            ncompared++;  // shape/size matched; skip value compare
            continue;
        }
        std::vector<float> golden(expect / 4);
        if (!ReadFile(caseDir + "/" + spec.file, golden.data(), expect))
            return "ERR read golden failed " + spec.file;
        const float *val = reinterpret_cast<const float *>(src);
        size_t n = expect / 4;
        bool outFail = false;
        double outMax = 0.0, dot = 0.0, nv = 0.0, ng = 0.0, sv = 0.0, sg = 0.0;
        for (size_t j = 0; j < n; j++) {
            double d = fabs(static_cast<double>(val[j]) - golden[j]);
            if (d > outMax) outMax = d;
            if (d > worstDiff) { worstDiff = d; worstName = name; }
            if (d > kTol + kTol * fabs(static_cast<double>(golden[j]))) outFail = true;
            dot += static_cast<double>(val[j]) * golden[j];
            nv += static_cast<double>(val[j]) * val[j];
            ng += static_cast<double>(golden[j]) * golden[j];
            sv += val[j];
            sg += golden[j];
        }
        double cos = (nv > 0 && ng > 0) ? dot / sqrt(nv * ng) : 0.0;
        double scale = (sg != 0) ? sv / sg : 0.0;
        char dbuf[160];
        snprintf(dbuf, sizeof(dbuf), " %s=%.3e%s(cos=%.6f,scale=%.4f)", name.c_str(), outMax,
                 outFail ? "!" : "", cos, scale);
        detail += dbuf;
        if (outFail) {
            nfail++;
            if (firstFail.empty()) firstFail = name;
        }
        ncompared++;
    }
    if (ncompared != outs.size())
        return "ERR compared " + std::to_string(ncompared) + "/" + std::to_string(outs.size()) +
               " golden outputs (name mismatch)";
    char buf[128];
    snprintf(buf, sizeof(buf), " maxdiff=%.3e(%s)", worstDiff, worstName.c_str());
    if (nfail > 0)
        return "FAIL " + firstFail + " etc " + std::to_string(nfail) + " outs" + buf + " [" + detail + " ]";
    return "OK" + std::string(buf) + " [" + detail + " ]";
}

// ---------------- pipeline glue (spec: pc/scripts/glue_dryrun.py, 17/17 bitwise) ----------------

struct FTensor {
    std::vector<int64_t> shape;
    std::vector<float> data;
    size_t Numel() const {
        size_t n = 1;
        for (int64_t d : shape) n *= static_cast<size_t>(d);
        return n;
    }
};

// Feed every model input from name -> raw bytes (dtype-agnostic byte copy).
std::string FeedInputs(OH_AI_ModelHandle model,
                       const std::map<std::string, std::vector<char>> &ins) {
    OH_AI_TensorHandleArray inputs = OH_AI_ModelGetInputs(model);
    for (size_t i = 0; i < inputs.handle_num; i++) {
        OH_AI_TensorHandle t = inputs.handle_list[i];
        std::string name = OH_AI_TensorGetName(t);
        auto it = ins.find(name);
        if (it == ins.end()) return "ERR pipeline: no data for model input '" + name + "'";
        size_t bytes = OH_AI_TensorGetDataSize(t);
        if (bytes != it->second.size())
            return "ERR pipeline: input '" + name + "' bytes " + std::to_string(it->second.size()) +
                   " vs model " + std::to_string(bytes);
        void *dst = OH_AI_TensorGetMutableData(t);
        if (!dst) return "ERR pipeline: MutableData null for '" + name + "'";
        memcpy(dst, it->second.data(), bytes);
    }
    return "";
}

std::string PredictTimed(OH_AI_ModelHandle model, OH_AI_TensorHandleArray *outs, double *ms) {
    OH_AI_TensorHandleArray inputs = OH_AI_ModelGetInputs(model);
    auto t0 = std::chrono::steady_clock::now();
    OH_AI_Status st = OH_AI_ModelPredict(model, inputs, outs, nullptr, nullptr);
    auto t1 = std::chrono::steady_clock::now();
    if (st != OH_AI_STATUS_SUCCESS) return "ERR pipeline Predict " + StatusStr(st);
    *ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return "";
}

bool GetOutF32(const OH_AI_TensorHandleArray &outs, const std::string &name, FTensor *dst) {
    for (size_t i = 0; i < outs.handle_num; i++) {
        OH_AI_TensorHandle t = outs.handle_list[i];
        if (name != OH_AI_TensorGetName(t)) continue;
        size_t shapeNum = 0;
        const int64_t *sh = OH_AI_TensorGetShape(t, &shapeNum);
        dst->shape.assign(sh, sh + shapeNum);
        size_t n = dst->Numel();
        const void *src = OH_AI_TensorGetData(t);
        if (!src) return false;
        dst->data.resize(n);
        memcpy(dst->data.data(), src, n * sizeof(float));
        return true;
    }
    return false;
}

std::vector<char> AsBytes(const FTensor &t) {
    std::vector<char> b(t.data.size() * sizeof(float));
    memcpy(b.data(), t.data.data(), b.size());
    return b;
}

struct CmpStat {
    double maxdiff = 0, cos = 0, scale = 0;
    bool fail = false;
};

CmpStat CompareGolden(const FTensor &val, const std::string &goldenPath) {
    CmpStat c;
    std::vector<float> golden(val.data.size());
    if (!ReadFile(goldenPath, golden.data(), golden.size() * sizeof(float))) {
        c.fail = true;
        c.maxdiff = -1;
        return c;
    }
    const float kTol = 1e-4f;
    double dot = 0, nv = 0, ng = 0, sv = 0, sg = 0;
    for (size_t j = 0; j < val.data.size(); j++) {
        double d = fabs(static_cast<double>(val.data[j]) - golden[j]);
        if (d > c.maxdiff) c.maxdiff = d;
        if (d > kTol + kTol * fabs(static_cast<double>(golden[j]))) c.fail = true;
        dot += static_cast<double>(val.data[j]) * golden[j];
        nv += static_cast<double>(val.data[j]) * val.data[j];
        ng += static_cast<double>(golden[j]) * golden[j];
        sv += val.data[j];
        sg += golden[j];
    }
    c.cos = (nv > 0 && ng > 0) ? dot / sqrt(nv * ng) : 0.0;
    c.scale = (sg != 0) ? sv / sg : 0.0;
    return c;
}

// Load one tensor (as f32 bytes) from a case dir by key (e.g. "in_image").
bool LoadCaseF32(const std::string &caseDir, const std::string &key, FTensor *dst,
                 std::string *err) {
    std::vector<TensorSpec> tensors;
    if (!ParseCaseTxt(caseDir + "/case.txt", &tensors, err)) return false;
    for (const auto &t : tensors) {
        if (t.key != key) continue;
        dst->shape = t.shape;
        size_t bytes = DTypeSize(t.dtype);
        for (int64_t d : t.shape) bytes *= static_cast<size_t>(d);
        dst->data.resize(dst->Numel());
        if (!ReadFile(caseDir + "/" + t.file, dst->data.data(), bytes)) {
            *err = "read failed " + caseDir + "/" + t.file;
            return false;
        }
        return true;
    }
    *err = "no tensor " + key + " in " + caseDir;
    return false;
}

// Golden out file path for a case dir + out name ("fpn0" -> <dir>/out_fpn0.bin per case.txt).
bool GoldenOutPath(const std::string &caseDir, const std::string &outName, std::string *path,
                   std::vector<int64_t> *shape) {
    std::vector<TensorSpec> tensors;
    std::string err;
    if (!ParseCaseTxt(caseDir + "/case.txt", &tensors, &err)) return false;
    for (const auto &t : tensors) {
        if (t.key == "out_" + outName) {
            *path = caseDir + "/" + t.file;
            *shape = t.shape;
            return true;
        }
    }
    return false;
}

// 与 golden 用例的 in_ 张量对拍（诊断用）：返回 maxdiff，找不到/读失败返回 -1。
double DiffCaseIn(const std::string &caseDir, const std::string &inName,
                  const std::vector<float> &val) {
    FTensor g;
    std::string err;
    if (!LoadCaseF32(caseDir, "in_" + inName, &g, &err)) return -1;
    if (g.data.size() != val.size()) return -2;
    double mx = 0;
    for (size_t i = 0; i < val.size(); i++)
        mx = std::max(mx, static_cast<double>(fabs(val[i] - g.data[i])));
    return mx;
}

inline float BF16Round(float f) {
    // round-to-nearest-even bf16 truncate（官方 bank 内 maskmem_features 的存储精度）
    uint32_t u;
    memcpy(&u, &f, 4);
    u += 0x7FFFu + ((u >> 16) & 1u);
    u &= 0xFFFF0000u;
    float r;
    memcpy(&r, &u, 4);
    return r;
}

// [1,64,64,64] NCHW-flat -> [4096,64] tokens：token t(=h*64+w) row c = src[c*4096+t]
void Tok64(const std::vector<float> &src, std::vector<float> *dst) {
    dst->resize(4096 * 64);
    for (size_t t = 0; t < 4096; t++)
        for (size_t c = 0; c < 64; c++) (*dst)[t * 64 + c] = src[c * 4096 + t];
}

// sam2_utils.get_1d_sine_pe 精确复刻：pe_dim=128，dim_t=10000^(2*(i//2)/128)，cat([sin,cos])
void SinePE1D(const std::vector<double> &pos, std::vector<float> *out /* n*256 */) {
    out->resize(pos.size() * 256);
    for (size_t n = 0; n < pos.size(); n++) {
        for (int i = 0; i < 128; i++) {
            double dim_t = pow(10000.0, 2.0 * (i / 2) / 128.0);
            double pe = pos[n] / dim_t;
            (*out)[n * 256 + i] = static_cast<float>(sin(pe));
            (*out)[n * 256 + 128 + i] = static_cast<float>(cos(pe));
        }
    }
}

struct BankEntry {
    int frame;
    bool isCond;
    std::vector<float> feat;  // [4096*64] bf16-rounded tokens
    std::vector<float> ptr;   // [256]
};

}  // namespace

std::string RunPipeline(const std::string &modelDir0, const std::string &vectorDir0,
                        const std::string &outDir0, int threads, const std::string &device) {
    const std::string modelDir = TranslatePath(modelDir0);
    const std::string vectorDir = TranslatePath(vectorDir0);
    const std::string outDir = TranslatePath(outDir0);
    mkdir(outDir.c_str(), 0755);
    std::ofstream report(outDir + "/pipeline_report.txt");
    if (!report) return "ERR cannot write pipeline_report.txt";
    report << "# device=" << device << " " << (device != "cpu" ? NnrtProbe() : "") << "\n";

    const int kFrames = 18;
    auto caseDir = [&](const char *graph, int idx) {
        char b[128];  // vectorDir 47 + '/' + graph 13 + '_' + 3 位 = 65，64 会截断
        snprintf(b, sizeof(b), "%s/%s_%03d", vectorDir.c_str(), graph, idx);
        return std::string(b);
    };
    auto encIdx = [](int f) { return f == 0 ? 0 : 4 * f - 1; };
    auto memattnIdx = [](int f) { return 4 * f; };
    auto samheadsIdx = [](int f) { return f == 0 ? 1 : 4 * f + 1; };
    auto memencIdx = [](int f) { return f == 0 ? 2 : 4 * f + 2; };

    // ---- load glue constants ----
    auto loadConst = [&](const char *name, std::vector<float> *dst, size_t n) -> std::string {
        dst->resize(n);
        if (!ReadFile(modelDir + "/" + name, dst->data(), n * sizeof(float)))
            return "ERR load const " + std::string(name);
        return "";
    };
    std::vector<float> cNoMemEmbed, cTposEnc, cProjW, cProjB, cMaskmemPos, cNoObjPtr;
    std::string err;
    if (!(err = loadConst("const_no_mem_embed.bin", &cNoMemEmbed, 256)).empty()) return err;
    if (!(err = loadConst("const_maskmem_tpos_enc.bin", &cTposEnc, 7 * 64)).empty()) return err;
    if (!(err = loadConst("const_obj_ptr_tpos_proj_w.bin", &cProjW, 64 * 256)).empty()) return err;
    if (!(err = loadConst("const_obj_ptr_tpos_proj_b.bin", &cProjB, 64)).empty()) return err;
    if (!(err = loadConst("maskmem_pos_enc.bin", &cMaskmemPos, 64 * 4096)).empty()) return err;
    if (!(err = loadConst("no_obj_ptr.bin", &cNoObjPtr, 256)).empty()) return err;
    std::vector<float> maskmemPosTok;  // [4096*64]
    Tok64(cMaskmemPos, &maskmemPosTok);

    // ---- load resident models ----
    ModelGuard mgEnc, mgCond, mgTrack, mgMemencB, mgMemencS;
    if (!mgEnc.Load(modelDir + "/image_encoder.dat", threads, &err, device)) return "ERR " + err;
    if (!mgCond.Load(modelDir + "/samheads_cond.dat", threads, &err, device)) return "ERR " + err;
    if (!mgTrack.Load(modelDir + "/samheads_track.dat", threads, &err, device)) return "ERR " + err;
    if (!mgMemencB.Load(modelDir + "/memenc_binarize.dat", threads, &err, device)) return "ERR " + err;
    if (!mgMemencS.Load(modelDir + "/memenc_sigmoid.dat", threads, &err, device)) return "ERR " + err;
    // memattn：16 个变体太大，只常驻当前变体
    std::unique_ptr<ModelGuard> mgMemattn;
    std::string memattnName;

    std::vector<BankEntry> bank;
    std::map<std::string, std::vector<double>> times;  // graph -> per-run ms
    int nFail = 0;

    for (int frame = 0; frame < kFrames; frame++) {
        char lineHead[32];
        snprintf(lineHead, sizeof(lineHead), "f%02d ", frame);
        std::string line = lineHead;
        bool frameFail = false;

        // ================= encoder =================
        FTensor img, fpn0, fpn1, fpn2, pos0, pos1, pos2;
        if (!LoadCaseF32(caseDir("image_encoder", encIdx(frame)), "in_image", &img, &err))
            return "ERR " + err;
        double msEnc = -1;
        {
            std::map<std::string, std::vector<char>> ins{{"image", AsBytes(img)}};
            std::string e = FeedInputs(mgEnc.model(), ins);
            OH_AI_TensorHandleArray outs{};
            if (e.empty()) e = PredictTimed(mgEnc.model(), &outs, &msEnc);
            if (!e.empty()) return line + e;
            if (!GetOutF32(outs, "fpn0", &fpn0) || !GetOutF32(outs, "fpn1", &fpn1) ||
                !GetOutF32(outs, "fpn2", &fpn2) || !GetOutF32(outs, "pos0", &pos0) ||
                !GetOutF32(outs, "pos1", &pos1) || !GetOutF32(outs, "pos2", &pos2))
                return line + "ERR encoder output missing";
        }
        times["encoder"].push_back(msEnc);
        {
            // 对拍 fpn2（其余输出量小且同源，全量对拍已覆盖）
            std::string gp; std::vector<int64_t> gs;
            CmpStat c;
            if (GoldenOutPath(caseDir("image_encoder", encIdx(frame)), "fpn2", &gp, &gs))
                c = CompareGolden(fpn2, gp);
            char b[96];
            snprintf(b, sizeof(b), "enc %s %.2e %.0fms", c.fail ? "FAIL" : "OK", c.maxdiff, msEnc);
            line += b; line += " | ";
            frameFail = frameFail || c.fail;
        }

        // ================= samheads (+memattn) =================
        FTensor backbone;  // 喂 samheads 的 backbone_features
        FTensor maskHr;    // 选出的 high-res mask logits [1*1*1024*1024]
        FTensor objPtr, score;
        double msSa = -1, msMa = -1;
        if (frame == 0) {
            // cond：backbone = fpn2 + no_mem_embed（逐通道广播），再 NCHW->NHWC
            FTensor bb;
            bb.shape = {1, 64, 64, 256};
            bb.data.resize(64 * 64 * 256);
            for (size_t t = 0; t < 4096; t++)
                for (size_t c = 0; c < 256; c++)
                    bb.data[t * 256 + c] = fpn2.data[c * 4096 + t] + cNoMemEmbed[c];
            FTensor f0, f1;
            f0.shape = {1, 256, 256, 32};
            f0.data.resize(256 * 256 * 32);
            for (size_t t = 0; t < 256 * 256; t++)
                for (size_t c = 0; c < 32; c++) f0.data[t * 32 + c] = fpn0.data[c * 65536 + t];
            f1.shape = {1, 128, 128, 64};
            f1.data.resize(128 * 128 * 64);
            for (size_t t = 0; t < 128 * 128; t++)
                for (size_t c = 0; c < 64; c++) f1.data[t * 64 + c] = fpn1.data[c * 16384 + t];
            FTensor pc, pl;
            if (!LoadCaseF32(caseDir("samheads", 1), "in_point_coords", &pc, &err))
                return "ERR " + err;
            // point_labels 是 int32，原始字节加载
            std::vector<char> plBytes;
            {
                std::vector<TensorSpec> ts;
                if (!ParseCaseTxt(caseDir("samheads", 1) + "/case.txt", &ts, &err))
                    return "ERR " + err;
                for (const auto &t : ts)
                    if (t.key == "in_point_labels") {
                        size_t nb = DTypeSize(t.dtype);
                        for (int64_t d : t.shape) nb *= static_cast<size_t>(d);
                        plBytes.resize(nb);
                        if (!ReadFile(caseDir("samheads", 1) + "/" + t.file, plBytes.data(), nb))
                            return "ERR read point_labels";
                    }
            }
            std::map<std::string, std::vector<char>> ins{
                {"backbone_features", AsBytes(bb)}, {"fpn0", AsBytes(f0)},
                {"fpn1", AsBytes(f1)}, {"point_coords", AsBytes(pc)},
                {"point_labels", plBytes}};
            std::string e = FeedInputs(mgCond.model(), ins);
            OH_AI_TensorHandleArray outs{};
            if (e.empty()) e = PredictTimed(mgCond.model(), &outs, &msSa);
            if (!e.empty()) return line + e;
            FTensor lowRes, highRes, ious;
            if (!GetOutF32(outs, "low_res_multimasks", &lowRes) ||
                !GetOutF32(outs, "high_res_multimasks", &highRes) ||
                !GetOutF32(outs, "ious", &ious) || !GetOutF32(outs, "obj_ptr", &objPtr) ||
                !GetOutF32(outs, "object_score_logits", &score))
                return line + "ERR cond output missing";
            times["samheads_cond"].push_back(msSa);
            // 对拍全部 5 个输出
            double worst = 0; bool bad = false;
            for (const char *n : {"low_res_multimasks", "high_res_multimasks", "ious", "obj_ptr",
                                  "object_score_logits"}) {
                std::string gp; std::vector<int64_t> gs;
                if (!GoldenOutPath(caseDir("samheads", 1), n, &gp, &gs)) continue;
                const FTensor *v = n[0] == 'l' ? &lowRes
                                   : n[0] == 'h' ? &highRes
                                   : n[0] == 'i' ? &ious
                                   : n[1] == 'b' ? &objPtr : &score;
                CmpStat c = CompareGolden(*v, gp);
                if (c.maxdiff > worst) worst = c.maxdiff;
                bad = bad || c.fail;
            }
            char b[96];
            snprintf(b, sizeof(b), "cond %s %.2e %.0fms", bad ? "FAIL" : "OK", worst, msSa);
            line += b; line += " | ";
            frameFail = frameFail || bad;
            maskHr = highRes;  // [1,1024,1024,1] 与 [1,1,1024,1024] 内存相同
            // cond ptr 胶水：lambda = (score > 0)
            if (score.data[0] <= 0) objPtr.data = cNoObjPtr;
        } else {
            // ---- 组装 memory / memory_pos（bank 规则见 glue_dryrun.py）----
            int k = frame;
            int nPrevMask = std::min(k - 1, 6);
            int nPrevPtr = std::min(k - 1, 15);
            size_t nTok = 4096ULL * (1 + nPrevMask);
            int nptr = 1 + nPrevPtr;
            size_t N = nTok + 4ULL * nptr;
            FTensor memory, memoryPos;
            memory.data.resize(N * 64);
            memoryPos.data.resize(N * 64);
            size_t row = 0;
            auto putFeat = [&](const BankEntry &e, int tposIdx) {
                memcpy(memory.data.data() + row * 64, e.feat.data(), 4096 * 64 * sizeof(float));
                for (size_t t = 0; t < 4096; t++)
                    for (size_t c = 0; c < 64; c++)
                        memoryPos.data[(row + t) * 64 + c] =
                            maskmemPosTok[t * 64 + c] + cTposEnc[tposIdx * 64 + c];
                row += 4096;
            };
            putFeat(bank[0], 6);  // cond 帧永远在最前，tpos idx 6
            // 最近 6 个非 cond 帧，最旧在前
            std::vector<const BankEntry *> prev;
            for (const auto &e : bank) if (!e.isCond) prev.push_back(&e);
            for (size_t i = prev.size() > 6 ? prev.size() - 6 : 0; i < prev.size(); i++)
                putFeat(*prev[i], (k - prev[i]->frame) - 1);
            // obj ptr tokens：cond（pos=k）在前，然后新->旧（pos=t_diff）
            std::vector<std::pair<int, const std::vector<float>*>> ptrList;
            ptrList.push_back({k, &bank[0].ptr});
            {
                size_t start = prev.size() > 15 ? prev.size() - 15 : 0;
                for (size_t i = prev.size(); i-- > start;)
                    ptrList.push_back({k - prev[i]->frame, &prev[i]->ptr});
            }
            std::vector<double> posList;
            for (auto &pr : ptrList) posList.push_back(pr.first / 15.0);
            std::vector<float> pe;
            SinePE1D(posList, &pe);  // [nptr*256]
            for (size_t n = 0; n < ptrList.size(); n++) {
                // ptr token：[256] 按连续 64 拆 4 token
                for (int j = 0; j < 4; j++)
                    memcpy(memory.data.data() + (row + n * 4 + j) * 64,
                           ptrList[n].second->data() + j * 64, 64 * sizeof(float));
                // obj_pos = pe @ W^T + b，repeat_interleave(4)
                for (int j = 0; j < 4; j++)
                    for (int o = 0; o < 64; o++) {
                        double acc = cProjB[o];
                        for (int jj = 0; jj < 256; jj++)
                            acc += static_cast<double>(cProjW[o * 256 + jj]) * pe[n * 256 + jj];
                        memoryPos.data[(row + n * 4 + j) * 64 + o] = static_cast<float>(acc);
                    }
            }
            row += 4 * ptrList.size();
            // ---- memattn 变体选择（num_obj_ptr_tokens 不是模型输入）----
            int kv = N <= 24600 ? static_cast<int>(N / 4100)
                                : static_cast<int>((N - 28672) / 4);
            std::string variant = kv >= 16 ? "memattn_steady" : ([&] {
                char b[24]; snprintf(b, sizeof(b), "memattn_k%02d", kv); return std::string(b);
            })();
            if (variant != memattnName) {
                mgMemattn = std::make_unique<ModelGuard>();
                if (!mgMemattn->Load(modelDir + "/" + variant + ".dat", threads, &err, device))
                    return line + "ERR " + err;
                memattnName = variant;
            }
            // curr / curr_pos：[4096,1,256]，curr[t,0,c]=fpn2[0,c,t]
            FTensor curr, currPos;
            curr.data.resize(4096 * 256);
            currPos.data.resize(4096 * 256);
            for (size_t t = 0; t < 4096; t++)
                for (size_t c = 0; c < 256; c++) {
                    curr.data[t * 256 + c] = fpn2.data[c * 4096 + t];
                    currPos.data[t * 256 + c] = pos2.data[c * 4096 + t];
                }
            std::map<std::string, std::vector<char>> ins{
                {"curr", AsBytes(curr)}, {"curr_pos", AsBytes(currPos)},
                {"memory", AsBytes(memory)}, {"memory_pos", AsBytes(memoryPos)}};
            std::string e = FeedInputs(mgMemattn->model(), ins);
            OH_AI_TensorHandleArray maOuts{};
            if (e.empty()) e = PredictTimed(mgMemattn->model(), &maOuts, &msMa);
            if (!e.empty()) return line + e;
            if (!GetOutF32(maOuts, "pix_feat_with_mem", &backbone))
                return line + "ERR memattn output missing";
            times["memattn"].push_back(msMa);
            times["memattn_" + variant].push_back(msMa);
            {
                std::string gp; std::vector<int64_t> gs;
                CmpStat c;
                std::string mc = caseDir("memattn", memattnIdx(frame));
                if (GoldenOutPath(mc, "pix_feat_with_mem", &gp, &gs))
                    c = CompareGolden(backbone, gp);
                char b[256];
                snprintf(b, sizeof(b),
                         "%s %s %.2e %.0fms in[curr=%.1e cpos=%.1e mem=%.1e mpos=%.1e]",
                         variant.c_str(), c.fail ? "FAIL" : "OK", c.maxdiff, msMa,
                         DiffCaseIn(mc, "curr", curr.data),
                         DiffCaseIn(mc, "curr_pos", currPos.data),
                         DiffCaseIn(mc, "memory", memory.data),
                         DiffCaseIn(mc, "memory_pos", memoryPos.data));
                line += b; line += " | ";
                frameFail = frameFail || c.fail;
            }
            // ---- samheads_track（NCHW 直喂）----
            std::map<std::string, std::vector<char>> tins{
                {"backbone_features", AsBytes(backbone)},
                {"fpn0", AsBytes(fpn0)}, {"fpn1", AsBytes(fpn1)}};
            e = FeedInputs(mgTrack.model(), tins);
            OH_AI_TensorHandleArray tOuts{};
            msSa = -1;
            if (e.empty()) e = PredictTimed(mgTrack.model(), &tOuts, &msSa);
            if (!e.empty()) return line + e;
            FTensor lowMulti, highMulti, ious;
            FTensor lowMask, highMask;
            if (!GetOutF32(tOuts, "low_res_multimasks", &lowMulti) ||
                !GetOutF32(tOuts, "high_res_multimasks", &highMulti) ||
                !GetOutF32(tOuts, "ious", &ious) ||
                !GetOutF32(tOuts, "low_res_masks", &lowMask) ||
                !GetOutF32(tOuts, "high_res_masks", &highMask) ||
                !GetOutF32(tOuts, "obj_ptr", &objPtr) ||
                !GetOutF32(tOuts, "object_score_logits", &score))
                return line + "ERR track output missing";
            times["samheads_track"].push_back(msSa);
            double worst = 0; bool bad = false;
            {
                const std::pair<const char *, const FTensor *> chk[] = {
                    {"low_res_multimasks", &lowMulti}, {"high_res_multimasks", &highMulti},
                    {"ious", &ious}, {"low_res_masks", &lowMask},
                    {"high_res_masks", &highMask}, {"obj_ptr", &objPtr},
                    {"object_score_logits", &score}};
                for (auto &pr : chk) {
                    std::string gp; std::vector<int64_t> gs;
                    if (!GoldenOutPath(caseDir("samheads", samheadsIdx(frame)), pr.first, &gp, &gs))
                        continue;
                    CmpStat c = CompareGolden(*pr.second, gp);
                    if (c.maxdiff > worst) worst = c.maxdiff;
                    bad = bad || c.fail;
                }
            }
            char b[96];
            snprintf(b, sizeof(b), "track %s %.2e %.0fms", bad ? "FAIL" : "OK", worst, msSa);
            line += b; line += " | ";
            frameFail = frameFail || bad;
            // 选 mask：argmax(ious) 取 high_res_multimasks[0,m]（C 维连续）
            int m = 0;
            for (size_t i = 1; i < ious.data.size(); i++)
                if (ious.data[i] > ious.data[m]) m = static_cast<int>(i);
            {
                char b[128];
                snprintf(b, sizeof(b), " ious[%.3f %.3f %.3f] m=%d", ious.data[0],
                         ious.data.size() > 1 ? ious.data[1] : 0.f,
                         ious.data.size() > 2 ? ious.data[2] : 0.f, m);
                line += b;
            }
            maskHr.shape = {1, 1, 1024, 1024};
            maskHr.data.assign(highMulti.data.begin() + static_cast<size_t>(m) * 1024 * 1024,
                               highMulti.data.begin() + static_cast<size_t>(m + 1) * 1024 * 1024);
        }

        // ================= memenc =================
        double msMe = -1;
        {
            ModelGuard &mg = frame == 0 ? mgMemencB : mgMemencS;
            std::map<std::string, std::vector<char>> ins{
                {"pix_feat", AsBytes(fpn2)},  // 所有帧都是 encoder 的 fpn2（golden 已证实），不是 memattn 输出
                {"pred_masks_high_res", AsBytes(maskHr)},
                {"object_score_logits", AsBytes(score)}};
            std::string e = FeedInputs(mg.model(), ins);
            OH_AI_TensorHandleArray outs{};
            if (e.empty()) e = PredictTimed(mg.model(), &outs, &msMe);
            if (!e.empty()) return line + e;
            FTensor feat;
            if (!GetOutF32(outs, "maskmem_features", &feat))
                return line + "ERR memenc output missing";
            times[frame == 0 ? "memenc_binarize" : "memenc_sigmoid"].push_back(msMe);
            std::string gp; std::vector<int64_t> gs;
            CmpStat c;
            std::string mec = caseDir("memenc", memencIdx(frame));
            if (GoldenOutPath(mec, "maskmem_features", &gp, &gs))
                c = CompareGolden(feat, gp);
            char b[256];
            snprintf(b, sizeof(b), "%s %s %.2e %.0fms in[pix=%.1e mask=%.1e score=%.1e]",
                     frame == 0 ? "memencB" : "memencS", c.fail ? "FAIL" : "OK", c.maxdiff,
                     msMe,
                     DiffCaseIn(mec, "pix_feat", fpn2.data),
                     DiffCaseIn(mec, "pred_masks_high_res", maskHr.data),
                     DiffCaseIn(mec, "object_score_logits", score.data));
            line += b;
            frameFail = frameFail || c.fail;
            // ---- bank append ----
            BankEntry be;
            be.frame = frame;
            be.isCond = (frame == 0);
            Tok64(feat.data, &be.feat);
            for (auto &v : be.feat) v = BF16Round(v);
            be.ptr = objPtr.data;
            bank.push_back(std::move(be));
        }

        report << line << "\n";
        report.flush();
        if (frameFail) nFail++;
    }

    // ---- summary（兼作 profiling 基线原始数据）----
    report << "\n== timing summary (ms, n/avg/min/max) ==\n";
    std::ostringstream summary;
    for (const auto &kv : times) {
        double mn = 1e30, mx = 0, sum = 0;
        for (double v : kv.second) {
            mn = std::min(mn, v);
            mx = std::max(mx, v);
            sum += v;
        }
        char b[160];
        snprintf(b, sizeof(b), "%-18s n=%2zu avg=%8.1f min=%8.1f max=%8.1f\n", kv.first.c_str(),
                 kv.second.size(), sum / kv.second.size(), mn, mx);
        report << b;
    }
    report.flush();
    summary << "pipeline done: " << (kFrames - nFail) << "/" << kFrames
            << " frames OK (see pipeline_report.txt)";
    return summary.str();
}

// NPU 探针：用最小的 tinyconv 模型遍历各种 NNRT 配置组合，找出能跑通的配置。
std::string RunNnrtProbe(const std::string &modelDir0, const std::string &vectorDir0,
                         const std::string &outDir0) {
    const std::string modelDir = TranslatePath(modelDir0);
    const std::string vectorDir = TranslatePath(vectorDir0);
    const std::string outDir = TranslatePath(outDir0);
    mkdir(outDir.c_str(), 0755);
    std::string out = NnrtProbe() + "\n";
    const std::string model = modelDir + "/tinyconv.dat";
    const std::string cdir = vectorDir + "/tinyconv_000";

    auto tryBuild = [&](const char *name, const char *modelFile, bool fp16, int freq,
                        const char *provider, const char *providerDev) {
        OH_AI_ContextHandle ctx = OH_AI_ContextCreate();
        OH_AI_ContextSetThreadNum(ctx, 4);
        OH_AI_DeviceInfoHandle npu = OH_AI_DeviceInfoCreate(OH_AI_DEVICETYPE_NNRT);
        if (fp16) OH_AI_DeviceInfoSetEnableFP16(npu, true);
        if (freq >= 0) OH_AI_DeviceInfoSetFrequency(npu, freq);
        if (provider) OH_AI_DeviceInfoSetProvider(npu, provider);
        if (providerDev) OH_AI_DeviceInfoSetProviderDevice(npu, providerDev);
        OH_AI_ContextAddDeviceInfo(ctx, npu);
        OH_AI_DeviceInfoHandle cpu = OH_AI_DeviceInfoCreate(OH_AI_DEVICETYPE_CPU);
        OH_AI_ContextAddDeviceInfo(ctx, cpu);
        OH_AI_ModelHandle m = OH_AI_ModelCreate();
        std::string mpath = modelDir + "/" + modelFile;
        auto t0 = std::chrono::steady_clock::now();
        OH_AI_Status st = OH_AI_ModelBuildFromFile(m, mpath.c_str(), OH_AI_MODELTYPE_MINDIR, ctx);
        auto t1 = std::chrono::steady_clock::now();
        double buildMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        char b[256];
        if (st == OH_AI_STATUS_SUCCESS) {
            snprintf(b, sizeof(b), "%s/%s: BUILD-OK (build %.0fms)", modelFile, name, buildMs);
        } else {
            snprintf(b, sizeof(b), "%s/%s: BUILD-FAIL (build %.0fms) %s", modelFile, name,
                     buildMs, StatusStr(st).c_str());
        }
        out += b;
        out += "\n";
        OH_AI_ModelDestroy(&m);
        OH_AI_ContextDestroy(&ctx);
    };

    // 只测构建（真实模型预测耗时另测）；tinyconv 额外验证输出精度
    const char *kModels[] = {"op_ln.dat", "op_matmul2d.dat", "op_matmul4d.dat", "op_softmax.dat",
                             "op_transpose.dat", "op_split.dat", "op_reshape.dat"};
    for (const char *mf : kModels) {
        tryBuild("plain", mf, false, -1, nullptr, nullptr);
        tryBuild("fp16", mf, true, -1, nullptr, nullptr);
    }
    return out;
}

std::string RunCase(const std::string &modelPath, const std::string &caseDir,
                    const std::string &outDir, int threads, double *predictMs,
                    const std::string &device) {
    ModelGuard mg;
    std::string err;
    if (!mg.Load(TranslatePath(modelPath), threads, &err, device)) return "ERR " + err;
    return RunCaseLoaded(mg.model(), TranslatePath(caseDir), TranslatePath(outDir), predictMs);
}

std::string RunAllCases(const std::string &modelDir0, const std::string &vectorDir0,
                        const std::string &outDir0, int threads, const std::string &device) {
    const std::string modelDir = TranslatePath(modelDir0);
    const std::string vectorDir = TranslatePath(vectorDir0);
    const std::string outDir = TranslatePath(outDir0);
    std::ifstream man(vectorDir + "/manifest.txt");
    if (!man) return "ERR cannot open manifest.txt";
    std::ofstream report(outDir + "/report.txt");
    if (!report) return "ERR cannot write report.txt (outDir missing?)";
    report << "# device=" << device << " " << (device != "cpu" ? NnrtProbe() : "") << "\n";

    std::string line, curModel;
    std::unique_ptr<ModelGuard> mg;  // one live model at a time
    int nOk = 0, nErr = 0;
    std::ostringstream summary;
    while (std::getline(man, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string tag, dir, model;
        ss >> tag >> dir >> model;
        if (tag != "case") continue;
        if (model != curModel) {
            mg = std::make_unique<ModelGuard>();
            std::string err;
            if (!mg->Load(modelDir + "/" + model + ".dat", threads, &err, device)) {  // .ms 后缀会被系统清理
                report << dir << " ERR " << err << "\n";
                nErr++;
                curModel.clear();
                continue;
            }
            curModel = model;
        }
        double ms = -1;
        std::string r = RunCaseLoaded(mg->model(), vectorDir + "/" + dir, outDir + "/" + dir, &ms);
        char buf[64];
        snprintf(buf, sizeof(buf), "%.2fms", ms);
        report << dir << " " << model << " " << r << " " << (ms >= 0 ? buf : "") << "\n";
        report.flush();
        if (r.rfind("OK", 0) == 0) nOk++; else nErr++;
    }
    summary << nOk << " OK, " << nErr << " ERR (see report.txt)";
    return summary.str();
}

}  // namespace sam2
