// MindSpore Lite (system NDK) per-graph parity/benchmark harness.
// Runs one golden "case" (flat manifest + raw .bin tensors) against a .ms model,
// writes outputs as raw .bin + a per-case report line.
#ifndef EFFICIENTSAM2_MSLITE_RUNNER_H
#define EFFICIENTSAM2_MSLITE_RUNNER_H

#include <string>

namespace sam2 {

// Run a single case: modelPath=/abs/x.ms, caseDir contains case.txt + *.bin,
// outDir gets <out_name>.bin written. Returns a human-readable status string
// (first token OK or ERR). predict_ms >= 0 on success.
std::string RunCase(const std::string &modelPath, const std::string &caseDir,
                    const std::string &outDir, int threads, double *predictMs,
                    const std::string &device = "cpu");

// Run all cases listed in vectorDir/manifest.txt (lines: case <dir> <model>).
// Models are cached while consecutive cases share the same model.
// Appends one line per case to outDir/report.txt.
std::string RunAllCases(const std::string &modelDir, const std::string &vectorDir,
                        const std::string &outDir, int threads,
                        const std::string &device = "cpu");

// Full 18-frame video pipeline: per frame encoder -> (memattn) -> samheads -> memenc,
// with the memory-bank glue done in C++ (spec: pc/scripts/glue_dryrun.py).
// Compares every graph's outputs against the per-case goldens and times each graph.
// Writes outDir/pipeline_report.txt (per-frame lines + timing summary).
std::string RunPipeline(const std::string &modelDir, const std::string &vectorDir,
                        const std::string &outDir, int threads,
                        const std::string &device = "cpu");

// NPU 探针：用 tinyconv 遍历 NNRT 配置组合，返回多行结果。
std::string RunNnrtProbe(const std::string &modelDir, const std::string &vectorDir,
                         const std::string &outDir);

}  // namespace sam2

#endif  // EFFICIENTSAM2_MSLITE_RUNNER_H
