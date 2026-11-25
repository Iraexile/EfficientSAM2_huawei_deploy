# Prepare MSLite-target ONNX for the two samheads graphs (PC side).
# Pipeline:
#   samheads_cond: export (arith patches) -> cut_tail_cond.py -> onnxsim
#                  -> strip_dropout.py fixup -> samheads_cond_fix.onnx
#   samheads_track: export -> onnxsim -> fixup -> samheads_track_fix.onnx
# Run AFTER scripts/export_onnx.py. All numerics verified by parity_check_mslite.py.
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ONNX = os.path.join(HERE, "..", "outputs", "onnx")
PY = sys.executable


def run(cmd):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=os.path.join(HERE, ".."))


# cond: tail cut (removes no_obj_ptr combine; glue does it) then simplify+fix
run([PY, os.path.join(HERE, "cut_tail_cond.py")])
run([PY, "-m", "onnxsim", os.path.join(ONNX, "samheads_cond_mslite.onnx"),
     os.path.join(ONNX, "samheads_cond_sim.onnx")])
run([PY, os.path.join(HERE, "strip_dropout.py"),
     os.path.join(ONNX, "samheads_cond_sim.onnx"),
     os.path.join(ONNX, "samheads_cond_fix.onnx")])

# track: simplify+fix
run([PY, "-m", "onnxsim", os.path.join(ONNX, "samheads_track.onnx"),
     os.path.join(ONNX, "samheads_track_sim.onnx")])
run([PY, os.path.join(HERE, "strip_dropout.py"),
     os.path.join(ONNX, "samheads_track_sim.onnx"),
     os.path.join(ONNX, "samheads_track_fix.onnx")])
print("PREPARE DONE")
