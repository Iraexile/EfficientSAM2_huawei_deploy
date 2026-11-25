# Export the 4 SAM2 modules to static-shape fp32 ONNX graphs.
# Variants: memattn k=1..15 + steady (static N, exact numerics); memenc binarize/sigmoid;
#           samheads cond (box, single-mask) / track (no points, multimask).
import os
import numpy as np
import torch
import torch.nn as nn

REPO = os.path.join(os.path.dirname(__file__), "..", "vendor", "Efficient-SAM2")  #源码位置
CKPT = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "sam2.1_hiera_tiny.pt")  #模型权责
GOLD = os.path.join(os.path.dirname(__file__), "..", "golden")  #用于导出的参考数据
ONNX_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "onnx") #最终onnx输出目录
os.makedirs(ONNX_DIR, exist_ok=True)

from sam2.build_sam import build_sam2_video_predictor

# --------------------------------------------------------------------------
# RoPE real-valued rewrite: the stock implementation uses complex tensors
# (torch.polar / view_as_complex), which the ONNX tracer cannot export.
# We patch compute_axial_cis / apply_rotary_enc with mathematically identical
# real-number versions BEFORE the model is built (RoPEAttention captures the
# function at __init__ via functools.partial).
import sam2.modeling.sam.transformer as _tfm


def compute_axial_cis_real(dim, end_x, end_y, theta=10000.0):
    """Real-valued twin of compute_axial_cis: returns [S, dim//2, 2] with
    [..., 0]=cos(angle), [..., 1]=sin(angle) instead of a complex tensor."""
    from sam2.modeling.position_encoding import init_t_xy

    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    t_x, t_y = init_t_xy(end_x, end_y)
    ang = torch.cat([torch.outer(t_x, freqs_x), torch.outer(t_y, freqs_y)], dim=-1)
    return torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)


def apply_rotary_enc_real(xq, xk, freqs_cis, repeat_freqs_k=False):
    """Real-valued twin of apply_rotary_enc. (x0+ix1)(cos+isin) expanded."""
    xq_r = xq.float().reshape(*xq.shape[:-1], -1, 2)
    fc = freqs_cis.view([1] * (xq_r.ndim - 3) + list(freqs_cis.shape))
    cos, sin = fc[..., 0], fc[..., 1]
    x0, x1 = xq_r[..., 0], xq_r[..., 1]
    xq_out = torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1).flatten(-2)
    if xk.shape[-2] == 0:
        return xq_out.type_as(xq).to(xq.device), xk
    xk_r = xk.float().reshape(*xk.shape[:-1], -1, 2)
    if repeat_freqs_k:
        r = xk_r.shape[-3] // xq_r.shape[-3]
        cos = cos.repeat(*([1] * (cos.ndim - 2)), r, 1)
        sin = sin.repeat(*([1] * (sin.ndim - 2)), r, 1)
    k0, k1 = xk_r[..., 0], xk_r[..., 1]
    xk_out = torch.stack([k0 * cos - k1 * sin, k0 * sin + k1 * cos], dim=-1).flatten(-2)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


_tfm.compute_axial_cis = compute_axial_cis_real
_tfm.apply_rotary_enc = apply_rotary_enc_real

# Under JIT tracing, tensor.shape[i] arrives as a 0-dim CPU tensor, and
# torch.repeat_interleave then crashes CUDA index_select. Patch with an
# equivalent expand+flatten implementation (exact for scalar repeats).
_orig_repeat_interleave = torch.repeat_interleave


def _repeat_interleave_safe(x, repeats, dim=None, **kw):
    if torch.is_tensor(x) and dim is not None and int(repeats) >= 0:
        r = int(repeats)
        if r == 1:
            return x
        shape = list(x.shape)
        x2 = x.unsqueeze(dim + 1).expand(*shape[: dim + 1], r, *shape[dim + 1 :])
        return x2.flatten(dim, dim + 1)
    return _orig_repeat_interleave(x, repeats, dim=dim, **kw)


torch.repeat_interleave = _repeat_interleave_safe

# MindSpore Lite converter does not support ONNX If nodes. The Hiera trunk's
# _get_pos_embed exports bicubic-interpolate scale guards as If; for the static
# 1024x1024 input the result is a constant anyway, so compute it once and cache
# it (numerically identical: same weights, same device, deterministic op).
from sam2.modeling.backbones.hieradet import Hiera as _Hiera

_orig_get_pos_embed = _Hiera._get_pos_embed
_pos_cache = {}


def _get_pos_embed_cached(self, hw):
    key = (int(hw[0]), int(hw[1]))
    if key not in _pos_cache:
        _pos_cache[key] = _orig_get_pos_embed(self, key)
    return _pos_cache[key]


_Hiera._get_pos_embed = _get_pos_embed_cached

# NNACL's Where kernel errors on the prompt encoder's bool-mask broadcast
# ([B,N,1] cond vs [B,N,256] branches). Rewrite _embed_points with 0/1-mask
# arithmetic, which is bit-exact for one-hot labels:
#   where(l==-1, not_a, pe)  ==  pe*(1-m) + not_a*m
#   where(l==k, pe+w_k, pe)  ==  pe + m_k*w_k
from sam2.modeling.sam.prompt_encoder import PromptEncoder as _PE


def _embed_points_arith(self, points, labels, pad):
    points = points + 0.5
    if pad:
        padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
        padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
        points = torch.cat([points, padding_point], dim=1)
        labels = torch.cat([labels, padding_label], dim=1)
    pe = self.pe_layer.forward_with_coords(points, self.input_image_size)
    m_neg = (labels == -1).unsqueeze(-1).to(pe.dtype)
    pe = pe * (1 - m_neg) + (torch.zeros_like(pe) + self.not_a_point_embed.weight) * m_neg
    for k in range(4):
        m_k = (labels == k).unsqueeze(-1).to(pe.dtype)
        pe = pe + m_k * self.point_embeddings[k].weight
    return pe


_PE._embed_points = _embed_points_arith

# PositionEmbeddingRandom.forward_with_coords uses in-place indexed assignment
# (coords[:, :, 0] = ...), which exports to ScatterND; MSLite cannot infer its
# shape. Replace with a broadcast divide: identical fp32 math, no ScatterND.
from sam2.modeling.position_encoding import PositionEmbeddingRandom as _PER


def _forward_with_coords_static(self, coords_input, image_size):
    denom = torch.tensor([image_size[1], image_size[0]],
                         dtype=coords_input.dtype, device=coords_input.device)
    return self._pe_encoding((coords_input / denom).to(torch.float))


_PER.forward_with_coords = _forward_with_coords_static

DEV = "cuda"
predictor = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_t.yaml", CKPT, device=DEV)
model = predictor  # SAM2Base
model.eval()
model.init_memory_info(enable_MeP_info=False)
model.Mem_Frame_Prune = False


def load_npz(name):
    return np.load(os.path.join(GOLD, name))


def t(a):
    return torch.from_numpy(a).to(DEV).float()


# ---------------------------------------------------------------- wrappers
class ImageEncoderW(nn.Module):
    """Mirror SAM2Base.forward_image: image_encoder + conv_s0/conv_s1 on fpn[0:2]."""

    def __init__(self, enc, conv_s0, conv_s1):
        super().__init__()
        self.enc = enc
        self.conv_s0 = conv_s0
        self.conv_s1 = conv_s1

    def forward(self, image):
        out = self.enc(image)
        fpn = list(out["backbone_fpn"])
        fpn[0] = self.conv_s0(fpn[0])
        fpn[1] = self.conv_s1(fpn[1])
        return (*fpn, *out["vision_pos_enc"])


class MemAttnW(nn.Module):
    def __init__(self, ma, num_obj_ptr_tokens):
        super().__init__()
        self.ma = ma
        self.n_ptr = num_obj_ptr_tokens

    def forward(self, curr, curr_pos, memory, memory_pos):
        out = self.ma(curr=curr, curr_pos=curr_pos, memory=memory, memory_pos=memory_pos,
                      num_obj_ptr_tokens=self.n_ptr)
        return out.permute(1, 2, 0).view(1, 256, 64, 64)


class MemEncW(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.menc = m.memory_encoder
        self.scale = m.sigmoid_scale_for_mem_enc
        self.bias = m.sigmoid_bias_for_mem_enc
        self.no_obj_embed_spatial = m.no_obj_embed_spatial

    def forward(self, pix_feat, pred_masks_high_res, object_score_logits):
        raise NotImplementedError


class MemEncSigmoidW(MemEncW):
    def forward(self, pix_feat, pred_masks_high_res, object_score_logits):
        mask_for_mem = torch.sigmoid(pred_masks_high_res) * self.scale + self.bias
        out = self.menc(pix_feat, mask_for_mem, skip_mask_sigmoid=True)
        feats = out["vision_features"]
        is_obj = (object_score_logits > 0).float()
        feats = feats + (1 - is_obj[..., None, None]) * self.no_obj_embed_spatial[..., None, None].expand(feats.shape)
        pos = out["vision_pos_enc"]
        return feats, (pos[0] if isinstance(pos, list) else pos)


class MemEncBinarizeW(MemEncW):
    def forward(self, pix_feat, pred_masks_high_res, object_score_logits):
        mask_for_mem = (pred_masks_high_res > 0).float() * self.scale + self.bias
        out = self.menc(pix_feat, mask_for_mem, skip_mask_sigmoid=True)
        feats = out["vision_features"]
        is_obj = (object_score_logits > 0).float()
        feats = feats + (1 - is_obj[..., None, None]) * self.no_obj_embed_spatial[..., None, None].expand(feats.shape)
        pos = out["vision_pos_enc"]
        return feats, (pos[0] if isinstance(pos, list) else pos)


class SamHeadsCondW(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, backbone_features, fpn0, fpn1, point_coords, point_labels):
        outs = self.m._forward_sam_heads(
            backbone_features=backbone_features,
            point_inputs={"point_coords": point_coords, "point_labels": point_labels},
            high_res_features=[fpn0, fpn1],
            multimask_output=False,
        )
        return outs


class SamHeadsTrackW(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, backbone_features, fpn0, fpn1):
        outs = self.m._forward_sam_heads(
            backbone_features=backbone_features,
            point_inputs=None,
            high_res_features=[fpn0, fpn1],
            multimask_output=True,
        )
        return outs


# ---------------------------------------------------------------- export
import sys
ONLY = set(sys.argv[1:])  # optional: export only these graph names


def export(name, module, args, input_names, output_names):
    if ONLY and name not in ONLY:
        print("skipped", name)
        return
    path = os.path.join(ONNX_DIR, name + ".onnx")
    torch.onnx.export(
        module, args, path,
        input_names=input_names, output_names=output_names,
        opset_version=17, dynamo=False, do_constant_folding=False,  # folding hits a ComplexFloat bug here
    )
    print("exported", name, f"({os.path.getsize(path)/1e6:.1f} MB)")


with torch.no_grad():
    # 1) image encoder
    d = load_npz("image_encoder_000.npz")
    img_enc_w = ImageEncoderW(model.image_encoder, model.sam_mask_decoder.conv_s0,
                              model.sam_mask_decoder.conv_s1)
    _ = img_enc_w(t(d["in_image"]))  # eager warm-up: populates the pos-embed cache
    export("image_encoder", img_enc_w,
           (t(d["in_image"]),),
           ["image"], ["fpn0", "fpn1", "fpn2", "pos0", "pos1", "pos2"])

    # 2) memory attention: variants k=1..15 + steady(k>=16)
    #    frame k: m = 1+min(k-1,6) memory frames, p = 1+min(k-1,15) obj ptrs (4 tokens each)
    ref = load_npz("memattn_004.npz")  # frame-1 call for realistic curr/curr_pos
    for k in list(range(1, 16)) + ["steady"]:
        kk = 16 if k == "steady" else k
        m = 1 + min(kk - 1, 6)
        p = 1 + min(kk - 1, 15)
        N = 4096 * m + 4 * p
        args = (t(ref["in_curr"]), t(ref["in_curr_pos"]),
                torch.zeros(N, 1, 64, device=DEV), torch.zeros(N, 1, 64, device=DEV))
        name = f"memattn_k{kk:02d}" if k != "steady" else "memattn_steady"
        export(name, MemAttnW(model.memory_attention, 4 * p), args,
               ["curr", "curr_pos", "memory", "memory_pos"], ["pix_feat_with_mem"])

    # 3) memory encoder: binarize (cond frame w/ point prompt) + sigmoid (tracking)
    d = load_npz("memenc_002.npz")
    export("memenc_binarize", MemEncBinarizeW(model),
           (t(d["in_pix_feat"]), t(d["in_pred_masks_high_res"]), t(d["in_object_score_logits"])),
           ["pix_feat", "pred_masks_high_res", "object_score_logits"],
           ["maskmem_features", "maskmem_pos_enc"])
    d = load_npz("memenc_006.npz")
    export("memenc_sigmoid", MemEncSigmoidW(model),
           (t(d["in_pix_feat"]), t(d["in_pred_masks_high_res"]), t(d["in_object_score_logits"])),
           ["pix_feat", "pred_masks_high_res", "object_score_logits"],
           ["maskmem_features", "maskmem_pos_enc"])

    # 4) sam heads: cond (box prompt, M=1) + track (no prompt, M=3)
    d = load_npz("samheads_001.npz")
    export("samheads_cond", SamHeadsCondW(model),
           (t(d["in_backbone_features"]), t(d["in_fpn0"]), t(d["in_fpn1"]),
            t(d["in_point_coords"]), torch.from_numpy(d["in_point_labels"]).to(DEV).int()),
           ["backbone_features", "fpn0", "fpn1", "point_coords", "point_labels"],
           ["low_res_multimasks", "high_res_multimasks", "ious", "low_res_masks",
            "high_res_masks", "obj_ptr", "object_score_logits"])
    d = load_npz("samheads_005.npz")
    export("samheads_track", SamHeadsTrackW(model),
           (t(d["in_backbone_features"]), t(d["in_fpn0"]), t(d["in_fpn1"])),
           ["backbone_features", "fpn0", "fpn1"],
           ["low_res_multimasks", "high_res_multimasks", "ious", "low_res_masks",
            "high_res_masks", "obj_ptr", "object_score_logits"])

print("ALL EXPORTS DONE")
