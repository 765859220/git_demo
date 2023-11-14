import torch
import torch.nn as nn

import sys
sys.path.append("/vepfs/home/wangxixi/sd_benchmark/3rdparty/videodiffusion")
from libs.models.attention import BasicTransformerBlock_V2


@torch.no_grad()
def smooth_ln_fcs(ln, fcs, act_scales, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]
    assert isinstance(ln, nn.LayerNorm)
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features == act_scales.numel()

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    weight_scales = torch.cat([fc.weight.abs().max(
        dim=0, keepdim=True)[0] for fc in fcs], dim=0)
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)

    scales = (act_scales.pow(alpha) / weight_scales.pow(1-alpha)
              ).clamp(min=1e-5).to(device).to(dtype)

    ln.weight.div_(scales)
    ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))


@torch.no_grad()
def smooth_lm(model, scales, alpha=0.5):
    for name, module in model.named_modules():
        if isinstance(module, BasicTransformerBlock_V2):
            attn_ln1 = module.norm1
            qkv1 = [module.attn1.to_q,
                   module.attn1.to_k, module.attn1.to_v]
            qkv_input_scales1 = scales[name + '.attn1.to_q']
            smooth_ln_fcs(attn_ln1, qkv1, qkv_input_scales1, alpha)

            attn_ln2 = module.norm2
            to_q = module.attn2.to_q
            q_input_scales = scales[name + '.attn2.to_q']
            smooth_ln_fcs(attn_ln2, to_q, q_input_scales, alpha)

            # ffn_ln = module.norm3
            # fc1 = module.fc1
            # fc1_input_scales = scales[name + '.fc1']
            # smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha)
        # elif isinstance(module, BloomBlock):
        #     attn_ln = module.input_layernorm
        #     qkv = module.self_attention.query_key_value
        #     qkv_input_scales = scales[name + '.self_attention.query_key_value']
        #     smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)

        #     ffn_ln = module.post_attention_layernorm
        #     fc1 = module.mlp.dense_h_to_4h
        #     fc1_input_scales = scales[name + '.mlp.dense_h_to_4h']
        #     smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha)
