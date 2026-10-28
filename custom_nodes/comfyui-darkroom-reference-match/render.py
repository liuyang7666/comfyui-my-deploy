"""GPU mask × colour-range rendering with constant native luminance."""
import numpy as np
import torch
import torch.nn.functional as F


def to_device(plan,device):
    def convert(x):
        if isinstance(x,dict): return {k:convert(v) for k,v in x.items()}
        if isinstance(x,np.ndarray): return torch.from_numpy(x).to(device=device,dtype=torch.float32)
        return x
    return convert(plan)


def apply_lab(lab,plan,masks):
    uv = lab[...,1:]/lab[...,:1].clamp(min=.08); delta = torch.zeros_like(uv); total = torch.zeros_like(lab[...,0])
    for name,p in plan['ranges'].items():
        if name not in masks: continue
        alpha = masks[name]
        distance = (((uv.unsqueeze(-2)-p['centers'])/p['width'])**2).sum(-1)
        t = (1-distance/9).clamp(0,1); weight = t*t*(3-2*t)
        c = torch.linalg.vector_norm(uv,dim=-1,keepdim=True)
        center_c = torch.linalg.vector_norm(p['centers'],dim=-1)
        alignment = (uv@p['centers'].T)/(c*center_c).clamp(min=1e-8)
        hue = ((alignment-.5)/.5).clamp(0,1); hue = hue*hue*(3-2*hue)
        hue = torch.where(center_c < .025,torch.ones_like(hue),hue)
        radial = (1-(torch.log((c+.01)/(center_c+.01))/1.0986122886681098)**2).clamp(0,1)
        weight *= hue*radial*radial*(3-2*radial)
        mapped = torch.einsum('kij,...j->...ki',p['matrix'],uv)+p['offset']
        change = (weight.unsqueeze(-1)*(mapped-uv.unsqueeze(-2))).sum(-2)/weight.sum(-1,keepdim=True).clamp(min=1)
        delta += alpha.unsqueeze(-1)*change; total += alpha
    delta /= total.clamp(min=1).unsqueeze(-1)
    ab = lab[...,1:]+(float(plan.get('strength',100))/100)*lab[...,:1].clamp(min=.08)*delta
    return torch.cat([lab[...,:1],ab],dim=-1)


def project_luminance(candidate,original):
    coeff = candidate.new_tensor([.2126,.7152,.0722]); y = original@coeff
    v = candidate-(candidate@coeff).unsqueeze(-1)
    bounds = torch.where(v > 1e-9,(1-y.unsqueeze(-1))/v.clamp(min=1e-9),
                         torch.where(v < -1e-9,-y.unsqueeze(-1)/v.clamp(max=-1e-9),torch.ones_like(v)))
    amount = bounds.amin(-1).clamp(0,1)
    return (y.unsqueeze(-1)+amount.unsqueeze(-1)*v).clamp(0,1)


@torch.inference_mode()
def render(image,plan,masks,gpu,device,chunk_pixels=450000):
    if plan.get('strength',100) == 0: return image.copy()
    params = to_device(plan,device); h,w = image.shape[:2]; rows = max(1,min(h,chunk_pixels//w))
    result = np.empty_like(image,dtype=np.float32); active = {}
    for name,mask in masks.items():
        if name not in params['ranges'] or not np.any(mask): continue
        m = torch.from_numpy(np.ascontiguousarray(mask)).to(device)
        if m.shape != (h,w): m = F.interpolate(m[None,None],size=(h,w),mode='bilinear',align_corners=False)[0,0]
        active[name] = m
    for start in range(0,h,rows):
        stop = min(h,start+rows); rgb = torch.from_numpy(np.ascontiguousarray(image[start:stop])).to(device)
        linear = gpu.srgb_to_linear(rgb); lab = gpu.linear_srgb_to_oklab(linear)
        styled = apply_lab(lab,params,{n:m[start:stop] for n,m in active.items()})
        encoded = gpu.linear_to_srgb(project_luminance(gpu.oklab_to_linear_srgb(styled),linear))
        changed = torch.any(styled[...,1:] != lab[...,1:],dim=-1)
        result[start:stop] = torch.where(changed.unsqueeze(-1),encoded,rgb).cpu().numpy()
    return result
