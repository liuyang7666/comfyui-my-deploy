"""Measure colour attainment and luminance independently on actual output pixels."""
import numpy as np
from .core import GROUPS, MIN_SAMPLES, Y_COEFF, chromaticity, group_samples, palette


def measure(image,result,profile,reference,color):
    a = color.linear_srgb_to_oklab(color.srgb_decode(image))
    b = color.linear_srgb_to_oklab(color.srgb_decode(result))
    y0 = color.srgb_decode(image)@Y_COEFF; y1 = color.srgb_decode(result)@Y_COEFF
    regions = {}
    for name,names in GROUPS.items():
        target = group_samples(reference,names)
        mask = profile['application'][name] >= .98
        ids = np.flatnonzero(mask)
        if len(ids) < MIN_SAMPLES: continue
        ids = ids[np.linspace(0,len(ids)-1,min(len(ids),4096)).astype(int)]
        if len(target) < MIN_SAMPLES:
            regions[name] = {'reference_observed':False,'core_pixels':int(mask.sum())}; continue
        centers,_ = palette(target)
        distances = []
        for lab in (a,b):
            uv = chromaticity(lab.reshape(-1,3)[ids])
            distances.append(np.min(np.linalg.norm(uv[:,None]-centers,axis=-1),axis=1).mean())
        regions[name] = {'reference_observed':True,'core_pixels':int(mask.sum()),
                         'error_before':float(distances[0]),'error_after':float(distances[1]),
                         'error_reduction':float(1-distances[1]/max(distances[0],1e-8))}
    return {'max_abs_linear_Y_change':float(np.max(np.abs(y1-y0))),
            'darkened_pixels_above_1e6':int(np.count_nonzero(y1 < y0-1e-6)),
            'regions':regions}
