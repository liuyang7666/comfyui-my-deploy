"""Colour ranges inside semantic masks, with native luminance preserved."""
from collections import OrderedDict
import hashlib
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

REGIONS = ('skin', 'sky', 'vegetation', 'grass', 'trees', 'water', 'architecture')
GROUPS = {'skin': ('skin',), 'sky': ('sky',), 'plants': ('grass', 'trees', 'vegetation'),
          'water': ('water',), 'architecture': ('architecture',)}
APPLICATION_MASKS = tuple(GROUPS) + ('person',)
Y_COEFF = np.array([.2126,.7152,.0722],np.float32)
EDGE, MIN_SAMPLES, MAX_SAMPLES = 1024,96,16000
ALGORITHM = 'masked-colour-ranges-native-luminance-v1'
_PLAN_CACHE = OrderedDict()


def fingerprint(image):
    return hashlib.sha256(memoryview(np.ascontiguousarray(image)).cast('B')).hexdigest()


def analysis_image(image,edge=EDGE):
    h,w = image.shape[:2]; scale = min(1.,edge/max(h,w))
    return cv2.resize(image,(max(1,round(w*scale)),max(1,round(h*scale))),interpolation=cv2.INTER_AREA) if scale < 1 else image.copy()


def inner_distance(mask):
    return distance_transform_edt(np.pad(mask,1))[1:-1,1:-1].astype(np.float32)


def prepare_masks(raw,shape,erosion=4.):
    masks = {}
    for name in REGIONS:
        m = np.asarray(raw.get(name,np.zeros(shape,np.float32)),np.float32)
        if m.ndim != 2 or not np.isfinite(m).all(): raise ValueError(name+': invalid mask')
        if m.shape != shape: m = cv2.resize(m,(shape[1],shape[0]),interpolation=cv2.INTER_LINEAR)
        masks[name] = np.clip(m,0,1)
    stats,application = {},{}
    radius = max(1.,erosion*max(shape)/EDGE)
    for name,m in masks.items():
        other = np.maximum.reduce([v for n,v in masks.items() if n != name])
        stats[name] = inner_distance((m >= .9)&(other <= .1)) > radius
        application[name] = m*(other < m)
    return stats,application


def sample(lab,mask=None,limit=MAX_SAMPLES):
    flat = lab.reshape(-1,3)
    ids = np.flatnonzero(mask.reshape(-1)) if mask is not None else np.arange(len(flat))
    if len(ids) > limit: ids = np.random.default_rng(1729).choice(ids,limit,replace=False)
    pixels = flat[ids]
    if len(pixels) >= MIN_SAMPLES:
        low,high = np.quantile(pixels[:,0],[.015,.985])
        pixels = pixels[(pixels[:,0] >= low)&(pixels[:,0] <= high)]
    return pixels.astype(np.float32)


def robust_center(pixels):
    center = np.median(pixels,axis=0)
    distance = np.linalg.norm(pixels[:,1:]-center[1:],axis=-1)
    mad = np.median(np.abs(distance-np.median(distance)))
    keep = distance <= np.median(distance)+max(3*1.4826*mad,.008)
    return np.median(pixels[keep],axis=0)


def describe_samples(pixels):
    if not len(pixels): return {'count':0}
    cuts = np.quantile(pixels[:,0],[1/3,2/3]); bands = np.searchsorted(cuts,pixels[:,0],side='right')
    centers = [robust_center(pixels[bands == i] if np.count_nonzero(bands == i) >= 24 else pixels) for i in range(3)]
    return {'count':len(pixels),'centers':np.asarray(centers).tolist(),
            'quantiles':np.quantile(pixels[:,0],[.025,.5,.975]).tolist()}


def chromaticity(lab):
    return lab[...,1:]/np.maximum(lab[...,:1],.08)


def palette(samples):
    """Keep colour modes; luminance is not a matching dimension."""
    uv = chromaticity(samples)
    if len(uv) > 4096: uv = uv[np.linspace(0,len(uv)-1,4096).astype(int)]
    center = np.median(uv,axis=0)
    distance = np.linalg.norm(uv-center,axis=-1)
    uv = uv[distance <= np.quantile(distance,.99)+1e-6]
    centers = [np.median(uv,axis=0)]
    for _ in range(5):
        distance = np.min(np.sum((uv[:,None]-np.asarray(centers)[None])**2,axis=-1),axis=1)
        if float(distance.max()) < .025**2: break
        centers.append(uv[np.argmax(distance)])
    centers = np.asarray(centers)
    for _ in range(12):
        labels = np.argmin(np.sum((uv[:,None]-centers[None])**2,axis=-1),axis=1)
        updated = np.array([np.mean(uv[labels == i],axis=0) if np.any(labels == i) else c for i,c in enumerate(centers)])
        if np.max(np.abs(updated-centers)) < 1e-5:
            centers = updated; break
        centers = updated
    keep = np.array([np.count_nonzero(labels == i) >= max(24,len(uv)*.015) for i in range(len(centers))])
    if not np.any(keep): keep[np.argmax(np.bincount(labels,minlength=len(centers)))] = True
    centers = centers[keep]
    labels = np.argmin(np.sum((uv[:,None]-centers[None])**2,axis=-1),axis=1)
    widths = [np.clip(np.quantile(np.abs(uv[labels == i]-c),.98,axis=0)*1.6+.02,.035,.12) for i,c in enumerate(centers)]
    return centers.astype(np.float32),np.asarray(widths,np.float32)


def group_samples(profile,names):
    parts = []
    for name in names:
        pixels = profile['samples'][name]
        if len(pixels) >= MIN_SAMPLES:
            parts.append(pixels[np.linspace(0,len(pixels)-1,min(4096,len(pixels))).astype(int)])
    return np.concatenate(parts) if parts else np.empty((0,3),np.float32)


def neutral_cast(source,reference):
    centers = []
    for p in (source,reference):
        pixels = p.get('neutral_person_samples',np.empty((0,3),np.float32))
        if len(pixels) < MIN_SAMPLES: return np.zeros(2,np.float32),'no_reliable_reference_colour'
        uv = chromaticity(pixels); c = np.linalg.norm(uv,axis=-1)
        centers.append(np.median(uv[c <= np.quantile(c,.25)],axis=0))
    delta = centers[1]-centers[0]
    delta *= min(1.,.03/max(float(np.linalg.norm(delta)),1e-8))
    return delta.astype(np.float32),'paired_person_neutrals_only'


def _profile_key(profile):
    if 'cache_key' in profile: return profile['cache_key']
    h = hashlib.sha256()
    for name,pixels in profile['samples'].items():
        h.update(name.encode()); h.update(np.ascontiguousarray(pixels).tobytes())
    h.update(np.ascontiguousarray(profile.get('neutral_person_samples',[])).tobytes())
    return h.hexdigest()


def solve(source,reference,strength=100):
    if not np.isfinite(strength) or not 0 <= strength <= 100: raise ValueError('strength must be within 0–100')
    key = (_profile_key(source),_profile_key(reference)); hit = key in _PLAN_CACHE
    if hit: full = _PLAN_CACHE.pop(key)
    else:
        ranges,regions,support = {},{},{}
        cast,evidence = neutral_cast(source,reference)
        for name,names in GROUPS.items():
            a,b = group_samples(source,names),group_samples(reference,names)
            support[name] = [len(a),len(b)]
            if len(a) < MIN_SAMPLES: continue
            centers,width = palette(a)
            if len(b) >= MIN_SAMPLES:
                target,_ = palette(b); chosen = []
                for c in centers:
                    candidates = target
                    if np.linalg.norm(c) < .025:
                        candidates = target[np.linalg.norm(target,axis=-1) < .04]
                    chosen.append(candidates[np.argmin(np.sum((candidates-c)**2,axis=1))] if len(candidates) else c)
                chosen = np.asarray(chosen)
                angles = np.arctan2(chosen[:,1],chosen[:,0])-np.arctan2(centers[:,1],centers[:,0])
                gain = np.clip(np.linalg.norm(chosen,axis=-1)/np.maximum(np.linalg.norm(centers,axis=-1),.01),0,2.)
                cosine,sine = np.cos(angles)*gain,np.sin(angles)*gain
                matrix = np.stack([cosine,-sine,sine,cosine],axis=-1).reshape(-1,2,2)
                offset = np.zeros_like(centers); mode = 'reference_colour_ranges'
            else:
                matrix = np.broadcast_to(np.eye(2),(*centers.shape[:1],2,2)).copy()
                offset = np.broadcast_to(cast,centers.shape).copy(); mode = evidence
            ranges[name] = {'centers':centers,'width':width,'matrix':matrix.astype(np.float32),'offset':offset.astype(np.float32)}
            regions[name] = {'mode':mode,'source_members':list(names),'source_samples':len(a),
                             'reference_samples':len(b),'colour_ranges':len(centers)}
        full = {'ranges':ranges,'regions':regions,'support':support,'order':['mask','colour_range','native_luminance'],
                'brightness_policy':'preserve_per_pixel_linear_Y','quality':{'scope':'masked colour targets; no exposure matching'}}
    _PLAN_CACHE[key] = full
    while len(_PLAN_CACHE) > 16: _PLAN_CACHE.popitem(last=False)
    return {**full,'strength':strength,'plan_cache_hit':hit}


def apply_lab(lab,plan,masks):
    uv = chromaticity(lab); delta = np.zeros_like(uv); total = np.zeros(lab.shape[:-1],np.float32)
    for name,p in plan['ranges'].items():
        alpha = np.asarray(masks.get(name,0),np.float32)
        if not np.any(alpha): continue
        distance = np.sum(((uv[...,None,:]-p['centers'])/p['width'])**2,axis=-1)
        t = np.clip(1-distance/9,0,1); weight = t*t*(3-2*t)
        c = np.linalg.norm(uv,axis=-1,keepdims=True)
        center_c = np.linalg.norm(p['centers'],axis=-1)
        alignment = (uv@p['centers'].T)/np.maximum(c*center_c,1e-8)
        hue = np.clip((alignment-.5)/.5,0,1)
        hue = hue*hue*(3-2*hue)
        hue = np.where(center_c < .025,1,hue)
        radial = np.clip(1-(np.log((c+.01)/(center_c+.01))/np.log(3.))**2,0,1)
        weight *= hue*radial*radial*(3-2*radial)
        mapped = np.einsum('kij,...j->...ki',p['matrix'],uv)+p['offset']
        change = np.sum(weight[...,None]*(mapped-uv[...,None,:]),axis=-2)/np.maximum(1,weight.sum(axis=-1,keepdims=True))
        delta += alpha[...,None]*change; total += alpha
    delta /= np.maximum(1,total)[...,None]
    result = lab.copy()
    result[...,1:] += (float(plan.get('strength',100))/100)*np.maximum(lab[...,:1],.08)*delta
    return result


def project_luminance(candidate,original):
    """Project into RGB gamut around ORIGINAL linear luminance."""
    y = original@Y_COEFF; v = candidate-(candidate@Y_COEFF)[...,None]
    bounds = np.where(v > 1e-9,(1-y[...,None])/np.maximum(v,1e-9),
                      np.where(v < -1e-9,-y[...,None]/np.minimum(v,-1e-9),1.))
    amount = np.clip(np.minimum(1,bounds.min(axis=-1)),0,1)
    return np.clip(y[...,None]+amount[...,None]*v,0,1).astype(np.float32)


def render_numpy(image,plan,masks,color,tile_rows=128):
    if plan.get('strength',100) == 0: return image.copy()
    h,w = image.shape[:2]
    resized = {n:cv2.resize(np.asarray(m,np.float32),(w,h),interpolation=cv2.INTER_LINEAR)
               if np.shape(m) != (h,w) else m for n,m in masks.items() if n in plan['ranges']}
    out = np.empty_like(image)
    for start in range(0,h,tile_rows):
        stop = min(h,start+tile_rows); rgb = image[start:stop]
        linear = color.srgb_decode(rgb); lab = color.linear_srgb_to_oklab(linear)
        styled = apply_lab(lab,plan,{n:m[start:stop] for n,m in resized.items()})
        changed = np.any(styled[...,1:] != lab[...,1:],axis=-1)
        encoded = color.srgb_encode(project_luminance(color.oklab_to_linear_srgb(styled),linear))
        out[start:stop] = np.where(changed[...,None],encoded,rgb)
    return out
