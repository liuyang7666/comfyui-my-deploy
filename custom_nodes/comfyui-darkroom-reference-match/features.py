"""Cache inward statistics and all application masks once per source image."""
import hashlib,json,logging,time,uuid,zipfile
from pathlib import Path
import cv2
import numpy as np
from .core import (ALGORITHM,APPLICATION_MASKS,GROUPS,REGIONS,analysis_image,
                   describe_samples,fingerprint,inner_distance,prepare_masks,sample)
from .masks import MODEL,SEGMENTATION_VERSION,segment_images

CACHE_VERSION = ALGORITHM+':'+SEGMENTATION_VERSION


def refine_protection(image,masks):
    guide = image@np.array([.2126,.7152,.0722],np.float32)
    def box(x): return cv2.boxFilter(x,-1,(5,5),borderType=cv2.BORDER_REFLECT)
    mean = box(guide); variance = np.maximum(box(guide*guide)-mean*mean,0); result = {}
    for name,mask in masks.items():
        p = box(mask); a = (box(guide*mask)-mean*p)/(variance+.002); b = p-a*mean
        guided = np.clip(box(a)*guide+box(b),0,1)
        support = mask > 0
        # No grown support; inward statistics erosion never becomes application.
        result[name] = (guided*np.clip(inner_distance(support)/2,0,1)*support).astype(np.float32)
    return result


def make_profile(image,raw,color):
    stats,_ = prepare_masks(raw,image.shape[:2]); lab = color.linear_srgb_to_oklab(color.srgb_decode(image))
    samples = {'global':sample(lab)}
    samples.update({n:sample(lab,stats[n]) for n in REGIONS})
    person = np.clip(np.asarray(raw.get('person',np.zeros(image.shape[:2])),np.float32),0,1)
    application = {}
    for name,names in GROUPS.items():
        union = np.maximum.reduce([np.asarray(raw.get(n,np.zeros(image.shape[:2])),np.float32) for n in names])
        if name != 'skin': union = union*(1-person)
        application[name] = np.clip(union,0,1)
    application = refine_protection(image,application); application['person'] = person
    neutral = (person >= .98)&(np.linalg.norm(lab[...,1:],axis=-1) < .05)&(lab[...,0] > .6)
    return {'application':application,'samples':samples,'descriptions':{n:describe_samples(p) for n,p in samples.items()},
            'statistics_pixels':{n:int(m.sum()) for n,m in stats.items()},'neutral_person_samples':sample(lab,neutral,limit=4000)}


def cache_key(image,model_signature,color_signature):
    config = json.dumps([CACHE_VERSION,list(image.shape),model_signature,color_signature],sort_keys=True)
    return hashlib.sha256(config.encode()+fingerprint(image).encode()).hexdigest()


def load_profile(path,key):
    try:
        with np.load(path,allow_pickle=False) as z:
            meta = json.loads(str(z['metadata']))
            if meta['version'] != CACHE_VERSION or meta['key'] != key: return None
            p = {'application':{n:z['mask_'+n].astype(np.float32) for n in APPLICATION_MASKS},
                 'samples':{n:z['samples_'+n].astype(np.float32) for n in ('global',)+REGIONS},'cache_key':key,
                 'descriptions':meta['descriptions'],'statistics_pixels':meta['statistics_pixels'],
                 'neutral_person_samples':z['neutral_person_samples'].astype(np.float32)}
            if not all(np.isfinite(x).all() for x in list(p['application'].values())+list(p['samples'].values())): return None
            return p
    except (OSError,ValueError,KeyError,EOFError,zipfile.BadZipFile):
        logging.warning('[Darkroom reference] Invalid mask/statistics cache: %s',path.name)
        return None


def save_profile(path,key,p):
    arrays = {'mask_'+n:m for n,m in p['application'].items()}
    arrays.update({'samples_'+n:v for n,v in p['samples'].items()}); arrays['neutral_person_samples'] = p['neutral_person_samples']
    arrays['metadata'] = np.array(json.dumps({'key':key,'version':CACHE_VERSION,'descriptions':p['descriptions'],
                                             'statistics_pixels':p['statistics_pixels']}))
    temporary = path.with_suffix('.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('wb') as stream: np.savez_compressed(stream,**arrays)
    temporary.replace(path)


def analyze_images(images,color,directory,model_signature,color_signature,detect=segment_images):
    directory = Path(directory); directory.mkdir(parents=True,exist_ok=True); started = time.monotonic()
    previews = [analysis_image(im) for im in images]
    keys = [cache_key(im,model_signature,color_signature) for im in previews]; profiles,missing = {},{}
    for key,im in zip(keys,previews):
        if key in profiles or key in missing: continue
        path = directory/(key+'.npz'); cached = load_profile(path,key) if path.is_file() else None
        if cached is None: missing[key] = im
        else: profiles[key] = cached
    if missing:
        raw = detect(list(missing.values()))
        if len(raw) != len(missing): raise RuntimeError('SAM3结果数量与输入不一致。')
        for (key,im),masks in zip(missing.items(),raw):
            profiles[key] = make_profile(im,masks,color); profiles[key]['cache_key'] = key
            save_profile(directory/(key+'.npz'),key,profiles[key])
    return [profiles[k] for k in keys],{'unique_images':len(set(keys)),'cache_hits':len(set(keys))-len(missing),
        'image_encoder_calls':len(missing),'statistics_rebuilt':len(missing),'seconds':round(time.monotonic()-started,3)}


def local_model_signature():
    import folder_paths
    path = folder_paths.get_full_path('sam3',MODEL+'.safetensors')
    if path is None: raise RuntimeError('未找到本地SAM3模型。')
    stat = Path(path).stat(); return [str(path),stat.st_size,stat.st_mtime_ns]
