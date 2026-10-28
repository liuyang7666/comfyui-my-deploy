"""Skin-only adaptive colour blemish detection. No face models or landmarks."""
import cv2
import numpy as np
import torch


def _detect_spots(score, skin, sensitivity, max_spot_size, min_spot_size=5):
    # Accumulate accepted components from fixed contrast levels. At a low threshold
    # pores and blemishes can join one oversized component; that must never erase
    # a valid blemish already found at a higher contrast level.
    spots = np.zeros(score.shape, np.uint8)
    levels = list(np.arange(0., sensitivity + 1e-7, .05))
    levels.append(float(sensitivity))
    for level in levels:
        threshold = .008 + .14 * (1-level)**2
        candidates = ((score > threshold) & skin).astype(np.uint8)
        # Remove fine texture before connected-component selection. Opening breaks
        # thin pore chains too; a large bounding box alone cannot qualify a pore.
        diameter = max(1, int(min_spot_size))
        if diameter > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
            candidates = cv2.morphologyEx(candidates, cv2.MORPH_OPEN, kernel)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, 8)
        keep = np.zeros(n, np.uint8)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area >= max(3, .5*diameter*diameter) and min(w,h) >= diameter and max(w,h) <= max_spot_size and max(w,h)/max(1,min(w,h)) <= 3 and area/(w*h) >= .22:
                keep[i] = 1
        spots |= keep[labels]
    return spots


def _exclude_smallest(spots, percent):
    """Rank per-frame candidate components before dilation, by pixel area.

    Drop ceil(N * percent / 100) components. Equal areas use scan order.
    Return their footprint too, so later dilation cannot repaint excluded pixels.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(spots, 8)
    excluded = np.zeros_like(spots)
    count = min(n-1, int(np.ceil((n-1)*percent/100)))
    if count:
        order = np.argsort(stats[1:, cv2.CC_STAT_AREA], kind='stable')[:count]+1
        dropped = np.zeros(n, np.uint8)
        dropped[order] = 1
        excluded = dropped[labels]
    return spots * (1-excluded), excluded


def _mask(value, shape, name):
    if value is None:
        return None
    if value.ndim != 3 or tuple(value.shape[1:]) != shape[1:] or value.shape[0] not in (1, shape[0]):
        raise ValueError(f"{name}: expected (B,H,W), batch 1 or image batch")
    return value.detach().float().cpu().numpy().clip(0, 1)


def process(image, sensitivity=.65, max_spot_size=32, skin_range=1., strength=1.,
            dark_amount=1., red_amount=1., bright_amount=.5, mask=None, protect_mask=None,
            ignore_smallest_percent=0., min_spot_size=5):
    if image.ndim != 4 or image.shape[-1] != 3 or not torch.is_floating_point(image):
        raise ValueError("image must be floating RGB (B,H,W,3)")
    values = (sensitivity, skin_range, strength, dark_amount, red_amount, bright_amount, ignore_smallest_percent)
    bounds = ((0,1),(.5,2),(0,1),(0,2),(0,2),(0,2),(0,100))
    if any(not np.isfinite(v) or not lo <= v <= hi for v,(lo,hi) in zip(values,bounds)):
        raise ValueError("Invalid detection parameters")
    if not 4 <= max_spot_size <= 256:
        raise ValueError("max_spot_size must be 4..256 pixels")
    if not isinstance(min_spot_size, (int, np.integer)) or not 1 <= min_spot_size <= max_spot_size:
        raise ValueError("最小瑕疵直径必须为1到最大瑕疵尺寸之间的整数")
    src = image.detach().float().cpu().numpy().clip(0,1)
    user = _mask(mask, src.shape[:3], "mask")
    protect = _mask(protect_mask, src.shape[:3], "protect_mask")
    output, skin_maps, spot_maps = [], [], []
    for b, rgb in enumerate(src):
        h,w = rgb.shape[:2]
        ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
        y,cr,cb = cv2.split(ycc)
        limit = np.ones((h,w),bool) if user is None else user[min(b,len(user)-1)] > .5
        # Broad physical prior used only for sampling normal skin chroma.
        seed = limit & (cr>.53) & (cr<.73) & (cb>.27) & (cb<.54) & (y>.04) & (y<.98)
        if protect is not None:
            seed &= protect[min(b,len(protect)-1)] <= .5
        # Sample on a grid to bound colour statistics cost on large photographs.
        step = max(1,int(np.ceil(max(h,w)/768)))
        samples = ycc[::step,::step,1:][seed[::step,::step]]
        if len(samples) < 8:
            output.append(rgb.copy()); skin_maps.append(np.zeros((h,w),np.float32)); spot_maps.append(np.zeros((h,w),np.float32)); continue
        centre = np.median(samples,axis=0)
        spread = np.maximum(1.4826*np.median(np.abs(samples-centre),axis=0), .012)
        distance = np.max(np.abs(ycc[...,1:]-centre)/(spread*3*skin_range),axis=-1)
        normal = (distance<1) & limit & (y>.025) & (y<.99)
        # Bridge small red/dark blemishes using surrounding skin; their own colour
        # must not exclude the very marks being removed. This never crosses user limits.
        radius = max(2,int(max_spot_size)//2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))
        skin = cv2.morphologyEx(normal.astype(np.uint8),cv2.MORPH_CLOSE,kernel).astype(bool) & limit
        if protect is not None:
            skin &= protect[min(b,len(protect)-1)] <= .5
        # Multi-scale local backgrounds: small comedones and larger spots independently.
        dark = np.zeros((h,w),np.float32); red=dark.copy(); bright=dark.copy()
        redness = rgb[...,0]-.5*(rgb[...,1]+rgb[...,2])
        # Suppress fine texture in the detection signal itself, before local
        # backgrounds create a bright ring around each dark pore. Repair uses RGB.
        prefilter = min(5, int(min_spot_size))
        if prefilter % 2 == 0:
            prefilter -= 1
        detection_y = cv2.medianBlur(y, prefilter) if prefilter > 1 else y
        if prefilter > 1:
            redness = cv2.medianBlur(redness, prefilter)
        for size in sorted(set((max(4,int(max_spot_size)//4),int(max_spot_size)))):
            scale = max(1,int(np.ceil(size/4)))
            small = (max(1,int(np.ceil(w/scale))),max(1,int(np.ceil(h/scale))))
            def background(channel):
                reduced = cv2.resize(channel,small,interpolation=cv2.INTER_AREA)
                median = cv2.medianBlur(reduced,5)
                return cv2.resize(median,(w,h),interpolation=cv2.INTER_LINEAR)
            base = background(detection_y)
            dark=np.maximum(dark,base-detection_y); bright=np.maximum(bright,detection_y-base)
            red=np.maximum(red,redness-background(redness))
        score=np.maximum.reduce((dark*dark_amount, red*red_amount, bright*bright_amount))
        spots=_detect_spots(score,skin,sensitivity,max_spot_size,min_spot_size)
        spots,excluded=_exclude_smallest(spots,ignore_smallest_percent)
        spots=cv2.dilate(spots,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))) * skin.astype(np.uint8)
        spots *= 1-excluded
        healed=rgb.copy()
        # Inpaint each local ROI to bound memory, keeping float image precision.
        n,labels,stats,_=cv2.connectedComponentsWithStats(spots,8)
        for i in range(1,n):
            x,yy,ww,hh,_=stats[i]; pad=max(8,max(ww,hh))
            x0,x1=max(0,x-pad),min(w,x+ww+pad); y0,y1=max(0,yy-pad),min(h,yy+hh+pad)
            local=spots[y0:y1,x0:x1]
            crop=rgb[y0:y1,x0:x1]
            fill=np.stack([cv2.inpaint(np.ascontiguousarray(crop[...,c]*255.0),local,3,cv2.INPAINT_TELEA)/255.0 for c in range(3)],axis=-1)
            target=labels[y0:y1,x0:x1]==i
            healed[y0:y1,x0:x1][target]=fill[target].clip(0,1)
        alpha=cv2.GaussianBlur(spots.astype(np.float32),(3,3),.65)*spots*strength
        if user is not None: alpha*=user[min(b,len(user)-1)]
        if protect is not None: alpha*=1-protect[min(b,len(protect)-1)]
        result=rgb+(healed-rgb)*alpha[...,None]
        output.append(result.clip(0,1)); skin_maps.append(skin.astype(np.float32)); spot_maps.append(spots.astype(np.float32))
    def tensor(items): return torch.from_numpy(np.stack(items)).to(image.device)
    return tensor(output),tensor(spot_maps),tensor(skin_maps)


class SkinBlemish:
    CATEGORY="MY/修图"
    RETURN_TYPES=("IMAGE","MASK","MASK")
    RETURN_NAMES=("image","spot_mask","skin_mask")
    FUNCTION="retouch"

    @classmethod
    def INPUT_TYPES(cls):
        def f(default,lo,hi,tip):
            return ("FLOAT",{"default":default,"min":lo,"max":hi,"step":.05,"tooltip":tip})
        return {"required":{
            "image":("IMAGE",),
            "sensitivity":f(.65,0,1,"检测灵敏度，越高越容易检出浅斑，也可能选中毛孔"),
            "max_spot_size":("INT",{"default":32,"min":4,"max":256,"tooltip":"允许的最大瑕疵直径，原图像素"}),
            "skin_range":f(1,.5,2,"根据当前图肤色统计扩展颜色范围"),
            "strength":f(1,0,1,"0保留原图，0.5减淡，1完整修补"),
            "dark_amount":f(1,0,2,"暗斑、黑头、痦子检测权重；0关闭"),
            "red_amount":f(1,0,2,"红点、红痘检测权重；0关闭"),
            "bright_amount":f(.5,0,2,"白头、亮点检测权重；0关闭")},
            "optional":{"min_spot_size":("INT",{"default":5,"min":1,"max":256,"tooltip":"原图像素。在检测阶段排除容不下此直径圆核的细小纹理；提高可减少毛孔误检，也会漏掉小痘点。"}),"ignore_smallest_percent":("FLOAT",{"default":0.,"min":0.,"max":100.,"step":1.,"tooltip":"可选相对面积排序过滤，0关闭；毛孔过滤请用最小瑕疵直径。"}),"protect_hair":("BOOLEAN",{"default":True,"tooltip":"自动分割头发并保护，不调用人脸检测"}),"mask":("MASK",{"tooltip":"可选皮肤搜索范围，白色允许；不会调用人脸检测"}),
                        "protect_mask":("MASK",{"tooltip":"白色保护，可保护眼睛、嘴唇、痣等"})}}

    def retouch(self,protect_hair=True,**kwargs):
        if protect_hair:
            from .hair import hair_mask
            hair=hair_mask(kwargs["image"])
            existing=kwargs.get("protect_mask")
            kwargs["protect_mask"]=hair if existing is None else torch.maximum(hair,existing.to(hair.device))
        return process(**kwargs)

NODE_CLASS_MAPPINGS={"MYSkinBlemish":SkinBlemish}
NODE_DISPLAY_NAME_MAPPINGS={"MYSkinBlemish":"皮肤瑕疵去除 / 减淡"}


