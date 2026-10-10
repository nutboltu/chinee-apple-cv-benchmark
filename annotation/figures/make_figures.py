import csv, collections
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
Image.MAX_IMAGE_PIXELS=None
D=Path('/Users/farhadyasir/Workspace/phd-research/chinee-apple-cv-benchmark/chinee-apple-dataset/matured-dry-30m')
OUT=Path('/Users/farhadyasir/Workspace/phd-research/chinee-apple-cv-benchmark/annotation/figures')

def font(sz):
    for p in ["/System/Library/Fonts/Supplemental/Arial Bold.ttf","/System/Library/Fonts/Helvetica.ttc"]:
        try: return ImageFont.truetype(p, sz)
        except Exception: pass
    return ImageFont.load_default()

rows=[r for r in csv.DictReader((D/'processed/markers_classified.csv').open())]
by=collections.defaultdict(list)
for r in rows: by[r['arrow_colour']].append(r)
# pick the clearest exemplar per class, not merely the largest board
by['blue'].sort(key=lambda r:-int(r['blue_px']))
by['pink'].sort(key=lambda r:-int(r['pink_px']))
by['blank'].sort(key=lambda r:-int(r['area_px']))
by['unknown'].sort(key=lambda r:(int(r['blue_px'])+int(r['pink_px']), int(r['area_px'])))

cache={}
def crop(r, half=150, out=240):
    s=r['source']
    if s not in cache: cache[s]=Image.open(D/'raw'/s).convert('RGB')
    im=cache[s]
    cx=(int(r['x0'])+int(r['x1']))//2; cy=(int(r['y0'])+int(r['y1']))//2
    cx=max(half,min(im.width-half,cx)); cy=max(half,min(im.height-half,cy))
    return im.crop((cx-half,cy-half,cx+half,cy+half)).resize((out,out),Image.LANCZOS)

# ---------- Figure 1: marker classes ----------
classes=[('blue','Blue arrow','261 markers, 47.6%'),
         ('pink','Pink arrow','114 markers, 20.8%'),
         ('blank','Blank board','40 markers, 7.3%'),
         ('unknown','Paint not detected','133 markers, 24.3%')]
S=240; PAD=12; HDR=52; CAP=40
W=len(classes)*(S+PAD)+PAD; H=HDR+S+CAP+PAD
fig=Image.new('RGB',(W,H),(255,255,255)); d=ImageDraw.Draw(fig)
d.text((PAD,14),"Cardboard ground-truth markers, by arrow colour",fill=(20,20,20),font=font(22))
for i,(key,title,sub) in enumerate(classes):
    x=PAD+i*(S+PAD); y=HDR
    r=by[key][0] if by[key] else None
    half = 80 if key in ('pink','unknown') else 150
    if r: fig.paste(crop(r,half,S),(x,y))
    col={'blue':(0,110,220),'pink':(220,40,120),'blank':(110,110,110),'unknown':(190,140,0)}[key]
    d.rectangle([x,y,x+S-1,y+S-1],outline=col,width=4)
    d.text((x,y+S+6),title,fill=col,font=font(18))
    d.text((x,y+S+24),sub,fill=(90,90,90),font=font(14))
fig.save(OUT/'marker_classes.png')
print('marker_classes.png', fig.size)

# ---------- Figure 2: blue marker in context, arrow points at the plant ----------
import numpy as np
from scipy import ndimage as ndi

SRC, BLOB = 'DJI_20251118125709_0656.JPG', '630'
m=[r for r in rows if r['source']==SRC and r['blob']==BLOB][0]
im=Image.open(D/'raw'/SRC).convert('RGB')
X0,Y0,X1,Y1 = 5950,1750,6750,2550          # window holding both board and plant
SIDE=620; k=SIDE/(X1-X0)
ctx=im.crop((X0,Y0,X1,Y1)).resize((SIDE,SIDE),Image.LANCZOS)

# locate the marked plant by excess green inside the window, left of the board
a=np.asarray(ctx).astype(np.int16)
R,G,B=a[...,0],a[...,1],a[...,2]
veg=ndi.binary_opening((2*G-R-B)>18, np.ones((9,9),bool))
lab,n=ndi.label(veg)
best=None
if n:
    areas=ndi.sum_labels(veg,lab,index=np.arange(1,n+1))
    for i in np.argsort(areas)[::-1]:
        sl=ndi.find_objects(lab)[i]
        if areas[i]>3000: best=sl; break

mb=[(int(m['x0'])-X0)*k,(int(m['y0'])-Y0)*k,(int(m['x1'])-X0)*k,(int(m['y1'])-Y0)*k]

PADX=12; HDR=52
W2=SIDE*2+PADX*3; H2=HDR+SIDE+46+PADX
fig2=Image.new('RGB',(W2,H2),(255,255,255)); d2=ImageDraw.Draw(fig2)
d2.text((PADX,14),"The blue arrow identifies the target plant",fill=(20,20,20),font=font(22))
fig2.paste(ctx,(PADX,HDR))
d2.rectangle([PADX,HDR,PADX+SIDE-1,HDR+SIDE-1],outline=(60,60,60),width=2)
d2.text((PADX,HDR+SIDE+6),"Marker in context. The blue arrow points up and left,",fill=(60,60,60),font=font(15))
d2.text((PADX,HDR+SIDE+24),"away from the board and into the plant it marks.",fill=(60,60,60),font=font(15))

ann=ctx.copy(); da=ImageDraw.Draw(ann)
da.rectangle(mb,outline=(0,230,255),width=3)
da.text((mb[0],max(2,mb[1]-20)),"marker",fill=(0,230,255),font=font(17))
if best:
    ys,xs=best
    da.ellipse([xs.start,ys.start,xs.stop,ys.stop],outline=(50,255,80),width=3)
    # keep the label on-canvas when the blob runs to the top edge
    ty = ys.start-20 if ys.start>=22 else ys.start+6
    da.text((xs.start+6,ty),"target plant",fill=(50,255,80),font=font(17))
x2=PADX*2+SIDE
fig2.paste(ann,(x2,HDR))
d2.rectangle([x2,HDR,x2+SIDE-1,HDR+SIDE-1],outline=(60,60,60),width=2)
d2.text((x2,HDR+SIDE+6),"Cyan is the detected board. Green is the plant that gets",fill=(60,60,60),font=font(15))
d2.text((x2,HDR+SIDE+24),"the polygon. Only the plant is annotated.",fill=(60,60,60),font=font(15))
fig2.save(OUT/'marker_in_context.png')
print('marker_in_context.png', fig2.size)
