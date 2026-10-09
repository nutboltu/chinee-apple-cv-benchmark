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

# ---------- Figure 2: marker in context, arrow points at the plant ----------
src='DJI_20251118124919_0185.JPG'
im=Image.open(D/'raw'/src).convert('RGB')
ctx=im.crop((4600,2300,5600,3300)).resize((620,620),Image.LANCZOS)
W2=620*2+PAD*3; H2=HDR+620+CAP+PAD
fig2=Image.new('RGB',(W2,H2),(255,255,255)); d2=ImageDraw.Draw(fig2)
d2.text((PAD,14),"The arrow identifies the target plant",fill=(20,20,20),font=font(22))
fig2.paste(ctx,(PAD,HDR))
d2.rectangle([PAD,HDR,PAD+619,HDR+619],outline=(60,60,60),width=2)
d2.text((PAD,HDR+626),"Marker in context. The board sits on the canopy edge,",fill=(60,60,60),font=font(15))
d2.text((PAD,HDR+644),"arrow pointing down into the plant it marks.",fill=(60,60,60),font=font(15))
# annotated copy
ann=ctx.copy(); da=ImageDraw.Draw(ann)
da.rectangle([(5029-4600)*0.62,(2703-2300)*0.62,(5093-4600)*0.62,(2781-2300)*0.62],outline=(0,230,255),width=3)
da.ellipse([150,300,470,600],outline=(50,255,80),width=3)
da.text((150,278),"target plant",fill=(50,255,80),font=font(17))
da.text((262,222),"marker",fill=(0,230,255),font=font(17))
x2=PAD*2+620
fig2.paste(ann,(x2,HDR))
d2.rectangle([x2,HDR,x2+619,HDR+619],outline=(60,60,60),width=2)
d2.text((x2,HDR+626),"Cyan is the detected board. Green is the plant that gets",fill=(60,60,60),font=font(15))
d2.text((x2,HDR+644),"the polygon. Only the plant is annotated.",fill=(60,60,60),font=font(15))
fig2.save(OUT/'marker_in_context.png')
print('marker_in_context.png', fig2.size)
