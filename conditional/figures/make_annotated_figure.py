import json, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import font_manager
from PIL import Image
for f in font_manager.findSystemFonts():
    if 'texgyretermes' in f.lower(): font_manager.fontManager.addfont(f)
# usage: python make_annotated_figure.py <generated_dir> <spec.json> <out.pdf>
G=sys.argv[1].rstrip('/')+'/'
spec=json.load(open(sys.argv[2])); out=sys.argv[3]
cols=['#00B4FF','#FF2EC4']
with plt.rc_context({'font.family':'serif','font.serif':['TeX Gyre Termes','Times New Roman','DejaVu Serif'],'font.size':7.5,'pdf.fonttype':42}):
    fig=plt.figure(figsize=(7.0,5.1))
    W=4; panel_w=1/W
    for k,p in enumerate(spec):
        r,c=divmod(k,W)
        x0=c*panel_w+0.005; top=1-r*0.5
        img=np.asarray(Image.open(G+p['file']).convert('RGB'))
        ax=fig.add_axes([x0, top-0.30, panel_w-0.01, 0.27])
        ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(p['title'],fontsize=8.5,pad=2,fontweight='bold')
        n=len(p['boxes'])
        for j,b in enumerate(p['boxes']):
            x,y,s=b['x'],b['y'],b['s']
            ax.add_patch(Rectangle((x,y),s,s,fill=False,ec=cols[j],lw=1.3))
            ax.text(x+s-3,y+3,'i'*(j+1),color=cols[j],fontsize=7,va='top',ha='right',fontweight='bold')
            wz=(panel_w-0.01)/n
            az=fig.add_axes([x0+j*wz+0.004, top-0.43, wz-0.008, 0.12])
            az.imshow(img[y:y+s,x:x+s],interpolation='lanczos'); az.set_xticks([]); az.set_yticks([])
            for sp in az.spines.values(): sp.set_edgecolor(cols[j]); sp.set_linewidth(1.5)
            az.set_xlabel(f"({'i'*(j+1)}) "+b['label'],fontsize=6.5,labelpad=1.5,linespacing=0.95)
    fig.savefig(out,dpi=300)
