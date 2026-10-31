import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SURFACE='#fcfcfb'
INK='#0b0b0b'
INK2='#52514e'
MUTED='#898781'
GRID='#e1e0d9'
BLUE='#2a78d6'
ORANGE='#eb6834'

def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ['top','right']:
        ax.spines[side].set_visible(False)
    for side in ['left','bottom']:
        ax.spines[side].set_color('#c3c2b7')
    ax.tick_params(colors=INK2,labelsize=9)
    ax.yaxis.grid(True,color=GRID,linewidth=0.8)
    ax.set_axisbelow(True)

def mean_of(values):
    return sum(values)/len(values)

def dots(ax,x,vals,color,fmt='%.3f'):
    offsets=[-0.12,-0.06,0,0.06,0.12]
    for k in range(len(vals)):
        ax.scatter([x+offsets[k]],[vals[k]],s=40,color=color,zorder=3)
    mean=mean_of(vals)
    ax.plot([x-0.25,x+0.25],[mean,mean],color=INK,linewidth=2,zorder=4)
    ax.text(x+0.28,mean,fmt%mean,color=INK,fontsize=9,va='center')

os.makedirs('kaggle/figures',exist_ok=True)

steps=[
    ('baseline',[0.0845,0.0935,0.0855,0.0900,0.0740]),
    ('+ left-right\nflip',[0.1140,0.1130,0.0985,0.1045,0.1060]),
    ('+ up-down\nflip',[0.1160,0.1215,0.1165,0.1255,0.1160]),
    ('+ 50 epochs\n(instead of 20)',[0.1200,0.1190,0.1325,0.1195,0.1235]),
]
fig,ax=plt.subplots(figsize=(8,4.2),facecolor=SURFACE)
style(ax)
for x in range(len(steps)):
    color=INK2
    if x==2:
        color=BLUE
    dots(ax,x,steps[x][1],color)
ax.set_xticks(range(len(steps)))
labels=[]
for s in steps:
    labels.append(s[0])
ax.set_xticklabels(labels,fontsize=8.5)
ax.set_xlim(-0.6,len(steps)-0.3)
ax.set_ylabel('validation top-1 accuracy',color=INK2,fontsize=9)
ax.set_title('Tuning the baseline: flips helped, 50 epochs did not',loc='left',fontsize=10.5,color=INK)
fig.text(0.01,0.01,'200 species, 64x64, 5 seeds (dots), mean (bar). Blue = the setting we kept (20 epochs, both flips).',fontsize=8,color=MUTED)
fig.tight_layout(rect=(0,0.04,1,1))
fig.savefig('kaggle/figures/tuning_steps.png',dpi=200,facecolor=SURFACE)
plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(10,4),facecolor=SURFACE)
style(axes[0])
photos=[10,20,30]
val_photos=[0.037,0.058,0.086]
axes[0].plot(photos,val_photos,color=BLUE,linewidth=2,marker='o',markersize=8)
for k in range(3):
    axes[0].text(photos[k],val_photos[k]+0.004,'%.3f'%val_photos[k],color=INK2,fontsize=9,ha='center')
axes[0].set_xticks(photos)
axes[0].set_xlabel('training photos per species',color=INK2,fontsize=9)
axes[0].set_ylabel('validation top-1 accuracy',color=INK2,fontsize=9)
axes[0].set_ylim(0,0.11)
axes[0].set_title('More photos per species helps',loc='left',fontsize=10.5,color=INK)
style(axes[1])
species=[100,200,400]
train_s=[0.705,0.664,0.539]
val_s=[0.170,0.119,0.082]
axes[1].plot(species,train_s,color=INK2,linewidth=2,marker='o',markersize=8,linestyle='--',label='train')
axes[1].plot(species,val_s,color=BLUE,linewidth=2,marker='o',markersize=8,label='validation')
for k in range(3):
    axes[1].text(species[k],train_s[k]+0.03,'%.2f'%train_s[k],color=INK2,fontsize=9,ha='center')
    axes[1].text(species[k],val_s[k]+0.03,'%.2f'%val_s[k],color=INK2,fontsize=9,ha='center')
axes[1].set_xticks(species)
axes[1].set_xlabel('number of species',color=INK2,fontsize=9)
axes[1].set_ylim(0,0.85)
axes[1].set_title('More species: lower accuracy, same big gap',loc='left',fontsize=10.5,color=INK)
axes[1].legend(frameon=False,fontsize=8.5,labelcolor=INK2,loc='upper right')
fig.text(0.01,0.01,'Early tests: seed 1, 15 epochs, no flips, 30 training photos per species (left: 200 species). Single runs.',fontsize=8,color=MUTED)
fig.tight_layout(rect=(0,0.04,1,1))
fig.savefig('kaggle/figures/data_amount.png',dpi=200,facecolor=SURFACE)
plt.close(fig)

def read500(cfg):
    out=[]
    for s in range(1,4):
        path='kaggle/results/taxo500_%s_s%d.txt'%(cfg,s)
        if not os.path.exists(path):
            continue
        r={}
        for ln in open(path).read().split('\n'):
            p=ln.split()
            if len(p)>=5 and p[0]=='test' and p[1]=='top1':
                r['t1']=float(p[2])
                r['t5']=float(p[4])
            if len(p)>=10 and p[0]=='sev' and p[1]=='test':
                r['ord']=100*int(p[7])/int(p[3])
        if 'ord' in r:
            out.append(r)
    return out

conds=[('baseline','none',INK2),('+ order\nhead','order',BLUE),('+ class, order,\nfamily heads','class-order-family',BLUE)]
fig,axes=plt.subplots(1,3,figsize=(11,4),facecolor=SURFACE)
panels=[('t1','test top-1 accuracy','%.3f'),('t5','test top-5 accuracy','%.3f'),('ord','wrong guesses in the right order (%)','%.1f')]
for ax,(key,title,fmt) in zip(axes,panels):
    style(ax)
    for x in range(len(conds)):
        runs=read500(conds[x][1])
        vals=[]
        for r in runs:
            vals.append(r[key])
        offsets=[-0.08,0,0.08]
        for k in range(len(vals)):
            ax.scatter([x+offsets[k]],[vals[k]],s=40,color=conds[x][2],zorder=3)
        if len(vals)>0:
            mean=mean_of(vals)
            ax.plot([x-0.22,x+0.22],[mean,mean],color=INK,linewidth=2,zorder=4)
            ax.text(x+0.25,mean,fmt%mean,color=INK,fontsize=9,va='center')
    ax.set_xticks(range(len(conds)))
    names=[]
    for c in conds:
        names.append(c[0])
    ax.set_xticklabels(names,fontsize=8.5)
    ax.set_xlim(-0.6,len(conds)-0.1)
    ax.set_title(title,loc='left',fontsize=10,color=INK)
fig.text(0.01,0.01,'500 species, test set, 30 epochs, 3 seeds (dots), mean (bar). Chance top-1 = 0.002.',fontsize=8,color=MUTED)
fig.tight_layout(rect=(0,0.04,1,1))
fig.savefig('kaggle/figures/species_500.png',dpi=200,facecolor=SURFACE)
plt.close(fig)
print('made 3 charts')
