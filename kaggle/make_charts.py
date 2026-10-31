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

def read_run(prefix,cfg,seed):
    f='kaggle/results/%s_%s_s%d.txt'%(prefix,cfg,seed)
    if not os.path.exists(f):
        return None
    r={'epochs':[]}
    for ln in open(f).read().split('\n'):
        p=ln.split()
        if len(p)==4 and p[0].isdigit():
            r['epochs'].append((float(p[1]),float(p[2]),float(p[3])))
        if p and p[0]=='test' and len(p)>=5:
            r['t1']=float(p[2])
        if p and p[0]=='sev' and p[1]=='test':
            wrong=int(p[3])
            r['ord']=int(p[7])/wrong
            r['fam']=int(p[9])/wrong
    if 'ord' not in r:
        return None
    return r

def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ['top','right']:
        ax.spines[side].set_visible(False)
    for side in ['left','bottom']:
        ax.spines[side].set_color('#c3c2b7')
    ax.tick_params(colors=INK2,labelsize=9)
    ax.yaxis.grid(True,color=GRID,linewidth=0.8)
    ax.set_axisbelow(True)

os.makedirs('kaggle/figures',exist_ok=True)

conditions=[
    ('baseline','none','base'),
    ('shuffled\norder head','order-shuffle','control'),
    ('real\norder head','order','real'),
    ('shuffled\nclass+order\n+family','class-order-family-shuffle','control'),
    ('real\nclass+order\n+family','class-order-family','real'),
]
found=[]
for label,cfg,kind in conditions:
    runs=[]
    for s in range(1,6):
        r=read_run('taxo200',cfg,s)
        if r is not None:
            runs.append(r)
    if len(runs)==5:
        found.append((label,kind,runs))

fig,axes=plt.subplots(1,2,figsize=(10,4.2),facecolor=SURFACE)
panels=[('ord','Wrong guesses that land in the right order (%)'),('fam','Wrong guesses that land in the right family (%)')]
for ax,(key,title) in zip(axes,panels):
    style(ax)
    for x in range(len(found)):
        label,kind,runs=found[x]
        vals=[]
        for r in runs:
            vals.append(100*r[key])
        if kind=='real':
            face=BLUE
            edge=BLUE
        elif kind=='base':
            face=INK2
            edge=INK2
        else:
            face=SURFACE
            edge=MUTED
        offsets=[-0.12,-0.06,0,0.06,0.12]
        for k in range(len(vals)):
            ax.scatter([x+offsets[k]],[vals[k]],s=40,facecolor=face,edgecolor=edge,linewidth=1.5,zorder=3)
        mean=sum(vals)/len(vals)
        ax.plot([x-0.25,x+0.25],[mean,mean],color=INK,linewidth=2,zorder=4)
        ax.text(x+0.28,mean,'%.1f'%mean,color=INK,fontsize=9,va='center')
    ax.set_xticks(range(len(found)))
    labels=[]
    for f in found:
        labels.append(f[0])
    ax.set_xticklabels(labels,fontsize=8.5)
    ax.set_xlim(-0.6,len(found)-0.2)
    ax.set_title(title,loc='left',fontsize=10.5,color=INK)
fig.text(0.01,0.01,'Test set, 200 species, 20 epochs, 5 seeds per condition. Dots = one seed, bar = mean. Hollow = shuffled control.',fontsize=8,color=MUTED)
fig.tight_layout(rect=(0,0.04,1,1))
fig.savefig('kaggle/figures/right_group_share.png',dpi=200,facecolor=SURFACE)
plt.close(fig)

fig,ax=plt.subplots(figsize=(7,4.2),facecolor=SURFACE)
style(ax)
for cfg,name,color in [('none','baseline',ORANGE),('order','order head',BLUE)]:
    runs=[]
    for s in range(1,6):
        r=read_run('taxo200',cfg,s)
        if r is not None:
            runs.append(r)
    if len(runs)==0:
        continue
    n_epochs=len(runs[0]['epochs'])
    train_mean=[]
    test_mean=[]
    for e in range(n_epochs):
        a=0
        b=0
        for r in runs:
            a=a+r['epochs'][e][0]
            b=b+r['epochs'][e][2]
        train_mean.append(a/len(runs))
        test_mean.append(b/len(runs))
    xs=list(range(1,n_epochs+1))
    ax.plot(xs,train_mean,color=color,linewidth=2,linestyle='--',label=name+', train')
    ax.plot(xs,test_mean,color=color,linewidth=2,label=name+', test')
ax.set_xlabel('epoch',color=INK2,fontsize=9)
ax.set_ylabel('species top-1 accuracy',color=INK2,fontsize=9)
ax.set_title('Training keeps climbing, test accuracy flattens near 0.12',loc='left',fontsize=10.5,color=INK)
ax.legend(frameon=False,fontsize=8.5,labelcolor=INK2,loc='upper left')
fig.text(0.01,0.01,'200 species, both flips, mean of 5 seeds. Train accuracy is measured on 2,000 training photos.',fontsize=8,color=MUTED)
fig.tight_layout(rect=(0,0.04,1,1))
fig.savefig('kaggle/figures/learning_curves.png',dpi=200,facecolor=SURFACE)
plt.close(fig)
print('found conditions:',len(found))
