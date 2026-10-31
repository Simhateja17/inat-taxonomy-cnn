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
AQUA='#1baf7a'
MAGENTA='#e87ba4'
VIOLET='#6250d6'

levels=['kingdom','phylum','class','order','family','genus','species']
group_counts=[3,11,35,170,539,1566,2076]

def read_run(path):
    if not os.path.exists(path):
        return None
    r={'epochs':[]}
    for ln in open(path).read().split('\n'):
        p=ln.split()
        if len(p)==4 and p[0].isdigit():
            r['epochs'].append((float(p[1]),float(p[2]),float(p[3])))
        if len(p)>=8 and p[0]=='chance':
            r['chance']=float(p[1])
            r['big']=float(p[7])
        if len(p)>=5 and p[0]=='train' and p[1]=='top1':
            r['train1']=float(p[2])
        if len(p)>=5 and p[0]=='val' and p[1]=='top1':
            r['val1']=float(p[2])
        if len(p)>=5 and p[0]=='test' and p[1]=='top1':
            r['test1']=float(p[2])
            r['test5']=float(p[4])
    if 'test1' not in r:
        return None
    return r

def cnn_runs(level):
    out=[]
    for s in range(1,4):
        r=read_run('kaggle/results/all32_%s_s%d.txt'%(level,s))
        if r is not None:
            out.append(r)
    return out

def arch_runs(arch,level):
    out=[]
    for tag in ['s1','s2','s3','s2g']:
        r=read_run('kaggle/results/all32_%s_%s_%s.txt'%(arch,level,tag))
        if r is not None:
            out.append(r)
    return out

def mean_of(values):
    return sum(values)/len(values)

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

archs=[('LeNet','lenet',MUTED),('AlexNet-style','alexnet',ORANGE),('small ResNet','resnet',AQUA),('ResNet-18 style','resnet18',MAGENTA),('ResNet-18 pretrained','resnet18_pre',VIOLET)]

def by_level(key,filename,title,cifar):
    fig,ax=plt.subplots(figsize=(10,5.2),facecolor=SURFACE)
    style(ax)
    xs=list(range(len(levels)))
    chance=[]
    big=[]
    for lv in levels:
        runs=cnn_runs(lv)
        if len(runs)>0:
            chance.append(runs[0]['chance'])
            big.append(runs[0]['big'])
        else:
            chance.append(None)
            big.append(None)
    cx=[]
    cy=[]
    bx=[]
    by=[]
    for i in xs:
        if chance[i] is not None:
            cx.append(i)
            cy.append(chance[i])
            bx.append(i)
            by.append(big[i])
    if key=='test1':
        ax.plot(cx,cy,color=MUTED,linewidth=1.2,linestyle=':',label='chance')
        ax.plot(bx,by,color=MUTED,linewidth=1.2,linestyle='--',label='always guess the biggest group')
    mx=[]
    my=[]
    for i in xs:
        runs=cnn_runs(levels[i])
        if len(runs)==0:
            continue
        vals=[]
        for r in runs:
            vals.append(r[key])
        for v in vals:
            ax.scatter([i],[v],s=22,color=BLUE,alpha=0.55,zorder=3)
        mx.append(i)
        my.append(mean_of(vals))
    if len(mx)>0:
        ax.plot(mx,my,color=BLUE,linewidth=2.2,marker='o',markersize=7,label='our CNN (mean of up to 3 seeds)',zorder=4)
    for name,arch,color in archs:
        ax_x=[]
        ax_y=[]
        most=0
        for i in xs:
            runs=arch_runs(arch,levels[i])
            if len(runs)>0:
                vals=[]
                for r in runs:
                    vals.append(r[key])
                    ax.scatter([i],[r[key]],s=14,color=color,alpha=0.5,zorder=3)
                ax_x.append(i)
                ax_y.append(mean_of(vals))
                if len(runs)>most:
                    most=len(runs)
        if len(ax_x)>0:
            label=name+' (%d seed'%most
            if most>1:
                label=label+'s, mean)'
            else:
                label=label+')'
            ax.plot(ax_x,ax_y,color=color,linewidth=1.6,marker='s',markersize=5,label=label,zorder=3)
    if cifar:
        ax.axhline(0.717,color=INK2,linewidth=1,linestyle='-.')
        ax.text(2.6,0.727,'CIFAR-10, same kind of CNN: 0.717',color=INK2,fontsize=8.5,ha='left')
    ax.set_xticks(xs)
    labels=[]
    for i in xs:
        labels.append('%s\n%d groups'%(levels[i],group_counts[i]))
    ax.set_xticklabels(labels,fontsize=8.5)
    ax.set_ylim(0,1.02)
    ax.set_ylabel('test accuracy',color=INK2,fontsize=9)
    ax.set_title(title,loc='left',fontsize=10.5,color=INK)
    ax.legend(frameon=False,fontsize=8,labelcolor=INK2,loc='center right',bbox_to_anchor=(1.0,0.5))
    fig.text(0.01,0.01,'All 2,076 species, 103,800 photos at 32x32, 62,280 training photos. One network per level.',fontsize=8,color=MUTED)
    fig.tight_layout(rect=(0,0.04,1,1))
    fig.savefig('kaggle/figures/'+filename,dpi=200,facecolor=SURFACE)
    plt.close(fig)

by_level('test1','full32_top1_by_level.png','Accuracy as the question gets finer: kingdom to species (top-1)',True)
by_level('test5','full32_top5_by_level.png','Top-5 accuracy as the question gets finer',False)

fig,axes=plt.subplots(2,4,figsize=(12,6),facecolor=SURFACE)
flat=[]
for row in axes:
    for ax in row:
        flat.append(ax)
for i in range(len(levels)):
    ax=flat[i]
    style(ax)
    runs=cnn_runs(levels[i])
    if len(runs)>0:
        n=len(runs[0]['epochs'])
        xs=list(range(1,n+1))
        tr=[]
        va=[]
        for e in range(n):
            tr.append(mean_of([r['epochs'][e][0] for r in runs]))
            va.append(mean_of([r['epochs'][e][1] for r in runs]))
        ax.plot(xs,tr,color=INK2,linewidth=1.8,linestyle='--',label='train')
        ax.plot(xs,va,color=BLUE,linewidth=2,label='validation')
    ax.set_title('%s (%d groups)'%(levels[i],group_counts[i]),loc='left',fontsize=9.5,color=INK)
    ax.set_xlabel('epoch',color=INK2,fontsize=8.5)
flat[0].legend(frameon=False,fontsize=8.5,labelcolor=INK2,loc='lower right')
flat[7].axis('off')
fig.text(0.01,0.01,'Our CNN, 32x32, full dataset, mean of up to 3 seeds. Train = first 5,000 training photos.',fontsize=8,color=MUTED)
fig.tight_layout(rect=(0,0.03,1,1))
fig.savefig('kaggle/figures/full32_curves_by_level.png',dpi=200,facecolor=SURFACE)
plt.close(fig)
print('made 3 charts')
