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

def read_file(path):
    if not os.path.exists(path):
        return None
    r={'epochs':[],'lvl':[]}
    for ln in open(path).read().split('\n'):
        p=ln.split()
        if len(p)==4 and p[0].isdigit():
            r['epochs'].append((float(p[1]),float(p[2]),float(p[3])))
        if len(p)==8 and p[0]=='lvl':
            r['lvl'].append((float(p[3]),float(p[5]),float(p[7])))
        if len(p)>=5 and p[0]=='train' and p[1]=='top1':
            r['train']=float(p[2])
        if len(p)>=5 and p[0]=='test' and p[1]=='top1':
            r['test']=float(p[2])
            r['test5']=float(p[4])
        if len(p)>=10 and p[0]=='sev' and p[1]=='test':
            wrong=int(p[3])
            r['ord']=int(p[7])/wrong
            r['fam']=int(p[9])/wrong
    if 'test' not in r:
        return None
    return r

def runs_of(prefix,cfg):
    out=[]
    for s in range(1,6):
        r=read_file('kaggle/results/%s_%s_s%d.txt'%(prefix,cfg,s))
        if r is not None:
            out.append(r)
    return out

def mean_of(values):
    return sum(values)/len(values)

def style(ax,grid_axis='y'):
    ax.set_facecolor(SURFACE)
    for side in ['top','right']:
        ax.spines[side].set_visible(False)
    for side in ['left','bottom']:
        ax.spines[side].set_color('#c3c2b7')
    ax.tick_params(colors=INK2,labelsize=9)
    if grid_axis=='y':
        ax.yaxis.grid(True,color=GRID,linewidth=0.8)
    else:
        ax.xaxis.grid(True,color=GRID,linewidth=0.8)
    ax.set_axisbelow(True)

os.makedirs('kaggle/figures',exist_ok=True)

archs=[('LeNet','lenet'),('AlexNet-\nstyle','alexnet'),('our CNN','cnn'),('small\nResNet','resnet'),('ResNet-18\nstyle','resnet18'),('ResNet-18\npretrained','resnet18_pre')]
found=[]
for label,cfg in archs:
    runs=runs_of('arch200',cfg)
    if len(runs)==5:
        found.append((label,cfg,runs))

fig,ax=plt.subplots(figsize=(8.5,4.4),facecolor=SURFACE)
style(ax)
for x in range(len(found)):
    label,cfg,runs=found[x]
    vals=[]
    for r in runs:
        vals.append(r['test'])
    color=BLUE
    if cfg!='resnet18_pre':
        color=INK2
    offsets=[-0.12,-0.06,0,0.06,0.12]
    for k in range(len(vals)):
        ax.scatter([x+offsets[k]],[vals[k]],s=40,color=color,zorder=3)
    mean=mean_of(vals)
    ax.plot([x-0.25,x+0.25],[mean,mean],color=INK,linewidth=2,zorder=4)
    ax.text(x+0.28,mean,'%.3f'%mean,color=INK,fontsize=9,va='center')
ax.set_xticks(range(len(found)))
tick_labels=[]
for f in found:
    tick_labels.append(f[0])
ax.set_xticklabels(tick_labels,fontsize=8.5)
ax.set_xlim(-0.6,len(found)-0.1)
ax.set_ylabel('test top-1 accuracy (200 species)',color=INK2,fontsize=9)
ax.set_title('Bigger networks trained from scratch did not help; ImageNet pretraining more than doubled accuracy',loc='left',fontsize=10,color=INK)
fig.text(0.01,0.01,'Dots = one seed, bar = mean of 5 seeds. Chance = 0.005. 64x64 photos, 6,000 training photos.',fontsize=8,color=MUTED)
fig.tight_layout(rect=(0,0.04,1,1))
fig.savefig('kaggle/figures/architectures.png',dpi=200,facecolor=SURFACE)
plt.close(fig)

fig,ax=plt.subplots(figsize=(8.5,4.2),facecolor=SURFACE)
style(ax,'x')
for i in range(len(found)):
    label,cfg,runs=found[i]
    tr=mean_of([r['train'] for r in runs])
    te=mean_of([r['test'] for r in runs])
    ax.plot([te,tr],[i,i],color=GRID,linewidth=3,zorder=1)
    ax.scatter([tr],[i],s=60,facecolor=SURFACE,edgecolor=INK2,linewidth=1.8,zorder=3)
    ax.scatter([te],[i],s=60,color=BLUE,zorder=3)
    ax.text(tr+0.015,i,'train %.2f'%tr,color=INK2,fontsize=8.5,va='center')
    ax.text(te-0.015,i,'test %.2f'%te,color=INK2,fontsize=8.5,va='center',ha='right')
ax.set_yticks(range(len(found)))
ylabels=[]
for f in found:
    ylabels.append(f[0].replace('-\n','-').replace('\n',' '))
ax.set_yticklabels(ylabels,fontsize=9)
ax.invert_yaxis()
ax.set_xlim(-0.05,1.0)
ax.set_xlabel('top-1 accuracy',color=INK2,fontsize=9)
ax.set_title('Train accuracy against test accuracy: the gap is the overfitting',loc='left',fontsize=10,color=INK)
fig.text(0.01,0.01,'Hollow = train, filled = test. Mean of 5 seeds, 200 species, 20 epochs.',fontsize=8,color=MUTED)
fig.tight_layout(rect=(0,0.04,1,1))
fig.savefig('kaggle/figures/overfitting_gap.png',dpi=200,facecolor=SURFACE)
plt.close(fig)

points=[
    ('baseline','taxo200','none',INK2),
    ('order head','taxo200','order',BLUE),
    ('class+order\n+family heads','taxo200','class-order-family',BLUE),
    ('soft labels\nbeta 14','soft200','beta14',ORANGE),
    ('soft labels\nbeta 10','soft200','beta10',ORANGE),
    ('soft labels\nbeta 7','soft200','beta7',ORANGE),
]
fig,ax=plt.subplots(figsize=(8.5,4.8),facecolor=SURFACE)
style(ax)
ax.grid(True,color=GRID,linewidth=0.8)
for label,prefix,cfg,color in points:
    runs=runs_of(prefix,cfg)
    if len(runs)<5:
        continue
    xs=[]
    ys=[]
    for r in runs:
        xs.append(r['test'])
        ys.append(100*r['ord'])
    mx=mean_of(xs)
    my=mean_of(ys)
    ax.errorbar([mx],[my],xerr=[[mx-min(xs)],[max(xs)-mx]],yerr=[[my-min(ys)],[max(ys)-my]],fmt='o',color=color,markersize=8,elinewidth=1.2,capsize=3,zorder=3)
    ax.text(mx,max(ys)+0.25,label,color=INK2,fontsize=8.5,ha='center',va='bottom')
ax.set_ylim(16.8,24.2)
ax.set_xlabel('test top-1 accuracy of the species guess',color=INK2,fontsize=9)
ax.set_ylabel('wrong guesses in the right order (%)',color=INK2,fontsize=9)
ax.set_title('Smarter mistakes cost little exact accuracy',loc='left',fontsize=10.5,color=INK)
fig.text(0.01,0.01,'200 species, test set, mean and range of 5 seeds. Blue = extra taxonomy heads, orange = soft labels, gray = baseline.',fontsize=8,color=MUTED)
fig.tight_layout(rect=(0,0.04,1,1))
fig.savefig('kaggle/figures/accuracy_vs_mistake_quality.png',dpi=200,facecolor=SURFACE)
plt.close(fig)

models=[('LeNet','lenet'),('our CNN','cnn'),('ResNet-18 style','resnet18'),('ResNet-18 pretrained','resnet18_pre')]
fig,axes=plt.subplots(1,4,figsize=(12,3.8),facecolor=SURFACE,sharey=True)
for ax,(title,cfg) in zip(axes,models):
    style(ax)
    runs=runs_of('arch200',cfg)
    if len(runs)==0:
        continue
    n=len(runs[0]['epochs'])
    series=[('class',0,BLUE),('order',1,ORANGE),('family',2,AQUA)]
    xs=list(range(1,n+1))
    for name,k,color in series:
        ys=[]
        for e in range(n):
            ys.append(mean_of([r['lvl'][e][k] for r in runs]))
        ax.plot(xs,ys,color=color,linewidth=2,label=name)
    ys=[]
    for e in range(n):
        ys.append(mean_of([r['epochs'][e][1] for r in runs]))
    ax.plot(xs,ys,color=MAGENTA,linewidth=2,label='species')
    ax.set_title(title,loc='left',fontsize=10,color=INK)
    ax.set_xlabel('epoch',color=INK2,fontsize=9)
axes[0].set_ylabel('validation accuracy',color=INK2,fontsize=9)
axes[0].legend(frameon=False,fontsize=8.5,labelcolor=INK2,loc='upper left')
fig.text(0.01,0.01,'Group guess = group with the largest summed species probability. 200 species, mean of 5 seeds. Always guessing the biggest class would score 0.35.',fontsize=8,color=MUTED)
fig.tight_layout(rect=(0,0.04,1,1))
fig.savefig('kaggle/figures/level_learning_curves.png',dpi=200,facecolor=SURFACE)
plt.close(fig)
print('made 4 charts')
