import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

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

def std_of(values):
    m=mean_of(values)
    total=0
    for v in values:
        total=total+(v-m)**2
    return (total/(len(values)-1))**0.5

def top1(path,key):
    out=None
    for ln in open(path):
        p=ln.split()
        if len(p)>=5 and p[0]==key and p[1]=='top1':
            out=float(p[2])
    return out

os.makedirs('figures',exist_ok=True)

conds=[('baseline\n(species head)','none',MUTED),('condition 2\n(+ order head)','order',BLUE)]
fig,axes=plt.subplots(1,3,figsize=(11,4.2),facecolor=SURFACE,sharey=True)
for ax,(key,title) in zip(axes,[('train','train error'),('val','validation error'),('test','test error')]):
    style(ax)
    for x in range(len(conds)):
        vals=[]
        for s in range(1,6):
            acc=top1('results/taxo200_%s_s%d.txt'%(conds[x][1],s),key)
            vals.append(1-acc)
        offsets=[-0.12,-0.06,0,0.06,0.12]
        for k in range(len(vals)):
            ax.scatter([x+offsets[k]],[vals[k]],s=40,color=conds[x][2],zorder=3)
        m=mean_of(vals)
        ax.plot([x-0.25,x+0.25],[m,m],color=INK,linewidth=2,zorder=4)
        ax.text(x+0.28,m,'%.3f\n(sd %.3f)'%(m,std_of(vals)),color=INK,fontsize=8.5,va='center')
    ax.set_xticks([0,1])
    names=[]
    for c in conds:
        names.append(c[0])
    ax.set_xticklabels(names,fontsize=8.5)
    ax.set_xlim(-0.6,1.9)
    ax.set_ylim(0.72,0.90)
    ax.set_title(title,loc='left',fontsize=10,color=INK)
axes[0].set_ylabel('top-1 error (1 - accuracy)',color=INK2,fontsize=9)
fig.text(0.01,0.01,'200 species, 20 epochs, both flips, 5 seeds (dots), mean (bar), sd = standard deviation over seeds. Random guessing = 0.995.',fontsize=8,color=MUTED)
fig.tight_layout(rect=(0,0.04,1,1))
fig.savefig('figures/two_conditions_errors.png',dpi=200,facecolor=SURFACE)
plt.close(fig)

fig,ax=plt.subplots(figsize=(11,4.6),facecolor=SURFACE)
ax.set_xlim(0,100)
ax.set_ylim(0,52)
ax.axis('off')

def box(x,y,w,h,text,color,edge):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.3,rounding_size=1.2',facecolor=color,edgecolor=edge,linewidth=1.4))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=9,color=INK)

def arrow(x1,y1,x2,y2,color):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='-|>',color=color,linewidth=1.6))

box(1,18,12,10,'photo\n64 x 64 x 3','#f2f1ec',MUTED)
box(18,18,22,10,'3 convolution blocks\n32, 64, 128 filters\n(3x3, ReLU, max pool)','#f2f1ec',MUTED)
box(44,18,16,10,'Flatten, Linear\n8192 to 256,\nReLU, Dropout 0.5','#f2f1ec',MUTED)
box(63,31,16,9,'species head\nLinear 256 to 200','#dbe8f8',BLUE)
box(63,5,16,9,'order head\nLinear 256 to 20','#dbe8f8',BLUE)
box(85,31,14,9,'species loss\n(cross-entropy)','#f2f1ec',MUTED)
box(85,5,14,9,'order loss\n(cross-entropy)','#f2f1ec',MUTED)
arrow(13.6,23,17.4,23,INK2)
arrow(40.6,23,44.4,23,INK2)
arrow(58.6,26,62.4,34,INK2)
arrow(58.6,20,62.4,12,BLUE)
arrow(79.6,35.5,84.4,35.5,INK2)
arrow(79.6,9.5,84.4,9.5,BLUE)
ax.text(1,49,'Condition 1 (baseline): species head and species loss only.',fontsize=10,color=INK2)
ax.text(1,46,'Condition 2: the blue order head and order loss are added; total loss = species loss + order loss.',fontsize=10,color=BLUE)
ax.text(45,14.5,'256 shared numbers',fontsize=8.5,color=INK2)
fig.text(0.01,0.01,'2,242,056 parameters in the baseline. Both heads read the same 256 numbers, so the order loss also trains the shared body.',fontsize=8,color=MUTED)
fig.savefig('figures/model_diagram.png',dpi=200,facecolor=SURFACE,bbox_inches='tight')
plt.close(fig)

levels=['kingdom','phylum','class','order','family','genus','species']
group_counts=[3,11,35,170,539,1566,2076]
big=[]
for lv in levels:
    for ln in open('results/all32_%s_s1.txt'%lv):
        p=ln.split()
        if len(p)>=8 and p[0]=='chance':
            big.append(float(p[7]))
fig,axes=plt.subplots(1,2,figsize=(10.5,4),facecolor=SURFACE)
style(axes[0])
axes[0].bar(range(7),group_counts,width=0.6,color=BLUE)
axes[0].set_yscale('log')
for i in range(7):
    axes[0].text(i,group_counts[i]*1.15,str(group_counts[i]),color=INK2,fontsize=8.5,ha='center')
axes[0].set_xticks(range(7))
axes[0].set_xticklabels(levels,fontsize=8,rotation=30)
axes[0].set_ylim(1,6000)
axes[0].set_ylabel('number of groups (log scale)',color=INK2,fontsize=9)
axes[0].set_title('The question gets finer: more groups at each level',loc='left',fontsize=10,color=INK)
style(axes[1])
axes[1].plot(range(7),big,color=MUTED,linewidth=2,marker='o',markersize=7)
for i in range(7):
    axes[1].text(i,big[i]+0.03,'%.3f'%big[i],color=INK2,fontsize=8.5,ha='center')
axes[1].set_xticks(range(7))
axes[1].set_xticklabels(levels,fontsize=8,rotation=30)
axes[1].set_ylim(0,0.65)
axes[1].set_ylabel('test share of the biggest group',color=INK2,fontsize=9)
axes[1].set_title('Accuracy of "always guess the biggest group"',loc='left',fontsize=10,color=INK)
fig.text(0.01,0.01,'All 2,076 species, 20,760 test photos. A group at one level is the whole path from the kingdom down to that level.',fontsize=8,color=MUTED)
fig.tight_layout(rect=(0,0.04,1,1))
fig.savefig('figures/taxonomy_levels.png',dpi=200,facecolor=SURFACE)
plt.close(fig)
print('made 3 charts')
