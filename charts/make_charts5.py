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

def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ['top','right']:
        ax.spines[side].set_visible(False)
    for side in ['left','bottom']:
        ax.spines[side].set_color('#c3c2b7')
    ax.tick_params(colors=INK2,labelsize=9)
    ax.yaxis.grid(True,color=GRID,linewidth=0.8)
    ax.set_axisbelow(True)

os.makedirs('figures',exist_ok=True)

fig,axes=plt.subplots(1,2,figsize=(10,4.2),facecolor=SURFACE,gridspec_kw={'width_ratios':[1,1.6]})

style(axes[0])
names=['CPU','Apple GPU']
seconds=[8.0,4.2]
colors=[MUTED,BLUE]
for i in range(2):
    axes[0].bar([i],[seconds[i]],width=0.5,color=colors[i])
    axes[0].text(i,seconds[i]+0.2,'%.1f s'%seconds[i],color=INK2,fontsize=9,ha='center')
axes[0].set_xticks([0,1])
axes[0].set_xticklabels(names,fontsize=9)
axes[0].set_ylim(0,9.5)
axes[0].set_ylabel('seconds per epoch',color=INK2,fontsize=9)
axes[0].set_title('Speed: the GPU is about 2 times faster',loc='left',fontsize=10,color=INK)

style(axes[1])
tests=['200 species\n(20 orders x 10)','200 random\nspecies','400 random\nspecies']
gpu=[0.086,0.104,None]
cpu=[0.075,0.119,0.082]
for i in range(3):
    if gpu[i] is not None:
        axes[1].scatter([i-0.12],[gpu[i]],s=70,color=BLUE,zorder=3,label='Apple GPU' if i==0 else None)
        axes[1].text(i-0.12,gpu[i]+0.007,'%.3f'%gpu[i],color=INK2,fontsize=8.5,ha='center')
    else:
        axes[1].text(i-0.12,0.012,'GPU ran out\nof memory',color=INK2,fontsize=8,ha='center')
    axes[1].scatter([i+0.12],[cpu[i]],s=70,color=MUTED,zorder=3,label='CPU' if i==0 else None)
    axes[1].text(i+0.12,cpu[i]+0.007,'%.3f'%cpu[i],color=INK2,fontsize=8.5,ha='center')
axes[1].set_xticks([0,1,2])
axes[1].set_xticklabels(tests,fontsize=8.5)
axes[1].set_xlim(-0.6,2.6)
axes[1].set_ylim(0,0.15)
axes[1].set_ylabel('validation top-1 accuracy',color=INK2,fontsize=9)
axes[1].set_title('Accuracy: no clear difference (seed noise is about 0.015)',loc='left',fontsize=10,color=INK)
axes[1].legend(frameon=False,fontsize=8.5,labelcolor=INK2,loc='upper right')

fig.text(0.01,0.01,'Speed: 6,000 photos at 64x64. Accuracy: seed 1, 15 epochs, 30 training photos per species, single runs.',fontsize=8,color=MUTED)
fig.tight_layout(rect=(0,0.04,1,1))
fig.savefig('figures/cpu_vs_gpu.png',dpi=200,facecolor=SURFACE)
plt.close(fig)
print('made 1 chart')
