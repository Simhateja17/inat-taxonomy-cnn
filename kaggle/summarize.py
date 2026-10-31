import sys
prefix=sys.argv[1]
cfgs=[]
n_seeds=5
args=sys.argv[2:]
i=0
while i<len(args):
    if args[i]=='--seeds':
        n_seeds=int(args[i+1])
        i=i+2
    else:
        cfgs.append(args[i])
        i=i+1

for cfg in cfgs:
    print('==',cfg)
    rows=[]
    for s in range(1,n_seeds+1):
        f='kaggle/results/%s_%s_s%d.txt'%(prefix,cfg,s)
        try:
            lines=open(f).read().split('\n')
        except Exception:
            print(s,'missing file')
            continue
        r={}
        for ln in lines:
            p=ln.split()
            if len(p)>=5 and p[0]=='test':
                r['t1']=float(p[2])
                r['t5']=float(p[4])
            if len(p)>=5 and p[0]=='val' and p[1]=='top1':
                r['v1']=float(p[2])
            if len(p)>=5 and p[0]=='train' and p[1]=='top1':
                r['tr']=float(p[2])
            if p and p[0]=='sev' and p[1]=='test':
                wrong=int(p[3])
                r['wrong']=wrong
                r['cls']=int(p[5])/wrong
                r['ord']=int(p[7])/wrong
                r['fam']=int(p[9])/wrong
        if 'cls' in r and 'tr' in r:
            rows.append(r)
            print(s,'train %.3f val %.4f test %.4f top5 %.4f | right class %.3f order %.3f family %.3f'%(r['tr'],r['v1'],r['t1'],r['t5'],r['cls'],r['ord'],r['fam']))
        else:
            print(s,'not finished')
    if len(rows)>0:
        n=len(rows)
        means={}
        for k in ['tr','v1','t1','t5','cls','ord','fam']:
            total=0
            for r in rows:
                total=total+r[k]
            means[k]=total/n
        print('mean  train %.3f val %.4f test %.4f top5 %.4f | class %.3f order %.3f family %.3f (n=%d)'%(means['tr'],means['v1'],means['t1'],means['t5'],means['cls'],means['ord'],means['fam'],n))
        for k in ['t1','t5','cls','ord','fam']:
            vals=[]
            for r in rows:
                vals.append(r[k])
            print('   range',k,'%.3f - %.3f'%(min(vals),max(vals)))
