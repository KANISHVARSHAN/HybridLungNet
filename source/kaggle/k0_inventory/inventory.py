import os, collections, json, time, subprocess
t=time.time()
for r in ['/kaggle/input']:
    for d in sorted(os.listdir(r)):
        print('INPUT', d, os.listdir(os.path.join(r,d))[:10])
# locate roots
lidc=None; patches=None
for root,dirs,files in os.walk('/kaggle/input'):
    if any(x.startswith('LIDC-IDRI-0') and len(x)==14 for x in dirs) and lidc is None and 'slices' not in root:
        print('PATIENT ROOT', root, len(dirs))
    if 'final_dataset.csv' in files: print('CSV', root, files[:5])
    depth=root.count('/')
    if depth>7: dirs[:]=[]
pats=collections.Counter(); series=0; xmls=0; nslices=[]; modal=collections.Counter()
import pydicom
roots=[]
for root,dirs,files in os.walk('/kaggle/input/lidcidri30' if os.path.exists('/kaggle/input/lidcidri30') else '/kaggle/input'):
    dcm=[f for f in files if f.endswith('.dcm')]
    if dcm:
        series+=1; nslices.append(len(dcm))
        pid=[p for p in root.split('/') if p.startswith('LIDC-IDRI-') and len(p)==14]
        if pid: pats[pid[0]]+=1
        if series<=400 or series%50==0:
            try: modal[pydicom.dcmread(os.path.join(root,dcm[0]),stop_before_pixels=True).Modality]+=1
            except Exception as e: modal['err']+=1
    xmls+=sum(f.endswith('.xml') for f in files)
print('patients',len(pats),'series',series,'xmls',xmls,'modality sample',modal)
print('patient range',min(pats),max(pats))
import numpy as np; print('slices per series pct',np.percentile(nslices,[0,10,50,90,100]))
print('seriesperpatient',collections.Counter(pats.values()))
for m in ['pydicom','SimpleITK','skimage','scipy','torch','sklearn']:
    try: __import__(m); print('have',m)
    except Exception as e: print('MISSING',m)
print(subprocess.run('df -h /kaggle/working /tmp; nproc; free -g',shell=True,capture_output=True,text=True).stdout)
print('time',time.time()-t)
