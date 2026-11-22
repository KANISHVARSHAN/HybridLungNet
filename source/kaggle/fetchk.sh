#!/bin/bash
# fetchk.sh slug outdir : download output with retries, print log tail
K=/c/mini-project/.venv/Scripts/kaggle.exe
mkdir -p "$2"
for i in 1 2 3 4 5; do $K kernels output "$1" -p "$2" >/dev/null 2>&1 && break; sleep 10; done
/c/mini-project/.venv/Scripts/python.exe - "$2" <<'PY'
import sys,json,glob
for f in glob.glob(sys.argv[1]+'/*.log'):
    try: print(''.join(e.get('data','') for e in json.load(open(f,encoding='utf-8')))[-int(6000):])
    except Exception as ex: print(open(f,encoding='utf-8',errors='replace').read()[-6000:])
PY
