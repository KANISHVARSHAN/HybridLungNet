#!/bin/bash
# dl_loop.sh slug outdir : keep retrying until no partial files remain and at least one .pth exists
K=/c/mini-project/.venv/Scripts/kaggle.exe
mkdir -p "$2"
for i in $(seq 1 60); do
  timeout 900 $K kernels output "$1" -p "$2" >/dev/null 2>&1
  if ! ls "$2"/*.kaggle-partial >/dev/null 2>&1 && ls "$2"/*.pth >/dev/null 2>&1; then echo "OK after $i"; ls -la "$2"; exit 0; fi
  rm -f "$2"/*.kaggle-partial; sleep 30
done
echo FAILED
