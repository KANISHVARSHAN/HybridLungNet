#!/bin/bash
# Assemble single-file Kaggle kernels from the shared hln/ modules.
cd "$(dirname "$0")"
H=../hln
mkdir -p k1_prep k2_det k3_cls k4_e2e
cat $H/lidc_io.py src/prep_body.py > k1_prep/prep.py
cat $H/lidc_io.py $H/models.py $H/splits.py $H/detect.py src/det_body.py > k2_det/det.py
cat $H/models.py $H/splits.py src/cls_body.py > k3_cls/cls.py
cat $H/lidc_io.py $H/models.py $H/splits.py $H/detect.py $H/pipeline.py src/e2e_body.py > k4_e2e/e2e.py
for f in k*/*.py; do /c/mini-project/.venv/Scripts/python.exe -c "import ast,sys;ast.parse(open(sys.argv[1],encoding='utf-8').read())" $f || echo "SYNTAX ERROR $f"; done
