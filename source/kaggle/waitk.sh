#!/bin/bash
K=/c/mini-project/.venv/Scripts/kaggle.exe
while true; do
  s=$($K kernels status "$1" 2>&1)
  if echo "$s" | grep -qE 'status "?(KernelWorkerStatus\.)?(COMPLETE|ERROR|CANCEL)|has status "(complete|error|cancel)'; then echo "$s"; break; fi
  echo "$s" | grep -qiE 'has status' && sleep 30 || sleep 20
done
