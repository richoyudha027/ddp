#!/bin/bash

OUT="gpu_monitor_$(date +%Y%m%d_%H%M%S).csv"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,clocks.sm,temperature.gpu,power.draw,memory.used \
  --format=csv,nounits -l 5 >> "$OUT"
