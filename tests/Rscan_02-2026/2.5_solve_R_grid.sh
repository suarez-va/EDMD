#!/bin/bash

cd RI
for Rdir in R*; do
  if [ ! -d "$Rdir" ]; then
    continue
  fi
  cd "$Rdir"
  while true; do
    job_assigned=false
    if [ -f "bospec.npz" ]; then
      echo "bospec.npz exists — skipping $etadir"
      break
    fi
    for core in {0..13}; do
      if ! ps -eo pid,psr,comm | awk -v c="$core" '$2 == c && $3 ~ /^python/' | grep -q .; then
        echo "Running in $(pwd) on core $core"
        taskset -c $core python template_R.py &
        job_assigned=true
        sleep 0.05
        break
      fi
    done
    if [ "$job_assigned" = true ]; then
      break
    else
      echo "All cores busy, wait a little and try again..."
      sleep 1
    fi
  done
  cd ../
done
cd ../

#cd RI
#
#for dir in R*; do
#  if [ -d "$dir" ]; then
#    cd $dir
#    pwd
#    if [ -f "eigspec.npz" ]; then
#      echo "eigspec.npz exists — skipping $dir"
#      cd ..
#      continue
#    fi
#    core=$((RANDOM % 14)) # random integer: 0–13
#    taskset -c "$core" python template_R.py
#    cd ../
#  fi
#done
#
#cd ../
