#!/bin/bash

cd Rk

for dir in R*; do
  if [ -d "$dir" ]; then
    cd $dir
    pwd
    if [ -f "eigspec.npz" ]; then
      echo "eigspec.npz exists — skipping $dir"
      cd ..
      continue
    fi
    core=$((RANDOM % 14)) # random integer: 0–13
    taskset -c "$core" python template_R.py
    cd ../
  fi
done

cd ../
