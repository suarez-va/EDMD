#!/bin/bash

cd etan

for dir in eta*; do
  if [ -d "$dir" ]; then
    cd $dir
    pwd
    #taskset -c 0 python template.py
    sbatch ../../submit_template.slm
    cd ../
  fi
done

cd ../
