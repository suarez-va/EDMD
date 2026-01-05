#!/bin/bash

cd etan

for dir in eta*; do
  if [ -d "$dir" ]; then
    cd $dir
    pwd
    #sbatch ../../submit_template.slm
    taskset -c 8 python template.py
    cd ../
  fi
done

cd ../
