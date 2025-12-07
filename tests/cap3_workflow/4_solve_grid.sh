#!/bin/bash

cd etan

for dir in eta*; do
  if [ -d "$dir" ]; then
    cd $dir
    pwd
    sbatch ../../submit_template.slm
    cd ../
  fi
done

cd ../
