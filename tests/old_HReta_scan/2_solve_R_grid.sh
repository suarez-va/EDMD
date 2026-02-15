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
    sbatch ../../submit_solve_R_grid.slm
    cd ../
  fi
done

cd ../

