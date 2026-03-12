#!/bin/bash

cd Ln

for dir in L*; do
  if [ -d "$dir" ]; then
    cd $dir
    pwd
    if [ -f "fcidvr_2ele_singlet.npz" ]; then
      echo "skipping $dir"
      cd ..
      continue
    fi
    sbatch ../../submit_solve_L_grid.slm
    cd ../
  fi
done

cd ../

