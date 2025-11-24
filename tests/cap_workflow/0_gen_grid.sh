#!/bin/bash

mkdir eta_traj
cd eta_traj
for n in {0..10}; do
  mkdir eta$n
  cd eta$n
  cp ../../template.py .

  echo "$n"
done
