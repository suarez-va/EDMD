#!/bin/bash

cd Rk

for Rdir in R*; do
  if [ -d "$Rdir" ]; then
    cd $Rdir
    pwd
    if [ ! -f "eigspec.npz" ]; then
      echo "eigspec.npz does not exist — skipping $Rdir"
      cd ..
      continue
    fi
    cd etal
    for etadir in eta*; do
      if [ ! -d "$etadir" ]; then
        continue
      fi
      cd "$etadir"
      while true; do
        job_assigned=false
        if [ -f "capdata.npz" ]; then
          echo "capdata.npz exists — skipping $etadir"
          break
        fi
        for core in {0..13}; do
          if ! ps -eo pid,psr,comm | awk -v c="$core" '$2 == c && $3 ~ /^python/' | grep -q .; then
            echo "Running in $(pwd) on core $core"
            taskset -c $core python template_eta.py &
            job_assigned=true
            sleep 0.1
            break
          fi
        done
        if [ "$job_assigned" = true ]; then
          break
        else
          echo "All cores busy, wait a little and try again..."
          sleep 5
        fi
      done
      cd ../
    done
    cd ../
    cd ../
    pwd
  fi
done
cd ../
