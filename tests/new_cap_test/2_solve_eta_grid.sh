#!/bin/bash

cd etai

for etadir in eta*; do
  if [ ! -d "$etadir" ]; then
    continue
  fi

  cd "$etadir"

  while true; do
    job_assigned=false
    for core in {0..13}; do
      if ! ps -eo pid,psr,comm | awk -v c="$core" '$2 == c && $3 ~ /^python/' | grep -q .; then
        echo "Running in $(pwd) on core $core"
        taskset -c $core python template_eta.py &
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
