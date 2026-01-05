#!/bin/bash

cd etan

for dir in eta*; do
  if [ ! -d "$dir" ]; then
    continue
  fi

  cd "$dir"
  while true; do
    job_assigned=false
    for core in {0..13}; do
      if ! ps -eo pid,psr,comm | awk -v c="$core" '$2 == c && $3 ~ /^python/' | grep -q .; then
        echo "Running in $(pwd) on core $core"
        taskset -c $core python template.py &
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
  cd ..
done
