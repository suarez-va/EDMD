#!/bin/bash

cd etan

for dir in eta*; do
  if [ -d "$dir" ]; then
    cd $dir
    pwd
    taskset -c 0 python template.py
    cd ../
  fi
done

cd ../
