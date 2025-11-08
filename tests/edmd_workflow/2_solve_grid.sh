#!/bin/bash

cd Rk

for dir in R*; do
    if [ -d "$dir" ]; then
        cd $dir
        pwd
        python template.py
        cd ../
    fi
done

cd ../

