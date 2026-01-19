#!/bin/bash

cd etal || exit 1

for etadir in eta*/; do
  [ -d "$etadir" ] || continue

  if [ -f "$etadir/capdata.npz" ]; then
    echo "capdata.npz exists — skipping $etadir"
    continue
  fi

  echo "Running in $etadir"
  (cd "$etadir" && python template_eta.py)
done

