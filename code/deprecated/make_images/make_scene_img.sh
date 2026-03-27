#!/bin/bash

# Loop through worlds (w1 to w8)
for world in {1..8}; do
  # Loop through levels (l1 to l4)
  for level in {1..4}; do
    label="w${world}l${level}"
    echo "Processing ${label}..."
    python make_scene_img.py -s all -l "${label}" || echo "Command failed for ${label}, skipping..."
  done
done
