#!/bin/bash

uv run main.py --method=netvlad --img_folder='assets/spacecraft/fused' \
--recent_frames_window 100  --no_labels  --log_dir spacecraft --recall_values 10 \
--save_descriptors --save_matched_pairs  --match_distance_threshold 0.7

uv run main.py --method=netvlad --img_folder='assets/spacecraft/fused' \
--recent_frames_window 100  --no_labels  --log_dir spacecraft --recall_values 10 \
--save_descriptors --save_matched_pairs  --match_distance_threshold 0.6

uv run main.py --method=netvlad --img_folder='assets/spacecraft/fused' \
--recent_frames_window 100  --no_labels  --log_dir spacecraft --recall_values 10 \
--save_descriptors --save_matched_pairs  --match_distance_threshold 0.5

uv run main.py --method=netvlad --img_folder='assets/spacecraft/fused' \
--recent_frames_window 100  --no_labels  --log_dir spacecraft --recall_values 10 \
--save_descriptors --save_matched_pairs  --match_distance_threshold 0.4