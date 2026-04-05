uv run main.py --method=eigenplaces --img_folder='../RSO-place-recognition/data/renders/tir' --recent_frames_window 100  --no_labels --log_dir spacecraft --recall_values 10 --save_descriptors --save_matched_pairs

 --num_preds_to_save 3

 uv run main.py --method=netvlad --img_folder='assets/spacecraft/tir_8bit' --rece
nt_frames_window 100  --no_labels  --log_dir spacecraft --recall_values 10 --save_descriptors --save_matched_pairs  --num_preds_to_save 5 --match_distance_threshold 0.4