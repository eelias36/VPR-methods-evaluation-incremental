import parser
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import faiss
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from tqdm import tqdm

import visualizations
import vpr_models
from test_dataset import TestDataset


def main(args):
    start_time = datetime.now()

    logger.remove()  # Remove possibly previously existing loggers
    log_dir = Path("logs") / args.log_dir / start_time.strftime("%Y-%m-%d_%H-%M-%S")
    logger.add(sys.stdout, colorize=True, format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
    logger.add(log_dir / "info.log", format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
    logger.add(log_dir / "debug.log", level="DEBUG")
    logger.info(" ".join(sys.argv))
    logger.info(f"Arguments: {args}")
    logger.info(
        f"Testing with {args.method} with a {args.backbone} backbone and descriptors dimension {args.descriptors_dimension}"
    )
    logger.info(f"The outputs are being saved in {log_dir}")

    model = vpr_models.get_model(args.method, args.backbone, args.descriptors_dimension)
    model = model.eval().to(args.device)

    test_ds = TestDataset(
        args.img_folder,
        angle_threshold=args.angle_threshold,
        image_size=args.image_size,
        use_labels=args.use_labels,
    )
    logger.info(f"Testing on {test_ds}")

    with torch.inference_mode():

        query_descriptor = np.empty((1, args.descriptors_dimension), dtype="float32")

        # Use a kNN to find predictions
        faiss_index = faiss.IndexFlatL2(args.descriptors_dimension)

        predictions = np.empty((test_ds.num_imgs - args.recent_frames_window, max(args.recall_values)), dtype="int64")
        descriptor_queue = []

        logger.info("Finding predictions for each image")

        for frame_number in tqdm(range(test_ds.num_imgs)):
      
            logger.debug(f"Extracting descriptors for query frame #{frame_number} with name {test_ds.images_paths[np.where(test_ds.frame_numbers == frame_number)[0][0]]} using batch size 1")
            queries_subset_ds = Subset(
                test_ds, list([np.where(test_ds.frame_numbers == frame_number)[0][0]])
            )
            query_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers, batch_size=1)
            for images, indices in query_dataloader:
                descriptors = model(images.to(args.device))
                query_descriptor = descriptors.cpu().numpy()

            if args.save_descriptors:
                logger.info(f"Saving the descriptors in {log_dir}")
                np.save(log_dir / "queries_descriptors.npy", query_descriptor)
                np.save(log_dir / "database_descriptors.npy", database_descriptors)

            if frame_number > args.recent_frames_window:
                faiss_index.add(descriptor_queue.pop(0))
                logger.debug(f"Finding matches for query frame {frame_number}")
                _, predictions[frame_number - args.recent_frames_window, :] = faiss_index.search(query_descriptor, max(args.recall_values))
            
            descriptor_queue.append(query_descriptor)

            logger.debug(f"Predictions for query frame {frame_number}: {predictions[frame_number - args.recent_frames_window, :]}")
    
    del database_descriptors
           
    print(predictions)

    # For each query, check if the predictions are correct
    if args.use_labels:
        positives_per_query = test_ds.get_positives()
        recalls = np.zeros(len(args.recall_values))
        for query_index, preds in enumerate(predictions):
            for i, n in enumerate(args.recall_values):
                if np.any(np.isin(preds[:n], positives_per_query[query_index])):
                    recalls[i:] += 1
                    break

        # Divide by num_queries and multiply by 100, so the recalls are in percentages
        recalls = recalls / test_ds.num_queries * 100
        recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])
        logger.info(recalls_str)

    # Save visualizations of predictions
    if args.num_preds_to_save != 0:
        logger.info("Saving final predictions")
        # For each query save num_preds_to_save predictions
        visualizations.save_preds(
            predictions[:, : args.num_preds_to_save], test_ds, log_dir, args.save_only_wrong_preds, args.use_labels, args.recent_frames_window
        )


if __name__ == "__main__":
    args = parser.parse_arguments()
    main(args)
