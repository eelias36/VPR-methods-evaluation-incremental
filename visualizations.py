from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import torch
from PIL import Image, ImageOps
import torchvision.transforms as tfm
import csv

# Height and width of a single image for visualization
IMG_HW = 512
TEXT_H = 175
FONTSIZE = 50
SPACE = 50  # Space between two images

def write_labels_to_image(labels=["text1", "text2"]):
    """Creates an image with text"""
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", FONTSIZE)
    img = Image.new("RGB", ((IMG_HW * len(labels)) + 50 * (len(labels) - 1), TEXT_H), (1, 1, 1))
    d = ImageDraw.Draw(img)
    for i, text in enumerate(labels):
        _, _, w, h = d.textbbox((0, 0), text, font=font)
        d.text(((IMG_HW + SPACE) * i + IMG_HW // 2 - w // 2, 1), text, fill=(0, 0, 0), font=font)
    return Image.fromarray(np.array(img)[:100] * 255)  # Remove some empty space


def draw_box(img, c=(0, 1, 0), thickness=20):
    """Draw a colored box around an image. Image should be a PIL.Image."""
    assert isinstance(img, Image.Image)
    img = tfm.ToTensor()(img)
    assert len(img.shape) >= 2, f"{img.shape=}"
    c = torch.tensor(c).type(torch.float).reshape(3, 1, 1)
    img[..., :thickness, :] = c
    img[..., -thickness:, :] = c
    img[..., :, -thickness:] = c
    img[..., :, :thickness] = c
    return tfm.ToPILImage()(img)

def save_matching_pairs(predictions, eval_ds, log_dir, recent_frames_window=0):
    """ Save a csv file containing the query and prediction for all matches."""
    with open(log_dir / "matching_pairs.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["query_path", "pred_path"])
        for query_index, preds in enumerate(tqdm(predictions, desc=f"Saving matches in {log_dir}")):
            query_path = eval_ds.images_paths[query_index + recent_frames_window + 1]
            for pred in preds:
                if pred > -1:
                    pred_path = eval_ds.images_paths[pred]
                    writer.writerow([Path(query_path).stem, Path(pred_path).stem])


def build_prediction_image(images_paths, preds_correct):
    """Build a row of images, where the first is the query and the rest are predictions.
    For each image, if is_correct then draw a green/red box.
    """
    assert len(images_paths) == len(preds_correct)
    labels = ["Query"]
    for i, is_correct in enumerate(preds_correct[1:]):
        if is_correct is None:
            labels.append(f"Pred{i}")
        else:
            labels.append(f"Pred{i} - {is_correct}")

    images = [Image.open(path) for path in images_paths]
    # Convert from 16-bit to 8-bit if needed
    images = [img.point(lambda x: x / 256).convert("L").convert("RGB") if img.mode == "I;16" else img.convert("RGB") for img in images]
    for img_idx, (img, is_correct) in enumerate(zip(images, preds_correct)):
        if is_correct is None:
            continue
        color = (0, 1, 0) if is_correct else (1, 0, 0)
        img = draw_box(img, color)
        images[img_idx] = img

    resized_images = [tfm.Resize(510, max_size=IMG_HW, antialias=True)(img) for img in images]
    resized_images = [ImageOps.pad(img, (IMG_HW, IMG_HW), color='white') for img in images]  # Apply padding to make them squared

    total_h = len(resized_images)*IMG_HW + max(0,len(resized_images)-1)*SPACE # 2
    concat_image = Image.new('RGB', (total_h, IMG_HW), (255, 255, 255))
    y=0
    for img in resized_images:
        concat_image.paste(img, (y, 0))
        y += IMG_HW + SPACE

    try:
        labels_image = write_labels_to_image(labels)
        # Transform the images to np arrays for concatenation
        final_image = Image.fromarray(np.concatenate((np.array(labels_image), np.array(concat_image)), axis=0))
    except OSError:  # Handle error in case of missing PIL ImageFont
        final_image = concat_image

    return final_image


def save_file_with_paths(query_path, preds_paths, positives_paths, output_path, use_labels=True):
    file_content = []
    file_content.append("Query path:")
    file_content.append(query_path + "\n")
    file_content.append("Predictions paths:")
    file_content.append("\n".join(preds_paths) + "\n")
    if use_labels:
        file_content.append("Positives paths:")
        file_content.append("\n".join(positives_paths) + "\n")
    with open(output_path, "w") as file:
        _ = file.write("\n".join(file_content))


def save_preds(predictions, eval_ds, log_dir, save_only_wrong_preds=None, use_labels=True, recent_frames_window=0):
    """For each query, save an image containing the query and its predictions,
    and a file with the paths of the query, its predictions and its positives.

    Parameters
    ----------
    predictions : np.array of shape [num_queries x num_preds_to_viz], with the preds
        for each query
    eval_ds : TestDataset
    log_dir : Path with the path to save the predictions
    save_only_wrong_preds : bool, if True save only the wrongly predicted queries,
        i.e. the ones where the first pred is uncorrect (further than 25 m)
    """
    if use_labels:
        positives_per_query = eval_ds.get_positives()

    viz_dir = log_dir / "preds"
    viz_dir.mkdir()
    for query_index, preds in enumerate(tqdm(predictions, desc=f"Saving preds in {viz_dir}")):
        query_path = eval_ds.images_paths[query_index + recent_frames_window + 1]
        list_of_images_paths = [query_path]
        # List of None (query), True (correct preds) or False (wrong preds)
        preds_correct = [None]
        for pred_index, pred in enumerate(preds):
            if pred > -1:
                pred_path = eval_ds.images_paths[pred]
                list_of_images_paths.append(pred_path)
                if use_labels:
                    is_correct = pred in positives_per_query[query_index]
                else:
                    is_correct = None
                preds_correct.append(is_correct)

        if save_only_wrong_preds and preds_correct[1]:
            continue

        prediction_image = build_prediction_image(list_of_images_paths, preds_correct)
        pred_image_path = viz_dir / f"{Path(query_path).stem}.jpg"
        prediction_image.save(pred_image_path)

        if use_labels:
            positives_paths = [eval_ds.images_paths[idx] for idx in positives_per_query[query_index]]
        else:
            positives_paths = None

        save_file_with_paths(
            query_path=list_of_images_paths[0],
            preds_paths=list_of_images_paths[1:],
            positives_paths=positives_paths,
            output_path=viz_dir / f"{Path(query_path).stem}.txt",
            use_labels=use_labels,
        )
