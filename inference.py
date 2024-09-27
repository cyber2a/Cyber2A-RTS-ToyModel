import argparse
import os

import cv2
import torch
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

from toy_model.model import get_model_instance_segmentation
from toy_model.transforms import get_transform


def load_model(model_path, num_classes, device):
    model = get_model_instance_segmentation(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path, transform, device):
    image = read_image(image_path)
    image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(
        torch.uint8
    )
    image = image[:3, ...]
    transformed_image = transform(image)
    return image, transformed_image[:3, ...].to(device)


def get_predictions(model, image):
    with torch.no_grad():
        predictions = model([image])
    return predictions[0]


def filter_predictions(pred, score_threshold=0.5):
    keep = pred["scores"] > score_threshold
    return {k: v[keep] for k, v in pred.items()}


def draw_predictions(image, pred):
    pred_labels = [f"RTS: {score:.3f}" for score in pred["scores"]]
    masks = (pred["masks"] > 0.7).squeeze(1)
    output_image = draw_segmentation_masks(image, masks, alpha=0.5, colors="red")
    pred_boxes = pred["boxes"].long()
    output_image = draw_bounding_boxes(
        output_image, pred_boxes, pred_labels, colors="black", width=0
    )
    return output_image


def save_image(image, output_path):
    image = image.permute(1, 2, 0).numpy()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, image)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("rts_model.pth", num_classes=2, device=device)
    image, transformed_image = preprocess_image(
        args.image_path, get_transform(train=False), device
    )
    pred = get_predictions(model, transformed_image)
    pred = filter_predictions(pred)
    output_image = draw_predictions(image, pred)
    base_name = os.path.basename(args.image_path)
    output_path = "pred_" + os.path.splitext(base_name)[0] + ".png"
    save_image(output_image, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on an image.")
    parser.add_argument("--image-path", required=True, help="Path to the input image.")
    args = parser.parse_args()
    main(args)
