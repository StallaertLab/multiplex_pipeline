"""CLI utility for segmenting images with SAM2 and saving core suggestions."""

import argparse
import os
import pickle as pkl
import time
from datetime import datetime

import torch
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2

from multiplex_pipeline.utils.im_utils import prepare_rgb_image


def set_cuda(model_path):
    """Configure CUDA settings for SAM2 segmentation.

    Args:
        model_path (str): Path to the directory with the SAM2 model.

    Returns:
        tuple[torch.device, str, str]: CUDA device, checkpoint path and model configuration path.
    """
    assert torch.cuda.is_available()

    device = torch.device("cuda")
    sam2_checkpoint = "./sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    return device, sam2_checkpoint, model_cfg


def sam2_segment(
    im_rgb,
    build_sam2,
    SAM2AutomaticMaskGenerator,
    device,
    sam2_checkpoint,
    model_cfg,
):
    """Segment an RGB image using the SAM2 model.

    Args:
        im_rgb (numpy.ndarray): RGB image to segment.
        build_sam2 (Callable): Function that builds a SAM2 model instance.
        SAM2AutomaticMaskGenerator (Callable): Mask generator class.
        device (torch.device): CUDA device used for inference.
        sam2_checkpoint (str): Path to the SAM2 checkpoint file.
        model_cfg (str): Path to the SAM2 configuration file.

    Returns:
        list[dict]: Segmentation masks for the input image.
    """

    # clear the cache
    torch.cuda.empty_cache()
    time.sleep(5)

    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    sam2 = build_sam2(
        model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False
    )

    mask_generator = SAM2AutomaticMaskGenerator(
        sam2,
        points_per_side=72,
        stability_score_thresh=0.9,
        pred_iou_thresh=0.8,
        use_m2m=False,
        crop_n_points_downscale_factor=8,
        crop_n_layers=0,
        box_nms_thresh=0.5,
    )

    with torch.no_grad():
        masks = mask_generator.generate(im_rgb)

    # clear the cache
    torch.cuda.empty_cache()

    return masks


def main():

    # Create the argument parser
    parser = argparse.ArgumentParser()

    # Compulsory arguments
    parser.add_argument("im", type=str, help="Path to the image for segmentation.")
    parser.add_argument(
        "req_level", type=int, help="Requested resolution level of the image."
    )
    parser.add_argument("model", type=str, help="Path to sam2 model.")
    parser.add_argument("output", type=str, help="Path to the output file.")

    # Parse the arguments
    args = parser.parse_args()

    # change the working directory to the model path
    os.chdir(args.model)

    # inform about the input and output pathways
    print(f"Input image: {args.im}")
    print(f"Results will be saved to: {args.output}\n")

    # preapare image for segmentation
    print("Preparing RGB image for segmentation...")
    im_rgb = prepare_rgb_image(args.im, req_level=int(args.req_level))

    # set the cuda environment
    device, sam2_checkpoint, model_cfg = set_cuda(args.model)

    # segment image
    print(
        f'Segmenting image. It should take around 1 min. Started at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}...'
    )
    masks = sam2_segment(
        im_rgb,
        build_sam2,
        SAM2AutomaticMaskGenerator,
        device,
        sam2_checkpoint,
        model_cfg,
    )

    # save the masks
    print("Saving masks...")
    with open(args.output, "wb") as f:
        pkl.dump(masks, f)


if __name__ == "__main__":

    main()
