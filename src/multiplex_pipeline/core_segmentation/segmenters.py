import numpy as np
import torch
from typing import Union, Sequence

class BaseSegmenter:
    def run(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Segmenters must implement the run() method.")

class DummySegmenter(BaseSegmenter):
    def run(self, image: np.ndarray) -> np.ndarray:
        return (image > image.mean()).astype(np.uint8)
    
class CellposeSegmenter(BaseSegmenter):

    def __init__(self, model_type: str = "cyto", **kwargs):
        from cellpose import models
        self.model = models.Cellpose(model_type=model_type, gpu = True)
        self.model_kwargs = kwargs

    def run(self, image: Union[np.ndarray, Sequence[np.ndarray]], **kwargs) -> np.ndarray:
        """
        Run segmentation using Cellpose.
        image: 
            - 2D np.ndarray (single-channel)
            - list of 1 np.ndarray (single-channel)
            - list of 3 np.ndarray (multi-channel, channels kwarg required)
            - 3D np.ndarray with shape (3, H, W) (multi-channel, channels kwarg required)
        kwargs: must include 'channels' for multi-channel.
        """
        # Convert list with 1 or 3 images to numpy array
        if isinstance(image, list):
            if len(image) == 1:
                image = image[0]  # Treat as single-channel
            elif len(image) == 3:
                image = np.stack(image, axis=0)  # (3, H, W)
            else:
                raise ValueError("List input must have 1 or 3 images.")

        # At this point, image is either a 2D array or (3, H, W)
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                # Single channel
                mask, *_ = self.model.eval(image, **self.model_kwargs)
                return mask
            elif image.ndim == 3 and image.shape[0] == 3:
                # Multi-channel: channels kwarg required
                channels = kwargs.get("channels") or self.model_kwargs.get("channels")
                if channels is None:
                    raise ValueError("The 'channels' keyword argument must be specified for multi-channel Cellpose segmentation.")
                mask, *_ = self.model.eval(image, channels=channels, **self.model_kwargs)
                return mask
            else:
                raise ValueError("If passing a 3D array, shape must be (3, H, W) for multi-channel Cellpose.")
        else:
            raise ValueError(
                "Input to CellposeSegmenter.run must be a 2D array, a list of 1 or 3 arrays, "
                "or a 3D array with shape (3, H, W)."
            )
        
class Cellpose4Segmenter(BaseSegmenter):

    def __init__(self, model_type: str = "cyto", **kwargs):
        from cellpose import models
        self.model = models.CellposeModel(gpu = True)
        self.model_kwargs = kwargs

    def run(self, image: Union[np.ndarray, Sequence[np.ndarray]], **kwargs) -> np.ndarray:
        """
        Run segmentation using Cellpose.
        image: 
            - 2D np.ndarray (single-channel)
            - list of 1 np.ndarray (single-channel)
            - list of 3 np.ndarray (multi-channel, channels kwarg required)
            - 3D np.ndarray with shape (3, H, W) (multi-channel, channels kwarg required)
        kwargs: must include 'channels' for multi-channel.
        """
        # Convert list with 1 or 3 images to numpy array
        if isinstance(image, list):
            if len(image) == 1:
                image = image[0]  # Treat as single-channel
            elif len(image) == 3:
                image = np.stack(image, axis=0)  # (3, H, W)
            else:
                raise ValueError("List input must have 1 or 3 images.")

        # At this point, image is either a 2D array or (3, H, W)
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                # Single channel
                mask, *_ = self.model.eval(image, **self.model_kwargs)
                return mask
            elif image.ndim == 3 and image.shape[0] == 3:
                # Multi-channel: channels kwarg required
                channels = kwargs.get("channels") or self.model_kwargs.get("channels")
                if channels is None:
                    raise ValueError("The 'channels' keyword argument must be specified for multi-channel Cellpose segmentation.")
                mask, *_ = self.model.eval(image, channels=channels, **self.model_kwargs)
                return mask
            else:
                raise ValueError("If passing a 3D array, shape must be (3, H, W) for multi-channel Cellpose.")
        else:
            raise ValueError(
                "Input to CellposeSegmenter.run must be a 2D array, a list of 1 or 3 arrays, "
                "or a 3D array with shape (3, H, W)."
            )
        
class InstansegSegmenter(BaseSegmenter):
    
    def __init__(self, model_type, **kwargs):
        """
        model: InstanSeg model instance (e.g., instanseg_fluorescence)
        kwargs: passed to model.eval_small_image as default settings
        """
        from instanseg import InstanSeg

        self.model = InstanSeg(model_type, verbosity=1)
        self.model_kwargs = kwargs

    def run(self, image, **kwargs):
        """
        Segment image using InstanSeg.
        """
        
        # Input normalization
        if isinstance(image, list):
            image = np.stack(image, axis=-1)  # (H, W, C)
        elif isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = image[..., np.newaxis]
            elif image.ndim == 3 and image.shape[0] in {1, 2, 3, 4}:
                # If channel-first, transpose to channel-last
                if image.shape[0] < min(image.shape[1:]):  # (C, H, W)
                    image = np.moveaxis(image, 0, -1)
        else:
            raise ValueError("Input must be a 2D/3D numpy array or a list of 2D arrays.")

        # Call InstanSeg
        labeled_output, _ = self.model.eval_small_image(image, **self.model_kwargs)

        segm_array = [np.array(x).astype(int) for x in labeled_output[0,:,:,:]]

        print(f"alloc={torch.cuda.memory_allocated()/1e9:.2f} GB | "
        f"reserved={torch.cuda.memory_reserved()/1e9:.2f} GB")

        # Clear CUDA cache (optional, can be disabled by kwarg)
        torch.cuda.empty_cache()

        print(f"alloc={torch.cuda.memory_allocated()/1e9:.2f} GB | "
        f"reserved={torch.cuda.memory_reserved()/1e9:.2f} GB")
        
        return segm_array