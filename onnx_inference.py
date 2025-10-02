import numpy as np
import os
from empatches import EMPatches
from PIL import Image
import albumentations as albu
import onnx
import onnxruntime as ort
from tqdm import tqdm

LOW_CLIP_VALUE = 1e-8
HIGH_CLIP_VALUE = 1000
DEPTH = "depth"
MASK = "mask"
IMAGE = "image"


def make_prediction_dpt(ort_session: ort.InferenceSession, img: np.array) -> np.array:
    """
    Make depth predictions using the Model.

    Args:
        ort_session (ort.InferenceSession): The ONNX Inference session.
        img (np.array): The input image array.

    Returns:
        np.array: The predicted depth array.
    """
    # Ensure img has 4 dimensions (batch size, channels, height, width)
    if img.shape[1:] != (384, 384):
        img_to_input = (
            np.transpose(
                np.array(
                    Image.fromarray((img).astype(np.uint8)).resize(
                        ((384, 384)), Image.BICUBIC
                    ),
                    dtype=np.float32,
                ),
                (2, 0, 1),
            )
            / 255.0
        )
    else:
        img_to_input = img
    if img_to_input.ndim != 4:
        img_to_input = np.expand_dims(img_to_input, axis=0)
    else:
        img_to_input = np.transpose(img_to_input, (0, 3, 1, 2))

    disparity_image = ort_session.run(None, {"pixel_values": img_to_input})[0]
    disparity_image = disparity_image.reshape(
        disparity_image.shape[0], 1, disparity_image.shape[1], disparity_image.shape[2]
    )
    ans = np.array(
        Image.fromarray(disparity_image[0, 0, :, :]).resize(
            img.shape[:2], Image.BICUBIC
        )
    )

    # Clip the values
    return np.clip(ans, LOW_CLIP_VALUE, HIGH_CLIP_VALUE)


def get_preprocess(dataset_name=None) -> albu.Compose:
    """
    Returns:
        Compose: The composed transformation pipeline.
    """
    if dataset_name == "DFC2018":
        mean = (0.47201676, 0.32118613, 0.3189363)
        std = (0.21518334, 0.15455843, 0.14943491)
    elif dataset_name == "Vaihingen":
        mean = (0.4643, 0.3185, 0.3141)
        std = (0.2171, 0.1561, 0.1496)
    elif dataset_name == "Tagil":
        mean = (0.51882998, 0.51636622, 0.40003074)
        std = (0.25592191, 0.22129431, 0.1813154)
    else:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    return albu.Normalize(mean=mean, std=std)


def predict_large_depth_map(img, ort_session, patch_size=384, overlap_size=0.5):
    """
    Make depth predictions of large Satellite by the Model.

    Args:
        img (np.array): Normalized Satellite Image.
        ort_session (ort.InferenceSession): The ONNX Inference session.
        patch_size (int): size of crop patch.
        overlap_size (float): crop overlapping coefficient

    Returns:
        np.array: The predicted depth array.
    """
    emp = EMPatches()

    if img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))

    img_patches, indices = emp.extract_patches(
        img, patchsize=patch_size, overlap=overlap_size
    )
    preds = []

    for img in tqdm(img_patches):
        pred = make_prediction_dpt(ort_session, img)
        preds.append(pred.squeeze())

    merged_img = emp.merge_patches(preds, indices, mode="max")
    return merged_img


def load_and_preprocess_dfc_image(rgb_image, datasetname="Tagil"):
    """
    Load and preprocess DFC image for prediction by DPT.

    Args:
        rgb_image (str): Path to load the image.

    Returns:
        np.array: The image array.
    """

    # img = Image.open(rgb_image).convert("RGB")
    img = np.array(img, dtype=np.float32)
    preprocess = get_preprocess(datasetname)
    img = preprocess(image=img)["image"]


    return img


def load_onnx_model(
    path_to_load_onnx_model: str = "../shared_data/SatelliteTo3D-Models/dpt/dpt_tagil.onnx",
) -> ort.InferenceSession:
    onnx_model = onnx.load(path_to_load_onnx_model)
    onnx.checker.check_model(onnx_model)
    ort_sess = ort.InferenceSession(path_to_load_onnx_model)
    return ort_sess
