"""Inference helpers: run the model and decode its token sequences into room polygons."""

import numpy as np
import torch
from shapely.geometry import Polygon

from datasets.discrete_tokenizer import TokenType


def _process_predictions(
    pred_corners,
    i,
    semantic_rich,
    poly2seq,
    fg_mask,
    image_size,
    pred_room_label,
    pred_room_logits,
    add_cls_token,
    per_token_sem_loss,
    wd_as_line,
    door_window_index,
    dataset_name,
):
    """
    Processes predictions for room layouts, extracting polygons for rooms, windows, and doors.
    This function handles two main modes: non-poly2seq (mask-based) and poly2seq (sequence-based).
    It filters and validates predicted corners based on various conditions, computes room types
    and window/door types if semantic information is rich, and ensures polygons meet area thresholds.

    Args:
        pred_corners (list or tensor): Predicted corner coordinates for scenes or sequences.
        i (int): Index of the current scene in the batch.
        semantic_rich (bool): Whether to include semantic information (room types, window/door types).
        poly2seq (bool): Whether the predictions are in sequence format (poly2seq mode).
        fg_mask (tensor): Foreground masks for valid corners per room (used in non-poly2seq mode).
        image_size (int): Size of the image (used for scaling coordinates).
        pred_room_label (tensor): Predicted room labels for the scene.
        pred_room_logits (tensor): Logits for room predictions.
        add_cls_token (int): Number of CLS tokens added in sequences (used in poly2seq mode).
        per_token_sem_loss (bool): Whether to compute semantic loss per token (affects class aggregation).
        wd_as_line (bool): Whether to treat windows/doors as lines (affects classification logic).
        door_window_index (list or dict): Indices or mapping for door/window classes.
        dataset_name (str): Name of the dataset being processed.

    Returns:
        dict: A dictionary containing:
            - "room_polys" (list of np.ndarray): List of valid room polygons (scaled and rounded).
            - "room_types" (list or None): List of room types if semantic_rich, else None.
            - "window_doors" (list or None): List of window/door polygons if semantic_rich, else None.
            - "window_doors_types" (list or None): List of window/door types if semantic_rich, else None.
            - "pred_room_label_per_scene" (list): Processed room labels for the scene.
    """

    np_softmax = lambda x: np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
    pred_corners_per_scene = pred_corners[i]
    room_polys = []

    if semantic_rich:
        room_types = []
        window_doors = []
        window_doors_types = []

        pred_room_label_per_scene = pred_room_label[i].cpu().numpy()
        pred_room_logit_per_scene = pred_room_logits[i].cpu().numpy()

    # Process predictions based on mode
    if not poly2seq:
        fg_mask_per_scene = fg_mask[i]

        # process per room
        for j in range(fg_mask_per_scene.shape[0]):
            fg_mask_per_room = fg_mask_per_scene[j]
            pred_corners_per_room = pred_corners_per_scene[j]
            valid_corners_per_room = pred_corners_per_room[fg_mask_per_room]
            if len(valid_corners_per_room) > 0:
                corners = (valid_corners_per_room * (image_size - 1)).cpu().numpy()
                corners = np.around(corners).astype(np.int32)

                if not semantic_rich:
                    # only regular rooms
                    if len(corners) >= 4 and Polygon(corners).area >= 100:
                        room_polys.append(corners)
                else:
                    # regular rooms
                    if len(corners) >= 3 and Polygon(corners).area >= 100:
                        room_polys.append(corners)
                        room_types.append(pred_room_label_per_scene[j])
                    # window / door
                    elif len(corners) == 2:
                        window_doors.append(corners)
                        window_doors_types.append(pred_room_label_per_scene[j])

        if not semantic_rich:
            pred_room_label_per_scene = len(room_polys) * [-1]
    else:
        # poly2seq mode: sequence-based processing
        all_room_polys = []
        tmp = []
        all_length_list = [0]

        for j in range(len(pred_corners_per_scene)):
            token = pred_corners_per_scene[j]
            if isinstance(token, int):
                # Handle separator, end-of-sequence, or class tokens
                # New format: sep=1, eos=2, cls=3 (following TokenType)
                # Old format: sep=2, eos=-1, cls=-1
                if (
                    token == TokenType.sep.value
                    or token == TokenType.eos.value
                    or token == TokenType.cls.value
                    or token == -1
                ):
                    if tmp:
                        all_room_polys.append(tmp)
                        all_length_list.append(len(tmp) + 1 + add_cls_token)
                        tmp = []
                
                if token == -1 or token == TokenType.eos.value:
                    break
                continue
            tmp.append(token)

        if len(tmp):
            all_room_polys.append(tmp)
            all_length_list.append(len(tmp) + 1 + add_cls_token)

        start_poly_indices = np.cumsum(all_length_list)

        final_pred_classes = []
        for j, poly in enumerate(all_room_polys):
            if len(poly) < 2:
                continue
            corners = np.array(poly, dtype=np.float32) * (image_size - 1)
            corners = np.around(corners).astype(np.int32)

            if not semantic_rich:
                # only regular rooms
                if len(corners) >= 4 and Polygon(corners).area >= 100:
                    room_polys.append(corners)
            else:
                if per_token_sem_loss:
                    pred_classes, counts = np.unique(
                        pred_room_label_per_scene[start_poly_indices[j] : start_poly_indices[j + 1]][:-1],
                        return_counts=True,
                    )
                    pred_class = pred_classes[np.argmax(counts)]
                    pred_logit = pred_room_logit_per_scene[start_poly_indices[j] : start_poly_indices[j + 1]][:-1]
                else:
                    pred_class = pred_room_label_per_scene[
                        start_poly_indices[j + 1] - 1
                    ]  # get last cls token in the seq
                final_pred_classes.append(pred_class)

                if wd_as_line:
                    # regular rooms
                    if len(corners) >= 3 and Polygon(corners).area >= 100:
                        room_polys.append(corners)
                        room_types.append(pred_class)
                    # window / door
                    elif len(corners) == 2:
                        window_doors.append(corners)
                        if (
                            door_window_index is not None
                            and pred_class not in door_window_index
                            and dataset_name != "r2g"
                        ):
                            wd_prob = np_softmax(pred_logit[:, door_window_index].sum(0))
                            pred_class = door_window_index[wd_prob.argmax()]
                        window_doors_types.append(pred_class)
                else:
                    # regular rooms
                    if door_window_index is not None and pred_class not in door_window_index:
                        room_polys.append(corners)
                        room_types.append(pred_class)
                    else:
                        window_doors.append(corners)
                        window_doors_types.append(pred_class)

        if not semantic_rich:
            pred_room_label_per_scene = len(all_room_polys) * [-1]
        else:
            pred_room_label_per_scene = final_pred_classes

    return {
        "room_polys": room_polys,
        "room_types": room_types if semantic_rich else None,
        "window_doors": window_doors if semantic_rich else None,
        "window_doors_types": window_doors_types if semantic_rich else None,
        "pred_room_label_per_scene": pred_room_label_per_scene,
    }


def generate(
    model,
    samples,
    semantic_rich=False,
    poly2seq: bool = False,
    use_cache=True,
    per_token_sem_loss=False,
    drop_wd=False,
):
    """
    Generate room polygons and labels from model predictions.

    This function supports two generation modes:
    - Non-poly2seq mode (poly2seq=False): RoomFormer generation with fg_mask filtering
    - Poly2seq mode (poly2seq=True): Raster2Seq generation with sequence-based predictions

    Args:
        model: The trained model to use for inference.
        samples: Input image samples (list of tensors).
        semantic_rich (bool, optional): If True, predict room types and window/door elements. Default is False.
        poly2seq (bool, optional): If True, uses sequence-based prediction mode with forward_inference.
                                   If False, uses mask-based filtering with fg_mask. Defaults to False.
        use_cache (bool, optional): If True, use caching in forward_inference. Only used when poly2seq=True. Default is True.
        per_token_sem_loss (bool, optional): If True, computes semantic class using voting across tokens.
                                             Only used when poly2seq=True. Default is False.
        drop_wd (bool, optional): If True, exclude window/door elements from output. Default is False.

    Returns:
        dict: Dictionary containing:
            - 'room': List of room polygons (and optionally window/door lines) per scene
            - 'labels': List of class labels per scene
    """

    model.eval()
    image_size = samples[0].size(2)

    # Get predictions based on mode
    if poly2seq:
        outputs = model.forward_inference(samples, use_cache)
        pred_corners = outputs["gen_out"]
        fg_mask = None
    else:
        outputs = model(samples)
        pred_corners = outputs["pred_coords"]
        pred_logits = outputs["pred_logits"]
        fg_mask = torch.sigmoid(pred_logits) > 0.5  # select valid corners

    bs = outputs["pred_logits"].shape[0]

    if "pred_room_logits" in outputs:
        pred_room_logits = outputs["pred_room_logits"]
        prob = torch.nn.functional.softmax(pred_room_logits, -1)
        _, pred_room_label = prob[..., :-1].max(-1)

    outputs = []
    output_classes = []

    # process per scene
    for i in range(bs):
        room_polys = []

        if semantic_rich:
            room_types = []
            window_doors = []
            window_doors_types = []
        else:
            window_doors = None
            room_types = None

        scene_outputs = _process_predictions(
            pred_corners,
            i,
            semantic_rich,
            poly2seq,
            fg_mask,
            image_size,
            pred_room_label if semantic_rich else None,
            pred_room_logits if semantic_rich else None,
            False,
            per_token_sem_loss,
            wd_as_line=True,
            door_window_index=None,
            dataset_name=None,
        )
        room_polys = scene_outputs["room_polys"]
        room_types = scene_outputs["room_types"]
        window_doors = scene_outputs["window_doors"]
        window_doors_types = scene_outputs["window_doors_types"]

        if not drop_wd and window_doors:
            outputs.append(room_polys + window_doors)
            output_classes.append(room_types + window_doors_types)
        else:
            outputs.append(room_polys)
            output_classes.append(room_types)

    out_dict = {"room": outputs, "labels": output_classes}
    return out_dict
