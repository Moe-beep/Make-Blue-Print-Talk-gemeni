"""
Utilities for floorplan visualization.
"""

import cv2
import numpy as np
from PIL import ImageColor

# plotly.colors.qualitative.Set3 + Dark2 (ColorBrewer), hardcoded to avoid a plotly dependency
ROOM_PALETTE = [
    "#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3", "#fdb462",
    "#b3de69", "#fccde5", "#d9d9d9", "#bc80bd", "#ccebc5", "#ffed6f",
    "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02",
    "#a6761d", "#666666",
]


S3D_LABEL = {
    0: "Living Room",
    1: "Kitchen",
    2: "Bedroom",
    3: "Bathroom",
    4: "Balcony",
    5: "Corridor",
    6: "Dining room",
    7: "Study",
    8: "Studio",
    9: "Store room",
    10: "Garden",
    11: "Laundry room",
    12: "Office",
    13: "Basement",
    14: "Garage",
    15: "Misc.",
    16: "Door",
    17: "Window",
}


CC5K_LABEL = {
    0: "Outdoor",
    1: "Kitchen",
    2: "Living Room",
    3: "Bed Room",
    4: "Bath",
    5: "Entry",
    6: "Storage",
    7: "Garage",
    8: "Undefined",
    9: "Window",
    10: "Door",
}


R2G_LABEL = {
    0: "unknown",
    1: "living_room",
    2: "kitchen",
    3: "bedroom",
    4: "bathroom",
    5: "restroom",
    6: "balcony",
    7: "closet",
    8: "corridor",
    9: "washing_room",
    10: "PS",
    11: "outside",
}


def auto_crop_whitespace(image, color_invert=True):
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Invert the image so floorplan is white and background is black
    if color_invert:
        _, binary = cv2.threshold(255 - gray, 1, 255, cv2.THRESH_BINARY)
    else:
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Find non-zero (non-white) content
    coords = cv2.findNonZero(binary)
    x, y, w, h = cv2.boundingRect(coords)

    # Crop image
    cropped_image = image[y : y + h, x : x + w].copy()

    # if polygons is None:
    #     return cropped_image, None

    # # Shift polygon coordinates
    # shifted_polygons = [
    #     [(px - x, py - y) for (px, py) in poly]
    #     for poly in polygons
    # ]
    return cropped_image, [x, y, w, h]  # shifted_polygons


def plot_room_map(preds, room_map, room_id=0, im_size=256, plot_text=True):
    """Draw room polygons overlaid on the density map"""
    centroid_x = int(np.mean(preds[:, 0]))
    centroid_y = int(np.mean(preds[:, 1]))

    # Get text size to create a background box
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3
    thickness = 1
    text = str(room_id)
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    border_color = (252, 252, 0)

    for i, corner in enumerate(preds):
        if i == len(preds) - 1:
            cv2.line(
                room_map,
                (round(corner[0]), round(corner[1])),
                (round(preds[0][0]), round(preds[0][1])),
                border_color,
                2,
            )
        else:
            cv2.line(
                room_map,
                (round(corner[0]), round(corner[1])),
                (round(preds[i + 1][0]), round(preds[i + 1][1])),
                border_color,
                2,
            )
        cv2.circle(room_map, (round(corner[0]), round(corner[1])), 2, (0, 0, 255), 2)
        # cv2.putText(room_map, str(i), (round(corner[0]), round(corner[1])), cv2.FONT_HERSHEY_SIMPLEX,
        #            0.4, (0, 255, 0), 1, cv2.LINE_AA)

        # Draw white background box with transparency
        # overlay = room_map.copy()
        # cv2.addWeighted(overlay, 0.7, room_map, 0.3, 0, room_map)  # 70% opacity

        # Draw text
        if plot_text:
            cv2.rectangle(
                room_map,
                (centroid_x - text_width // 2 - 2, centroid_y - text_height // 2 - 2),
                (centroid_x + text_width // 2 + 2, centroid_y + text_height // 2 + 2),
                (255, 255, 255),  # (0, 0, 0),
                -1,
            )  # Filled rectangle
            cv2.putText(
                room_map,
                text,
                (centroid_x - text_width // 2, centroid_y + text_height // 2),
                font,
                font_scale,
                (0, 100, 0),
                thickness,
            )

    return room_map


def plot_density_map(sample, image_size, room_polys, pred_room_label_per_scene, plot_text=True):
    if not isinstance(sample, np.ndarray):
        density_map = np.transpose(sample.cpu().numpy(), [1, 2, 0])
    else:
        density_map = sample
    if density_map.shape[2] == 3:
        density_map = density_map * (image_size - 1)
    else:
        density_map = np.repeat(density_map, 3, axis=2) * (image_size - 1)
    pred_room_map = np.zeros([image_size, image_size, 3])

    for room_poly, room_id in zip(room_polys, pred_room_label_per_scene):
        pred_room_map = plot_room_map(room_poly, pred_room_map, room_id, im_size=image_size, plot_text=plot_text)

    alpha = 0.4  # Adjust for desired transparency
    pred_room_map = cv2.addWeighted(density_map.astype(np.uint8), alpha, pred_room_map.astype(np.uint8), 1 - alpha, 0)
    return pred_room_map


def plot_semantic_rich_floorplan_opencv(
    polygons,
    file_name,
    img_w=256,
    img_h=256,
    door_window_index=[16, 17],
    semantics_label_mapping=S3D_LABEL,
    is_bw=False,
    plot_text=True,
    one_color=False,
    scale=1,
    is_sem=True,
    highlight_indices=None,
):
    """
    Plot semantically-rich floorplan using OpenCV with improved quality.

    Args:
        polygons (list): A list of polygons, where each polygon is a list of (x, y) coordinates.
        file_name (str): Path to save the output image.
        img_w (int): Width of the output image.
        img_h (int): Height of the output image.
        door_window_index (list): Indices for door and window types.
        semantics_label_mapping (dict): Mapping from polygon type to semantic label.
        is_bw (bool): If True, use black and white colors only.
        line_thickness (int): Thickness of lines for polygons and doors/windows.
        text_padding (int): Padding around text labels.
        font_scale (float): Scale factor for text size.
        room_alpha (float): Transparency for room colors (0.0-1.0).
        anti_aliasing (bool): Whether to use anti-aliasing for lines.
    """
    line_thickness = 2
    text_padding = 1
    font_scale = 0.25
    room_alpha = 0.6

    if scale != 1:
        new_polygons = []
        for poly, poly_label in polygons:
            poly = (poly * scale).round().astype(np.int32)
            new_polygons.append([poly, poly_label])
        polygons = new_polygons

    if one_color:
        colors = ["#FFD700"]
    else:
        colors = ROOM_PALETTE

    # Create a white background image (more conventional for floorplans)
    if is_bw:
        image = np.ones((img_h, img_w), dtype=np.uint8) * 255  # White grayscale image
    else:
        image = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255  # White RGB image

    # Create a separate layer for room colors
    overlay = image.copy()

    highlight_indices_set = set(highlight_indices) if highlight_indices is not None else None

    # Track polygons for each type for proper layering
    room_polygons = []
    door_polygons = []
    window_polygons = []

    # Sort polygons by type (track original index for highlight support)
    for poly_idx, (poly, poly_type) in enumerate(polygons):
        if len(poly) < 2:  # Skip invalid polygons
            continue

        points = np.array(poly, dtype=np.int32)

        if door_window_index and poly_type == door_window_index[0]:  # Door
            door_polygons.append((points, poly_type, poly_idx))
        elif door_window_index and poly_type == door_window_index[1]:  # Window
            window_polygons.append((points, poly_type, poly_idx))
        else:  # Room
            room_polygons.append((points, poly_type, poly_idx))

    # Draw rooms first (bottom layer)
    for room_id, (points, poly_type, orig_idx) in enumerate(room_polygons):
        # Fill room with color
        if not is_bw:
            # If highlight_indices is set, use gold for non-highlighted polygons
            if highlight_indices_set is not None and orig_idx not in highlight_indices_set:
                rgb_color = ImageColor.getcolor("#FFD700", "RGB")
            elif not is_sem:
                rgb_color = ImageColor.getcolor(colors[room_id % len(colors)], "RGB")
            else:
                rgb_color = ImageColor.getcolor(colors[poly_type % len(colors)], "RGB")

            bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])
            cv2.fillPoly(overlay, [points], color=bgr_color)
        else:
            # Use light gray for rooms in BW mode
            cv2.fillPoly(overlay, [points], color=(240, 240, 240))

        # Draw room outline
        line_type = cv2.LINE_AA
        cv2.polylines(image, [points], isClosed=True, color=(0, 0, 0), thickness=line_thickness, lineType=line_type)

    # Blend overlay with transparency
    cv2.addWeighted(overlay, room_alpha, image, 1 - room_alpha, 0, image)

    # Draw doors with proper styling
    for points, _, _orig_idx in door_polygons:
        if len(points) >= 2:
            # For doors, we can improve by drawing arcs to represent swing
            # Here we draw them as thick lines with distinctive color
            door_color = (100, 100, 100) if is_bw else (0, 0, 255)  # Gray for BW, Red for RGB
            line_type = cv2.LINE_AA
            cv2.polylines(
                image, [points], isClosed=False, color=door_color, thickness=line_thickness * 2, lineType=line_type
            )

    # Draw windows with dashed styling
    for points, _, _orig_idx in window_polygons:
        if len(points) >= 2:
            window_color = (150, 150, 150) if is_bw else (255, 0, 0)  # Gray for BW, Blue for RGB

            # Create dashed line effect for windows
            if len(points) == 2:
                # For a simple line window
                pt1, pt2 = points[0], points[1]
                dash_length = 5

                # Calculate line parameters
                length = np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)
                if length > 0:
                    num_dashes = max(2, int(length / (2 * dash_length)))

                    for i in range(num_dashes):
                        start_ratio = i / num_dashes
                        end_ratio = (i + 0.5) / num_dashes

                        start_x = int(pt1[0] + (pt2[0] - pt1[0]) * start_ratio)
                        start_y = int(pt1[1] + (pt2[1] - pt1[1]) * start_ratio)
                        end_x = int(pt1[0] + (pt2[0] - pt1[0]) * end_ratio)
                        end_y = int(pt1[1] + (pt2[1] - pt1[1]) * end_ratio)

                        line_type = cv2.LINE_AA
                        cv2.line(
                            image,
                            (start_x, start_y),
                            (end_x, end_y),
                            window_color,
                            thickness=line_thickness,
                            lineType=line_type,
                        )
            else:
                # For multi-point windows
                line_type = cv2.LINE_AA
                cv2.polylines(
                    image, [points], isClosed=True, color=window_color, thickness=line_thickness, lineType=line_type
                )

    if plot_text:
        # Add room labels
        for i, (points, poly_type, _orig_idx) in enumerate(room_polygons):
            # if i > 1: continue # TODO:test
            # Calculate the centroid for text placement
            M = cv2.moments(points)
            if M["m00"] != 0:  # Avoid division by zero
                centroid_x = int(M["m10"] / M["m00"])
                centroid_y = int(M["m01"] / M["m00"])

                # Get room label
                label = semantics_label_mapping[poly_type]

                # Get text size for centering and background
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]

                # Calculate text background rectangle
                text_x = centroid_x - text_size[0] // 2
                text_y = centroid_y + text_size[1] // 2

                # Create background for text
                rect_top_left = (text_x - text_padding, text_y - text_size[1] - text_padding)
                rect_bottom_right = (text_x + text_size[0] + text_padding, text_y + text_padding)

                # Draw semi-transparent white background for text
                background = image.copy()
                cv2.rectangle(background, rect_top_left, rect_bottom_right, (255, 255, 255), -1)

                # Blend the background
                cv2.addWeighted(background, 0.7, image, 0.3, 0, image)

                # Draw the text
                cv2.putText(
                    image,
                    label,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 0, 0),  # Black text
                    1,  # Thickness
                    cv2.LINE_AA,  # Anti-aliased text
                )

    # Add border around the image for better framing
    # cv2.rectangle(image, (0, 0), (img_w-1, img_h-1), (0, 0, 0), 1, cv2.LINE_AA)

    # Save with high quality
    if file_name is not None:
        if is_bw:
            cv2.imwrite(file_name, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        else:
            cv2.imwrite(file_name, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        print(f"Saved improved floorplan to {file_name}")

    return image
