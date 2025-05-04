from __future__ import annotations

from enum import Enum
from math import ceil
from pathlib import Path
from typing import Annotated, Any, List, Literal, Optional, Sequence, Tuple, Union

from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field, create_model, model_validator

from src.utils.streamlit.components.extended_streamlit_pydantic import (
    annotate_dict_with_literal_values,
    get_dict_key_literal_values,
)
from src.utils.streamlit.st_pydantic_v2.input import SpecialFieldKeys

FONT_BASE = lambda suffix: f"/usr/share/fonts/truetype/liberation/LiberationSerif{suffix}.ttf"  # noqa: E731
FONT_REGULAR = FONT_BASE("-Regular")
FONT_BOLD = FONT_BASE("-Bold")


class GridOrganizer(BaseModel):
    """Organizes images into a grid based on row and column categories.

    This version doesn't use callable fields to avoid Pydantic serialization issues.
    """

    row_order: Optional[List[Union[str, int, Enum]]] = Field(
        default=None, description="Custom ordering for rows (optional)"
    )
    col_order: Optional[List[Union[str, int, Enum]]] = Field(
        default=None, description="Custom ordering for columns (optional)"
    )


def organize_images_to_grid_with_keys(
    items_with_keys: List[Tuple[Path, Any, Any]],  # (path, row_key, col_key)
    organizer: GridOrganizer,
) -> List[List[Optional[Path]]]:
    """Organize image paths into a grid based on pre-extracted row and column keys.

    Args:
        items_with_keys: List of tuples containing (image_path, row_key, col_key)
        organizer: Configuration for how to organize the grid

    Returns:
        A 2D grid (list of lists) of image paths organized by row and column categories
    """
    # Extract unique row and column keys
    row_keys = set()
    col_keys = set()

    for _, row_key, col_key in items_with_keys:
        row_keys.add(row_key)
        col_keys.add(col_key)

    # Use provided order or sort naturally
    if organizer.row_order:
        sorted_row_keys = [key for key in organizer.row_order if key in row_keys]
    else:
        sorted_row_keys = sorted(row_keys)

    if organizer.col_order:
        sorted_col_keys = [key for key in organizer.col_order if key in col_keys]
    else:
        sorted_col_keys = sorted(col_keys)

    # Create mapping from keys to indices
    row_indices = {key: idx for idx, key in enumerate(sorted_row_keys)}
    col_indices = {key: idx for idx, key in enumerate(sorted_col_keys)}

    # Initialize empty grid
    num_rows = len(sorted_row_keys)
    num_cols = len(sorted_col_keys)
    grid: List[List[Optional[Path]]] = [[None for _ in range(num_cols)] for _ in range(num_rows)]

    # Place items in the grid
    for img_path, row_key, col_key in items_with_keys:
        if row_key in row_indices and col_key in col_indices:
            row_idx = row_indices[row_key]
            col_idx = col_indices[col_key]
            grid[row_idx][col_idx] = img_path

    return grid


crop_kwargs = {
    SpecialFieldKeys.kwargs: {
        "min_value": 0.0,
        "max_value": 1.0,
        "step": 0.0001,
        "format": "%.4f",
    }
}


class CropParams(BaseModel):
    """Configuration for image cropping with special handling for edge images.

    All values are between 0.0 and 1.0, representing percentages of the image dimensions.
    Edge parameters apply to ALL images as a base crop amount.
    Regular crop parameters are ADDITIONAL crop amounts applied only to non-edge images.
    """

    enable_crop: bool = Field(default=False, description="Enable image cropping")

    # Edge parameters (base crop for all images)
    edge_left: float = Field(
        default=0.0,
        ge=0.0,
        description="Base crop from left for all images (0.0-1.0)",
        json_schema_extra={SpecialFieldKeys.column_group: "crop_edge", **crop_kwargs},
    )
    edge_right: float = Field(
        default=0.0,
        ge=0.0,
        description="Base crop from right for all images (0.0-1.0)",
        json_schema_extra={SpecialFieldKeys.column_group: "crop_edge", **crop_kwargs},
    )
    edge_top: float = Field(
        default=0.0,
        ge=0.0,
        description="Base crop from top for all images (0.0-1.0)",
        json_schema_extra={SpecialFieldKeys.column_group: "crop_edge", **crop_kwargs},
    )
    edge_bottom: float = Field(
        default=0.0,
        ge=0.0,
        description="Base crop from bottom for all images (0.0-1.0)",
        json_schema_extra={SpecialFieldKeys.column_group: "crop_edge", **crop_kwargs},
    )

    # Standard crop parameters (additional crop for non-edge images)
    crop_left: float = Field(
        default=0.0,
        ge=0.0,
        description="Additional crop from left for non-edge images (0.0-1.0)",
        json_schema_extra={SpecialFieldKeys.column_group: "crop_standard", **crop_kwargs},
    )
    crop_right: float = Field(
        default=0.0,
        ge=0.0,
        description="Additional crop from right for non-edge images (0.0-1.0)",
        json_schema_extra={SpecialFieldKeys.column_group: "crop_standard", **crop_kwargs},
    )
    crop_top: float = Field(
        default=0.0,
        ge=0.0,
        description="Additional crop from top for non-edge images (0.0-1.0)",
        json_schema_extra={SpecialFieldKeys.column_group: "crop_standard", **crop_kwargs},
    )
    crop_bottom: float = Field(
        default=0.0,
        ge=0.0,
        description="Additional crop from bottom for non-edge images (0.0-1.0)",
        json_schema_extra={SpecialFieldKeys.column_group: "crop_standard", **crop_kwargs},
    )


class ImageGridParams(BaseModel):
    title: str = Field(
        default="", description="Title for the grid", json_schema_extra={SpecialFieldKeys.column_group: "titles"}
    )

    title_height: int = Field(
        default=30,
        ge=0,
        description="Height for titles",
        json_schema_extra={SpecialFieldKeys.column_group: "titles"},
    )
    # font_style: Literal[ name of fonts...
    font_size: int = Field(
        default=25, ge=0, description="Font size", json_schema_extra={SpecialFieldKeys.column_group: "titles"}
    )

    img_width: int = Field(
        default=-1,
        ge=-1,
        description="If -1, will use max width of images",
        json_schema_extra={SpecialFieldKeys.column_group: "img_size"},
    )
    img_height: int = Field(
        default=-1,
        ge=-1,
        description="If -1, will use max height of images",
        json_schema_extra={SpecialFieldKeys.column_group: "img_size"},
    )
    padding: int = Field(
        default=10,
        ge=0,
        description="Padding between images",
        json_schema_extra={SpecialFieldKeys.column_group: "img_size"},
    )

    # Separator
    sep1: None = Field(default=None, json_schema_extra={SpecialFieldKeys.separator: True})

    show_row_labels: bool = Field(
        default=False,
        description="Show row labels",
        json_schema_extra={SpecialFieldKeys.column_group: "labels_display"},
    )
    show_col_labels: bool = Field(
        default=False,
        description="Show column labels",
        json_schema_extra={SpecialFieldKeys.column_group: "labels_display"},
    )

    rows_labels_override: dict[str, str] = Field(default_factory=dict, description="Override labels for rows")
    columns_labels_override: dict[str, str] = Field(default_factory=dict, description="Override labels for columns")

    label_font_size: int = Field(default=20, ge=0, description="Font size for labels")

    # Put crop params in an expander
    crop_params: CropParams = Field(
        default_factory=CropParams,
        description="Image cropping configuration",
        json_schema_extra={SpecialFieldKeys.expander: "Crop Settings"},
    )

    # --- derived helpers ---------------------------------------------------- #

    @model_validator(mode="after")
    def _validate_img_size(self):
        if self.img_width == 0 or self.img_height == 0:
            raise ValueError("``img_width`` and ``img_height`` may be ‒1 (auto) or ≥1.")
        return self

    # Convenience constructor for static row/column names -------------------- #
    @classmethod
    def specify_config(cls, rows: list[str], columns: list[str]) -> ImageGridParams:
        overrides = {}
        if not rows:
            overrides["show_row_labels"] = Annotated[Literal[False], Field(default=False)]
        if not columns:
            overrides["show_col_labels"] = Annotated[Literal[False], Field(default=False)]
        return create_model(  # type: ignore
            f"{cls.__name__}Config",
            __base__=cls,
            rows_labels_override=annotate_dict_with_literal_values(rows, str),
            columns_labels_override=annotate_dict_with_literal_values(columns, str),
            **overrides,
        )


# --------------------------------------------------------------------------- #
#  Core logic                                                                 #
# --------------------------------------------------------------------------- #


def _safe_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    """Attempt to load the requested font; gracefully fall back to Pillow default."""
    return ImageFont.truetype(path, size)


def _calculate_crop_box(
    width: int,
    height: int,
    crop_params: CropParams,
    is_left_edge: bool,
    is_right_edge: bool,
    is_top_edge: bool,
    is_bottom_edge: bool,
) -> tuple[int, int, int, int]:
    """Calculate the crop box coordinates based on image position and crop parameters.

    Args:
        width: Image width
        height: Image height
        crop_params: Cropping configuration
        is_left_edge: Whether the image is on the left edge of the grid
        is_right_edge: Whether the image is on the right edge of the grid
        is_top_edge: Whether the image is on the top edge of the grid
        is_bottom_edge: Whether the image is on the bottom edge of the grid

    Returns:
        Tuple of (left, top, right, bottom) crop box coordinates
    """
    # Apply base edge crop to all images
    left = int(crop_params.edge_left * width)
    top = int(crop_params.edge_top * height)
    right = width - int(crop_params.edge_right * width)
    bottom = height - int(crop_params.edge_bottom * height)

    # Apply additional crop for non-edge images
    if not is_left_edge:
        additional_left = int(crop_params.crop_left * width)
        left += additional_left

    if not is_right_edge:
        additional_right = int(crop_params.crop_right * width)
        right -= additional_right

    if not is_top_edge:
        additional_top = int(crop_params.crop_top * height)
        top += additional_top

    if not is_bottom_edge:
        additional_bottom = int(crop_params.crop_bottom * height)
        bottom -= additional_bottom

    # Ensure valid crop box (left < right, top < bottom)
    left = min(left, right - 1)
    top = min(top, bottom - 1)

    return left, top, right, bottom


def _auto_image_size_with_crop(
    image_grid: Sequence[Sequence[Optional[Path]]], crop_params: CropParams, num_rows: int, num_cols: int
) -> tuple[int, int]:
    """Return maximal (width, height) across all non-None images after applying cropping.

    This version applies cropping before determining the max dimensions.

    Args:
        image_grid: 2D grid of image paths
        crop_params: Cropping configuration
        num_rows: Number of rows in the grid
        num_cols: Number of columns in the grid

    Returns:
        Tuple of (max_width, max_height) after cropping
    """
    max_w = max_h = 0

    for row_idx, row in enumerate(image_grid):
        for col_idx, pth in enumerate(row):
            if pth is None:
                continue

            with Image.open(pth) as im:
                if crop_params.enable_crop:
                    # Determine if this image is on an edge
                    is_left_edge = col_idx == 0
                    is_right_edge = col_idx == num_cols - 1
                    is_top_edge = row_idx == 0
                    is_bottom_edge = row_idx == num_rows - 1

                    # Calculate crop box
                    left, top, right, bottom = _calculate_crop_box(
                        im.width, im.height, crop_params, is_left_edge, is_right_edge, is_top_edge, is_bottom_edge
                    )

                    # Calculate dimensions after cropping
                    crop_width = right - left
                    crop_height = bottom - top

                    max_w = max(max_w, crop_width)
                    max_h = max(max_h, crop_height)
                else:
                    max_w = max(max_w, im.width)
                    max_h = max(max_h, im.height)

    if max_w == 0 or max_h == 0:
        raise ValueError("At least one valid image is required when auto-sizing.")

    return max_w, max_h


def _auto_image_size(image_grid: Sequence[Sequence[Optional[Path]]]) -> tuple[int, int]:
    """Return maximal (width, height) across all non-None images in the grid."""
    max_w = max_h = 0
    for row in image_grid:
        for pth in row:
            if pth is None:
                continue
            with Image.open(pth) as im:
                max_w = max(max_w, im.width)
                max_h = max(max_h, im.height)
    if max_w == 0 or max_h == 0:
        raise ValueError("At least one valid image is required when auto-sizing.")
    return max_w, max_h


def _get_text_height(text: str, font: ImageFont.FreeTypeFont) -> int:
    dummy_img = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    return ceil(bbox[3] - bbox[1]) * 2


def combine_image_grid(images: List[List[Path]], params: ImageGridParams) -> Image.Image:
    # Fonts
    font_title = _safe_font(FONT_REGULAR, params.font_size)
    font_label = _safe_font(FONT_BOLD, params.label_font_size)

    # --------------------------------------------------------------------- #
    #  Normalise grid dimensions                                            #
    # --------------------------------------------------------------------- #
    num_rows = len(images)
    num_cols = len(images[0])

    # Auto-determine tile size if requested, applying cropping first if enabled
    if params.img_width == -1 or params.img_height == -1:
        if params.crop_params.enable_crop:
            auto_w, auto_h = _auto_image_size_with_crop(images, params.crop_params, num_rows, num_cols)
        else:
            auto_w, auto_h = _auto_image_size(images)

        img_w = auto_w if params.img_width == -1 else params.img_width
        img_h = auto_h if params.img_height == -1 else params.img_height
    else:
        img_w, img_h = params.img_width, params.img_height

    # --------------------------------------------------------------------- #
    #  Calculate canvas size                                                #
    # --------------------------------------------------------------------- #
    #  Title row (optional) + column-label row (optional)
    title_h = params.title_height if params.title else 0
    col_label_h = _get_text_height("TEST", font_label) if params.show_col_labels else 0
    top_margin = title_h + col_label_h

    #  Row-label column (optional)
    row_label_h = _get_text_height("TEST", font_label) if params.show_row_labels else 0
    left_margin = row_label_h

    grid_w = num_cols * img_w + max(num_cols - 1, 0) * params.padding
    grid_h = num_rows * img_h + max(num_rows - 1, 0) * params.padding
    canvas_w = left_margin + grid_w
    canvas_h = top_margin + grid_h

    # --------------------------------------------------------------------- #
    #  Prepare drawing context                                              #
    # --------------------------------------------------------------------- #
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    # --------------------------------------------------------------------- #
    #  Draw title                                                           #
    # --------------------------------------------------------------------- #
    if params.title:
        bb = draw.textbbox((0, 0), params.title, font=font_title)
        txt_w, txt_h = bb[2] - bb[0], bb[3] - bb[1]
        draw.text(
            ((canvas_w - txt_w) // 2, (params.title_height - txt_h) // 2),
            params.title,
            fill="black",
            font=font_title,
        )

    # --------------------------------------------------------------------- #
    #  Draw column labels                                                   #
    # --------------------------------------------------------------------- #
    if params.show_col_labels:
        values = get_dict_key_literal_values(params, "columns_labels_override")
        assert values is not None
        assert len(values) == num_cols

        for col_idx in range(num_cols):
            label = values[col_idx]
            label = params.columns_labels_override.get(label, label)
            w, h = img_w, col_label_h  # no rotation
            label_img = Image.new("RGBA", (w, h), "white")
            label_draw = ImageDraw.Draw(label_img)

            bb = label_draw.textbbox((0, 0), label, font=font_label)
            txt_w, txt_h = bb[2] - bb[0], bb[3] - bb[1]

            label_draw.text(((w - txt_w) // 2, 0), label, fill="black", font=font_label)
            canvas.paste(label_img, (left_margin + col_idx * (img_w + params.padding), title_h), label_img)

    # --------------------------------------------------------------------- #
    #  Paint each tile & row labels                                         #
    # --------------------------------------------------------------------- #
    values = get_dict_key_literal_values(params, "rows_labels_override")
    if params.show_row_labels:
        assert values is not None
        assert len(values) == num_rows
    for row_idx, row in enumerate(images):
        y_top = top_margin + row_idx * (img_h + params.padding)
        # Row label (once per row)
        if params.show_row_labels:
            assert values is not None
            label = values[row_idx]
            label = params.rows_labels_override.get(label, label)
            w, h = img_h, row_label_h  # width is the image height (after rotation), height is the label height
            # Draw label on its own small canvas, then rotate 90°
            label_img = Image.new("RGBA", (w, h), "white")
            label_draw = ImageDraw.Draw(label_img)

            bb = label_draw.textbbox((0, 0), label, font=font_label)
            txt_w, txt_h = bb[2] - bb[0], bb[3] - bb[1]

            label_draw.text(
                ((w - txt_w) // 2, 0),
                label,
                fill="black",
                font=font_label,
            )
            label_img = label_img.rotate(90, expand=True)
            # Paste preserving alpha (row labels overlay the white canvas)
            canvas.paste(label_img, (0, y_top), label_img)

        # Images
        for col_idx, img_path in enumerate(row):
            x_left = row_label_h + col_idx * (img_w + params.padding)
            if img_path is None:
                continue  # leave blank

            with Image.open(img_path) as im:
                # Apply cropping if enabled
                if params.crop_params.enable_crop:
                    # Determine if this image is on an edge
                    is_left_edge = col_idx == 0
                    is_right_edge = col_idx == num_cols - 1
                    is_top_edge = row_idx == 0
                    is_bottom_edge = row_idx == num_rows - 1

                    # Calculate and apply crop
                    crop_box = _calculate_crop_box(
                        im.width,
                        im.height,
                        params.crop_params,
                        is_left_edge,
                        is_right_edge,
                        is_top_edge,
                        is_bottom_edge,
                    )
                    im = im.crop(crop_box)

                # Resize if needed after cropping
                if im.size != (img_w, img_h):
                    im = im.resize((img_w, img_h), Image.Resampling.LANCZOS)

                canvas.paste(im, (x_left, y_top))

    return canvas
