from __future__ import annotations

from enum import Enum
from math import ceil
from pathlib import Path
from typing import Annotated, Any, List, Literal, Optional, Tuple, Union

from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo

from src.utils.streamlit.components.extended_streamlit_pydantic import (
    annotate_dict_with_literal_values,
    get_dict_key_literal_values,
)
from src.utils.streamlit.st_pydantic_v2.input import SpecialFieldKeys
from src.utils.streamlit.ui_pydantic_v2.extra_types import Crop

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

    # Replace numerical fields with Crop objects
    edge_crop: Crop = Field(
        default_factory=Crop,
        description="Base crop for all images",
        json_schema_extra={SpecialFieldKeys.column_group: "crop_edge"},
    )

    standard_crop: Crop = Field(
        default_factory=Crop,
        description="Additional crop for non-edge images",
        json_schema_extra={SpecialFieldKeys.column_group: "crop_standard"},
    )

    @classmethod
    def set_image(cls, image: Image.Image):
        pass

        def _attach(fi: FieldInfo):
            """
            Clone `fi`, merge the extra JSON‑schema metadata, and
            return an Annotated type that Pydantic will pick up.
            """
            # Create a new dictionary with the desired values
            json_schema_extra = {}
            if fi.json_schema_extra is not None:
                # Copy each key-value pair manually
                if callable(fi.json_schema_extra):
                    fi.json_schema_extra(json_schema_extra)
                else:
                    for key, value in fi.json_schema_extra.items():
                        json_schema_extra[key] = value

            # Add new key-value pairs
            json_schema_extra["image"] = image
            json_schema_extra["aspect_dict"] = "Free"

            return Annotated[
                Crop,
                fi.merge_field_infos(json_schema_extra=json_schema_extra),
            ]

        # `create_model` gives us a fresh class that inherits every validator,
        # config option, etc. from `cls`; we simply override the two fields.
        return create_model(  # type: ignore[misc]
            f"{cls.__name__}WithImage",
            __base__=cls,
            edge_crop=_attach(cls.model_fields["edge_crop"]),
            standard_crop=_attach(cls.model_fields["standard_crop"]),
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

    # Convenience constructor for static row/column names -------------------- #
    @classmethod
    def specify_config(
        cls, rows: list[str], columns: list[str], image: Optional[Image.Image] = None
    ) -> ImageGridParams:
        overrides = {}
        if not rows:
            overrides["show_row_labels"] = Annotated[Literal[False], Field(default=False)]
        if not columns:
            overrides["show_col_labels"] = Annotated[Literal[False], Field(default=False)]
        if image is not None:
            image_grid_params = CropParams.set_image(image)
            overrides["crop_params"] = Annotated[image_grid_params, Field(default_factory=image_grid_params)]
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


def _get_extra_crop(width: int, height: int, edge_crop: Crop, standard_crop: Crop) -> tuple[float, float, float, float]:
    extra_left = standard_crop.left - edge_crop.left
    extra_top = standard_crop.top - edge_crop.top
    extra_right = (edge_crop.left + edge_crop.width) - (standard_crop.left + standard_crop.width)
    extra_bottom = (edge_crop.top + edge_crop.height) - (standard_crop.top + standard_crop.height)
    return extra_left * width, extra_top * height, extra_right * width, extra_bottom * height


def _calculate_crop_box(
    width: int,
    height: int,
    edge_crop: Crop,
    standard_crop: Crop,
    is_left_edge: bool,
    is_right_edge: bool,
    is_top_edge: bool,
    is_bottom_edge: bool,
) -> tuple[float, float, float, float]:
    # base_crop
    base = [
        standard_crop.left,
        standard_crop.top,
        standard_crop.left + standard_crop.width,
        standard_crop.top + standard_crop.height,
    ]

    extras = _get_extra_crop(1, 1, edge_crop, standard_crop)
    # apply diff_crop to base_crop
    if is_left_edge:
        base[0] -= extras[0]
    if is_top_edge:
        base[1] -= extras[1]
    if is_right_edge:
        base[2] += extras[2]
    if is_bottom_edge:
        base[3] += extras[3]

    return base[0] * width, base[1] * height, base[2] * width, base[3] * height


def _get_text_height(text: str, font: ImageFont.FreeTypeFont) -> int:
    dummy_img = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    return ceil(bbox[3] - bbox[1]) * 2


def combine_image_grid(images_paths_grid: List[List[Path]], params: ImageGridParams) -> Image.Image:
    # Fonts
    font_title = _safe_font(FONT_REGULAR, params.font_size)
    font_label = _safe_font(FONT_BOLD, params.label_font_size)

    num_rows = len(images_paths_grid)
    num_cols = len(images_paths_grid[0])

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
    right_margin = bottom_margin = 0
    # Load images
    original_image_size = (0, 0)
    images: list[list[Optional[Image.Image]]] = []
    for i, row_images_paths in enumerate(images_paths_grid):
        images.append([])
        row_images = images[-1]
        for j, img_path in enumerate(row_images_paths):
            if img_path is None:
                # Leave blank
                row_images.append(None)
                continue
            with Image.open(img_path) as im:
                if i == 0 and j == 0:
                    original_image_size = im.size
                else:
                    assert im.size == original_image_size, f"Image {img_path} has a different size than the first image"
                if params.crop_params.enable_crop:
                    # Calculate and apply crop
                    crop_box = _calculate_crop_box(
                        im.width,
                        im.height,
                        params.crop_params.edge_crop,
                        params.crop_params.standard_crop,
                        is_left_edge=j == 0,
                        is_right_edge=j == num_cols - 1,
                        is_top_edge=i == 0,
                        is_bottom_edge=i == num_rows - 1,
                    )
                    im = im.crop(crop_box)
                row_images.append(im)

    if params.crop_params.enable_crop:
        img_w = params.crop_params.standard_crop.width * original_image_size[0]
        img_h = params.crop_params.standard_crop.height * original_image_size[1]
        extras = _get_extra_crop(
            original_image_size[0],
            original_image_size[1],
            params.crop_params.edge_crop,
            params.crop_params.standard_crop,
        )
        left_margin += extras[0]
        right_margin += extras[2]
        top_margin += extras[1]
        bottom_margin += extras[3]
    else:
        extras = [0] * 4
        img_w, img_h = original_image_size

    grid_w = num_cols * img_w + max(num_cols - 1, 0) * params.padding
    grid_h = num_rows * img_h + max(num_rows - 1, 0) * params.padding
    canvas_w = int(left_margin + grid_w + right_margin)
    canvas_h = int(top_margin + grid_h + bottom_margin)

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
            label_img = Image.new("RGBA", (int(w), h), "white")
            label_draw = ImageDraw.Draw(label_img)

            bb = label_draw.textbbox((0, 0), label, font=font_label)
            txt_w, txt_h = bb[2] - bb[0], bb[3] - bb[1]

            label_draw.text(((w - txt_w) // 2, 0), label, fill="black", font=font_label)
            canvas.paste(label_img, (int(left_margin + col_idx * (img_w + params.padding)), title_h), label_img)

    # --------------------------------------------------------------------- #
    #  Paint each tile & row labels                                         #
    # --------------------------------------------------------------------- #
    values = get_dict_key_literal_values(params, "rows_labels_override")
    if params.show_row_labels:
        assert values is not None
        assert len(values) == num_rows
    for row_idx, row_images in enumerate(images):
        y_top = top_margin + row_idx * (img_h + params.padding)
        # Row label (once per row)
        if row_idx == 0:
            y_top -= extras[1]
        if params.show_row_labels:
            assert values is not None
            label = values[row_idx]
            label = params.rows_labels_override.get(label, label)
            w, h = img_h, row_label_h  # width is the image height (after rotation), height is the label height
            # Draw label on its own small canvas, then rotate 90°
            label_img = Image.new("RGBA", (int(w), h), "white")
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
            canvas.paste(label_img, (0, int(y_top)), label_img)

        # Images
        for col_idx, img in enumerate(row_images):
            x_left = left_margin + col_idx * (img_w + params.padding)
            if col_idx == 0:
                x_left -= extras[0]
            if img is None:
                continue  # leave blank

            canvas.paste(img, (int(x_left), int(y_top)))

    return canvas
