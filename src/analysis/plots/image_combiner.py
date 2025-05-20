from __future__ import annotations

from enum import Enum
from math import ceil
from pathlib import Path
from typing import Annotated, Any, List, Literal, Optional, Tuple, Union

from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo

from src.utils.infra.image_utils import resize_image
from src.utils.streamlit.components.extended_streamlit_pydantic import (
    annotate_dict_with_literal_values,
)
from src.utils.streamlit.st_pydantic_v2.input import SpecialFieldKeys
from src.utils.streamlit.ui_pydantic_v2.extra_types import PercentageCrop

dejavu_path = font_manager.findfont("DejaVu Sans")
bold_dejavu_path = font_manager.findfont(font_manager.FontProperties("DejaVu Sans", weight="bold"))
FONT_BASE = lambda suffix: f"/usr/share/fonts/truetype/liberation/LiberationSerif{suffix}.ttf"  # noqa: E731
FONT_REGULAR = FONT_BASE("-Regular")


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

    enable_crop: bool = Field(default=True, description="Enable image cropping")

    # Replace numerical fields with Crop objects
    edge_crop: PercentageCrop = Field(
        default_factory=lambda: PercentageCrop(
            left=3.43,
            top=20.5,
            width=96.3,
            height=79.5,
        ),
        description="Base crop for all images",
        json_schema_extra={SpecialFieldKeys.column_group: "crop_edge"},
    )

    standard_crop: PercentageCrop = Field(
        default_factory=lambda: PercentageCrop(
            left=14.57,
            top=19.29,
            width=84.72,
            height=69.5,
        ),
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
                PercentageCrop,
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


class LegendItem(BaseModel):
    label: str
    color: str
    linestyle: str


TAnchor = Literal[
    "la", "lt", "lm", "ls", "lb", "ld", "ma", "mt", "mm", "ms", "mb", "md", "ra", "rt", "rm", "rs", "rb", "rd"
]


class LegendParams(BaseModel):
    """Configuration for the legend appearance."""

    width: int = Field(
        default=80,
        ge=0,
        description="Width of color/line sample",
        json_schema_extra={SpecialFieldKeys.column_group: "line"},
    )
    height: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Height of sample as ratio of legend height",
        json_schema_extra={SpecialFieldKeys.column_group: "line"},
    )
    rows: int = Field(
        default=1,
        ge=1,
        description="Number of rows in the legend",
        json_schema_extra={SpecialFieldKeys.column_group: "line"},
    )
    horizontal_spacing: int = Field(
        default=50,
        ge=0,
        description="Spacing between sample and text",
        json_schema_extra={SpecialFieldKeys.column_group: "line"},
    )
    font_size: int = Field(
        default=50, ge=0, description="Font size for legend", json_schema_extra={SpecialFieldKeys.column_group: "text"}
    )
    anchor: TAnchor = Field(
        default="lm", description="Anchor for legend", json_schema_extra={SpecialFieldKeys.column_group: "text"}
    )
    show_border: bool = Field(
        default=True,
        title="Show",
        description="Show border line above legend",
        json_schema_extra={SpecialFieldKeys.column_group: "border"},
    )
    legend_padding: int = Field(
        default=30,
        ge=0,
        description="Padding between legend and border",
        json_schema_extra={SpecialFieldKeys.column_group: "border"},
    )
    border_width: int = Field(
        default=1, ge=0, description="Width of border line", json_schema_extra={SpecialFieldKeys.column_group: "border"}
    )


# Prefix styles for row and column labels
PrefixStyle = Literal[
    "none",  # No prefix
    "lowercase_letter_paren",  # (a), (b), ...
    "uppercase_letter_paren",  # (A), (B), ...
    "number_paren",  # (1), (2), ...
    "lowercase_roman_paren",  # (i), (ii), ...
    "uppercase_roman_paren",  # (I), (II), ...
    "lowercase_letter_dot",  # a., b., ...
    "uppercase_letter_dot",  # A., B., ...
    "number_dot",  # 1., 2., ...
]


def to_roman(num: int) -> str:
    """Convert an integer to a Roman numeral."""
    val = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    syms = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]
    roman_num = ""
    i = 0
    while num > 0:
        for _ in range(num // val[i]):
            roman_num += syms[i]
            num -= val[i]
        i += 1
    return roman_num


def generate_prefix(index: int, style: PrefixStyle) -> str:
    """Generate a prefix based on the index (0-based) and style."""
    if style == "none":
        return ""

    # Handle letter styles
    if style.startswith("lowercase_letter"):
        char = chr(97 + index)  # 'a' starts at ASCII 97
    elif style.startswith("uppercase_letter"):
        char = chr(65 + index)  # 'A' starts at ASCII 65

    # Handle number styles
    elif style.startswith("number"):
        char = str(index + 1)  # 1-based numbering for human readability

    # Handle roman numeral styles
    elif style.startswith("lowercase_roman"):
        char = to_roman(index + 1).lower()
    elif style.startswith("uppercase_roman"):
        char = to_roman(index + 1)
    else:
        return ""

    # Format with parentheses or dot
    if style.endswith("_paren"):
        return f"({char})"
    elif style.endswith("_dot"):
        return f"{char}."

    return char


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
        default=0,
        ge=0,
        description="Padding between images",
        json_schema_extra={SpecialFieldKeys.column_group: "img_size"},
    )
    allow_different_image_sizes: bool = Field(
        default=False,
        description="Allow different image sizes",
    )

    # Separator
    sep1: None = Field(default=None, json_schema_extra={SpecialFieldKeys.separator: True})

    column_header_padding: float = Field(
        default=-25.0,
        description="Additional padding for column headers (can be negative)",
        json_schema_extra={SpecialFieldKeys.column_group: "labels_display"},
    )

    show_row_labels: bool = Field(
        default=False,
        description="Show row labels",
        json_schema_extra={SpecialFieldKeys.column_group: "labels_display"},
    )
    show_col_labels: bool = Field(
        default=True,
        description="Show column labels",
        json_schema_extra={SpecialFieldKeys.column_group: "labels_display"},
    )

    row_prefix_style: PrefixStyle = Field(
        default="none",
        description="Prefix style for row labels",
        json_schema_extra={SpecialFieldKeys.column_group: "labels_display"},
    )

    column_prefix_style: PrefixStyle = Field(
        default="lowercase_letter_paren",
        description="Prefix style for column labels",
        json_schema_extra={SpecialFieldKeys.column_group: "labels_display"},
    )

    rows_labels_override: dict[str, str] = Field(default_factory=dict, description="Override labels for rows")
    columns_labels_override: dict[str, str] = Field(default_factory=dict, description="Override labels for columns")

    label_font_size: int = Field(default=70, ge=0, description="Font size for labels")

    # Put crop params in an expander
    crop_params: CropParams = Field(
        default_factory=CropParams,
        description="Image cropping configuration",
        json_schema_extra={SpecialFieldKeys.expander: "Crop Settings"},
    )

    # Legend configuration
    legend_params: LegendParams = Field(
        default_factory=LegendParams,
        description="Legend appearance configuration",
        json_schema_extra={SpecialFieldKeys.expander: "Legend Settings"},
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


def _get_text_height(text: str, font: ImageFont.FreeTypeFont) -> int:
    dummy_img = Image.new("RGBA", (1, 1))
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    return ceil(bbox[3] - bbox[1]) * 2


def _draw_legend(
    draw: ImageDraw.ImageDraw,
    canvas_w: float,
    legend_y: float,
    legend_h: float,
    legend_items: list[LegendItem],
    font_legend: ImageFont.FreeTypeFont,
    legend_params: LegendParams,
) -> None:
    """Draw a legend with the given items at the specified position.

    Legend items are laid out one after another and the whole row is centred
    horizontally; they are **not** stretched to fill the full canvas width.
    Multiple rows are supported via ``legend_params.rows``.
    """
    if not legend_items:
        return

    legend_padding = legend_params.legend_padding / 2

    legend_y += legend_padding

    # ── optional top border ────────────────────────────────────────────────
    if legend_params.show_border:
        draw.line(
            [(0, legend_y), (canvas_w, legend_y)],
            fill="black",
            width=legend_params.border_width,
        )

    legend_y += legend_padding

    # ── basic layout figures ───────────────────────────────────────────────
    num_items = len(legend_items)
    actual_rows = min(legend_params.rows, num_items)
    items_per_row = ceil(num_items / actual_rows)
    row_height = (legend_h - legend_params.legend_padding - legend_params.border_width) / actual_rows

    sample_width = legend_params.width
    sample_to_text = 10  # gap sample→text
    inter_item_spacing = legend_params.horizontal_spacing  # gap item→item

    item_index = 0
    for row_idx in range(actual_rows):
        # Items that belong to this row
        row_items = legend_items[item_index : item_index + items_per_row]
        item_index += items_per_row

        # Compute per‑item widths (sample + gap + text)
        per_item_widths: list[float] = []
        for item in row_items:
            bb = draw.textbbox((0, 0), item.label, font=font_legend)
            txt_w = bb[2] - bb[0]
            per_item_widths.append(sample_width + sample_to_text + txt_w)

        if not per_item_widths:
            continue

        # Centre the entire row of items
        row_total_w = sum(per_item_widths) + inter_item_spacing * (len(per_item_widths) - 1)
        cur_x = (canvas_w - row_total_w) / 2
        y_row_offset = row_idx * row_height

        # ── draw each legend item ──────────────────────────────────────────
        for item, item_w in zip(row_items, per_item_widths):
            sample_h = row_height * legend_params.height
            sample_y = legend_y + y_row_offset + (row_height - sample_h) / 2

            # Draw the line sample
            if item.linestyle in ("--", ":"):
                dash_len, gap_len = (6, 3) if item.linestyle == "--" else (2, 6)
                pos, end_x = cur_x, cur_x + sample_width
                while pos < end_x:
                    draw.line(
                        [
                            (pos, sample_y + sample_h / 2),
                            (min(pos + dash_len, end_x), sample_y + sample_h / 2),
                        ],
                        fill=item.color,
                        width=int(sample_h * 0.2),
                    )
                    pos += dash_len + gap_len
            else:
                draw.line(
                    [
                        (cur_x, sample_y + sample_h / 2),
                        (cur_x + sample_width, sample_y + sample_h / 2),
                    ],
                    fill=item.color,
                    width=int(sample_h * 0.4),
                )

            # Draw the label text
            text_x = cur_x + sample_width + sample_to_text
            text_y = legend_y + y_row_offset + row_height / 2
            draw.text(
                (text_x, text_y),
                item.label,
                fill="black",
                font=font_legend,
                anchor=legend_params.anchor,
            )

            # Move cursor to start of next item
            cur_x += item_w + inter_item_spacing


TRANSPARENT_WHITE = (255, 255, 255, 0)


def combine_image_grid(
    images_paths_grid: List[List[Path]],
    params: ImageGridParams,
    legend_items: list[LegendItem],
    row_labels: list[str],
    col_labels: list[str],
) -> Image.Image:
    # Fonts
    font_title = _safe_font(bold_dejavu_path, params.font_size)
    font_label = _safe_font(dejavu_path, params.label_font_size)
    font_legend = _safe_font(dejavu_path, params.legend_params.font_size)

    num_rows = len(images_paths_grid)
    num_cols = len(images_paths_grid[0])

    # --------------------------------------------------------------------- #
    #  Calculate canvas size                                                #
    # --------------------------------------------------------------------- #
    #  Title row (optional) + column-label row (optional)
    title_h = params.title_height if params.title else 0

    col_label_h = _get_text_height("TEST", font_label) if params.show_col_labels else 0
    # Ensure column header height is at least 1 if column labels are shown
    padded_col_label_h = max(0, col_label_h + params.column_header_padding)

    # Calculate legend height based on single row height multiplied by number of rows
    single_row_legend_h = _get_text_height("TEST", font_legend) if legend_items else 0

    # Calculate total legend height with rows
    legend_h = 0
    if legend_items:
        # Adjust for actual number of rows needed (minimum of specified rows or number of items)
        actual_rows = min(params.legend_params.rows, len(legend_items))
        legend_h = single_row_legend_h * actual_rows + params.legend_params.legend_padding

        # Add border width to legend height if border is enabled
        if params.legend_params.show_border:
            legend_h += params.legend_params.border_width

    row_label_h = _get_text_height("TEST", font_label) if params.show_row_labels else 0

    top_margin = title_h + padded_col_label_h
    bottom_margin = legend_h

    standard_crop = params.crop_params.standard_crop.box.to_unit("fraction")
    edge_crop = params.crop_params.edge_crop.box.to_unit("fraction")

    left_margin = row_label_h
    right_margin = 0
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
                im = resize_image(im)
                if i == 0 and j == 0:
                    original_image_size = im.size
                else:
                    if not params.allow_different_image_sizes:
                        assert im.size == original_image_size, (
                            f"Image {img_path} has a different size than the first image"
                        )
                if params.crop_params.enable_crop:
                    # Calculate and apply crop
                    crop_box = standard_crop.set_by_mask(
                        edge_crop,
                        {
                            "left": j == 0,
                            "right": j == num_cols - 1,
                            "top": i == 0,
                            "bottom": i == num_rows - 1,
                        },
                    ).to_unit(to_unit_type="absolute", dimensions=im.size)
                    im = im.crop(crop_box.ltrb)
                row_images.append(im)

    if params.crop_params.enable_crop:
        extras = edge_crop.gap(standard_crop).to_unit(to_unit_type="absolute", dimensions=original_image_size).ltrb
        img_w = standard_crop.width * original_image_size[0]
        img_h = standard_crop.height * original_image_size[1]
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
    assert canvas_w + canvas_h < 1e4, f"Canvas size is too large: {canvas_w} + {canvas_h} = {canvas_w + canvas_h}"
    canvas = Image.new("RGBA", (canvas_w, canvas_h), "white")
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
    #  Paint each tile & row labels                                         #
    # --------------------------------------------------------------------- #
    if params.show_row_labels:
        assert len(row_labels) == num_rows
    for row_idx, row_images in enumerate(images):
        y_top = top_margin + row_idx * (img_h + params.padding)
        # Row label (once per row)
        if row_idx == 0:
            y_top -= extras[1]
        if params.show_row_labels:
            label = row_labels[row_idx]

            # Generate prefix based on style
            prefix = generate_prefix(row_idx, params.row_prefix_style)

            # Combine prefix with label if there is a prefix
            if prefix:
                label = f"{prefix} {label}"

            w, h = img_h, row_label_h  # width is the image height (after rotation), height is the label height
            # Draw label on its own small canvas, then rotate 90°
            label_img = Image.new("RGBA", (int(w), h), TRANSPARENT_WHITE)
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

            canvas.paste(
                img,
                (int(x_left), int(y_top)),
            )

    # --------------------------------------------------------------------- #
    #  Draw column labels                                                   #
    # --------------------------------------------------------------------- #
    if params.show_col_labels:
        assert len(col_labels) == num_cols

        for col_idx in range(num_cols):
            prefix = generate_prefix(col_idx, params.column_prefix_style)

            label = col_labels[col_idx]
            if prefix:
                label = f"{prefix} {label}"
            w, h = img_w, col_label_h  # no rotation
            label_img = Image.new("RGBA", (int(w), h), TRANSPARENT_WHITE)
            label_draw = ImageDraw.Draw(label_img)

            bb = label_draw.textbbox((0, 0), label, font=font_label)
            txt_w, txt_h = bb[2] - bb[0], bb[3] - bb[1]

            # Calculate text position, ensuring it's visible even with negative padding
            text_y_pos = max(0, params.column_header_padding)
            if params.column_header_padding < 0:
                text_y_pos = 0

            label_draw.text(((w - txt_w) // 2, text_y_pos), label, fill="black", font=font_label)
            canvas.paste(label_img, (int(left_margin + col_idx * (img_w + params.padding)), title_h), label_img)

    # --------------------------------------------------------------------- #
    #  Draw legend at the bottom                                            #
    # --------------------------------------------------------------------- #
    if legend_items:
        legend_y = int(top_margin + grid_h + extras[3])
        _draw_legend(draw, canvas_w, legend_y, legend_h, legend_items, font_legend, params.legend_params)

    return canvas
