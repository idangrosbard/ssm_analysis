from enum import Enum
from pathlib import Path
from typing import Any, List, Optional, Tuple, TypeVar, Union

from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field
from pydantic_extra_types.color import Color

FONT_BASE = lambda x: f"/usr/share/fonts/truetype/liberation/LiberationSans{x}.ttf"  # noqa
FONT_REGULAR = FONT_BASE("-Regular")
FONT_BOLD = FONT_BASE("-Bold")


class ImageGridParams(BaseModel):
    padding: int = 10
    title_height: int = 30  # Height for titles
    background_color: str = "white"
    show_titles: bool = True
    font_size: int = 25
    img_width: int = Field(-1, ge=-1, description="If -1, will use max width of images")
    img_height: int = Field(-1, ge=-1, description="If -1, will use max height of images")
    color_dict: dict[str, Color] = Field(
        default_factory=lambda: {
            "gpt-4o": Color((255, 0, 0, 0.5)),
        }
    )
    row_labels: Optional[list[str]] = None
    col_labels: Optional[list[str]] = None
    label_width: int = 200  # Width for row labels
    label_height: int = 50  # Height for column labels
    label_font_size: int = 20


T = TypeVar("T")
R = TypeVar("R")


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
    grid_params: ImageGridParams = Field(
        default_factory=lambda: ImageGridParams(img_width=-1, img_height=-1), description="Parameters for the grid"
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


def combine_image_grid(images: list[list[Path]], params: ImageGridParams):
    """Combine images into a grid layout with optional row and column labels.

    Args:
        images: List of lists of image paths. Each inner list represents a row in the grid.
               Each row should have the same number of columns.
        params: Parameters for grid creation including optional row and column labels

    Returns:
        Combined image with all input images arranged in a grid
    """
    if not images:
        return None

    # Calculate max dimensions if not specified
    if params.img_width == -1 or params.img_height == -1:
        max_width = 0
        max_height = 0
        for row in images:
            for img_path in row:
                if img_path is not None:  # Some grid positions might be empty
                    with Image.open(img_path) as img:
                        max_width = max(max_width, img.width)
                        max_height = max(max_height, img.height)

        if params.img_width == -1:
            params.img_width = max_width
        if params.img_height == -1:
            params.img_height = max_height

    # Calculate grid dimensions
    num_rows = len(images)
    num_cols = max(len(row) for row in images) if images else 0

    # Calculate canvas dimensions including space for labels
    left_margin = params.label_width if params.row_labels else 0
    top_margin = params.label_height if params.col_labels else 0

    canvas_width = left_margin + num_cols * (params.img_width + params.padding) - params.padding
    canvas_height = top_margin + num_rows * (params.img_height + params.title_height + params.padding) - params.padding

    # Create blank canvas
    combined_image = Image.new("RGB", (canvas_width, canvas_height), params.background_color)
    draw = ImageDraw.Draw(combined_image)

    # Try to load fonts
    title_font = ImageFont.truetype(FONT_REGULAR, params.font_size)
    label_font = ImageFont.truetype(FONT_BOLD, params.label_font_size)

    # Add column labels if provided
    if params.col_labels:
        for col_idx, col_label in enumerate(params.col_labels[:num_cols]):
            x_offset = left_margin + col_idx * (params.img_width + params.padding)
            # Center the text in the column
            text_bbox = draw.textbbox((0, 0), col_label, font=label_font)
            text_width = text_bbox[2] - text_bbox[0]
            x_text = x_offset + (params.img_width - text_width) // 2
            draw.text(
                (x_text, 5),
                col_label,
                fill="black",
                font=label_font,
            )

    # Place images on canvas and add row labels
    for row_idx, row in enumerate(images):
        y_offset = top_margin + row_idx * (params.img_height + params.title_height + params.padding)

        # Add row label if provided
        if params.row_labels and row_idx < len(params.row_labels):
            # Center the text vertically in the row
            text_bbox = draw.textbbox((0, 0), params.row_labels[row_idx], font=label_font)
            text_height = text_bbox[3] - text_bbox[1]
            y_text = y_offset + (params.img_height - text_height) // 2
            draw.text(
                (5, y_text),
                params.row_labels[row_idx],
                fill="black",
                font=label_font,
            )

        for col_idx, img_path in enumerate(row):
            x_offset = left_margin + col_idx * (params.img_width + params.padding)

            if img_path is not None:
                with Image.open(img_path) as img:
                    # Resize image if needed
                    if img.size != (params.img_width, params.img_height):
                        img = img.resize((params.img_width, params.img_height))

                    # Paste image onto canvas
                    combined_image.paste(img, (x_offset, y_offset + params.title_height))

                    # Add title if enabled
                    if params.show_titles:
                        # Extract model info from path
                        title = img_path.parent.parent.name
                        draw.text(
                            (x_offset + 5, y_offset + 5),
                            title,
                            fill="black",
                            font=title_font,
                        )

    return combined_image
