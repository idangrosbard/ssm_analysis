from enum import Enum
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field

FONT_BASE = lambda x: f"/usr/share/fonts/truetype/liberation/LiberationSans{x}.ttf"  # noqa
FONT_REGULAR = FONT_BASE("-Regular")
FONT_BOLD = FONT_BASE("-Bold")


class ColorSelection(Enum):
    RANDOM = "random"
    SEQUENTIAL = "sequential"


class ImageGridParams(BaseModel):
    padding: int = 10
    title_height: int = 30  # Height for titles
    background_color: str = "white"
    show_titles: bool = True
    font_size: int = 25
    img_width: int = Field(-1, ge=-1, description="If -1, will use max width of images")
    img_height: int = Field(-1, ge=-1, description="If -1, will use max height of images")
    one_color: ColorSelection = ColorSelection.RANDOM


def combine_image_grid(images: list[list[Path]], params: ImageGridParams):
    """Combine images into a grid layout.

    Args:
        images: List of lists of image paths. Each inner list represents a row in the grid.
               Each row should have the same number of columns.
        params: Parameters for grid creation

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

    # Calculate canvas dimensions
    canvas_width = num_cols * (params.img_width + params.padding) - params.padding
    canvas_height = num_rows * (params.img_height + params.title_height + params.padding) - params.padding

    # Create blank canvas
    combined_image = Image.new("RGB", (canvas_width, canvas_height), params.background_color)
    draw = ImageDraw.Draw(combined_image)

    # Try to load font for titles
    font = ImageFont.truetype(FONT_REGULAR, params.font_size)

    # Place images on canvas
    for row_idx, row in enumerate(images):
        y_offset = row_idx * (params.img_height + params.title_height + params.padding)

        for col_idx, img_path in enumerate(row):
            x_offset = col_idx * (params.img_width + params.padding)

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
                            font=font,
                        )

    return combined_image
