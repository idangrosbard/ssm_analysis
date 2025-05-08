from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class Crop(BaseModel):
    """Represents crop coordinates as normalized values (0.0-1.0)."""

    left: float = Field(default=0.0, ge=0.0, le=1.0, description="Crop from left (0.0-1.0)")
    top: float = Field(default=0.0, ge=0.0, le=1.0, description="Crop from top (0.0-1.0)")
    width: float = Field(default=0.9, ge=0.0, le=1.0, description="Crop from right (0.0-1.0)")
    height: float = Field(default=0.9, ge=0.0, le=1.0, description="Crop from bottom (0.0-1.0)")

    @model_validator(mode="after")
    def _validate_not_too_small(self):
        if self.width < 0.1:
            self.width = 0.9
        if self.height < 0.1:
            self.height = 0.9

        return self
