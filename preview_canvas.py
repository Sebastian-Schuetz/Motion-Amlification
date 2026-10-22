from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import tkinter as tk
from PIL import Image, ImageTk


@dataclass(frozen=True)
class ImageFit:
    source_width: int
    source_height: int
    canvas_width: int
    canvas_height: int
    render_width: int
    render_height: int
    offset_x: int
    offset_y: int

    @property
    def scale(self) -> float:
        return self.render_width / self.source_width

    def canvas_to_image(self, x: int, y: int) -> tuple[int, int]:
        image_x = round((x - self.offset_x) / self.scale)
        image_y = round((y - self.offset_y) / self.scale)
        image_x = min(max(image_x, 0), self.source_width - 1)
        image_y = min(max(image_y, 0), self.source_height - 1)
        return image_x, image_y

    def image_to_canvas(self, x: int, y: int) -> tuple[int, int]:
        canvas_x = round(self.offset_x + x * self.scale)
        canvas_y = round(self.offset_y + y * self.scale)
        return canvas_x, canvas_y


class LetterboxCanvas:
    def __init__(self, canvas: tk.Canvas, size: tuple[int, int]) -> None:
        self.canvas = canvas
        self.size = size
        self.fit: Optional[ImageFit] = None
        self._image_ref: Optional[ImageTk.PhotoImage] = None
        self._image_id: Optional[int] = None

    def show_rgb(self, image_rgb: np.ndarray) -> ImageFit:
        if image_rgb.ndim == 2:
            image_rgb = np.repeat(image_rgb[:, :, None], 3, axis=2)

        source_height, source_width = image_rgb.shape[:2]
        canvas_width, canvas_height = self.size
        scale = min(canvas_width / source_width, canvas_height / source_height)
        render_width = max(1, round(source_width * scale))
        render_height = max(1, round(source_height * scale))
        offset_x = (canvas_width - render_width) // 2
        offset_y = (canvas_height - render_height) // 2

        self.fit = ImageFit(
            source_width=source_width,
            source_height=source_height,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            render_width=render_width,
            render_height=render_height,
            offset_x=offset_x,
            offset_y=offset_y,
        )

        self.canvas.delete("all")
        self.canvas.configure(bg="black")

        image = Image.fromarray(image_rgb.astype(np.uint8), mode="RGB")
        resampling = getattr(Image, "Resampling", Image)
        image = image.resize((render_width, render_height), resampling.LANCZOS)
        self._image_ref = ImageTk.PhotoImage(image)
        self._image_id = self.canvas.create_image(
            offset_x,
            offset_y,
            anchor="nw",
            image=self._image_ref,
        )
        return self.fit

    def canvas_to_image(self, x: int, y: int) -> tuple[int, int]:
        if self.fit is None:
            return 0, 0
        return self.fit.canvas_to_image(x, y)

    def image_rect_to_canvas(
        self,
        rect: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        if self.fit is None:
            return rect
        x1, y1 = self.fit.image_to_canvas(rect[0], rect[1])
        x2, y2 = self.fit.image_to_canvas(rect[2], rect[3])
        return x1, y1, x2, y2
