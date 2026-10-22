from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import customtkinter as ctk
import h5py
import matplotlib.cm as cm
import numpy as np
from CTkToolTip import CTkToolTip

from phaseAmplify import RieszMotionAmplifier
from preview_canvas import LetterboxCanvas


CANVAS_SIZE = (640, 480)


class ThresholdPopup(ctk.CTkToplevel):
    def __init__(
        self,
        master: ctk.CTk,
        h5_filepath: str | None = None,
        sampling_rate: float = 30.0,
    ) -> None:
        super().__init__(master)
        self.title("Threshold Selector")
        self.geometry("1000x700")
        self.resizable(False, False)

        self.sampling_rate = sampling_rate
        self.h5_filepath = h5_filepath
        self.file_path_var = tk.StringVar(value=h5_filepath or "")
        self.sampling_rate_var = tk.StringVar(value=str(sampling_rate))

        self.original_image: np.ndarray | None = None
        self.original_color_image: np.ndarray | None = None
        self.mask: np.ndarray | None = None
        self.rect: tuple[int, int, int, int] | None = None
        self.drag_start: tuple[int, int] | None = None
        self.rect_id: int | None = None
        self.preview: LetterboxCanvas | None = None
        self.riesz = RieszMotionAmplifier()

        self._build_widgets()
        self.focus()
        self.grab_set()

        if h5_filepath:
            self.load_h5_file(h5_filepath)

    def _build_widgets(self) -> None:
        left_frame = ctk.CTkFrame(self, width=CANVAS_SIZE[0], height=CANVAS_SIZE[1])
        left_frame.pack(side="left", padx=20, pady=20)

        right_frame = ctk.CTkFrame(self)
        right_frame.pack(side="right", fill="y", padx=20, pady=20)

        self.canvas = tk.Canvas(
            left_frame,
            width=CANVAS_SIZE[0],
            height=CANVAS_SIZE[1],
            bg="black",
            highlightthickness=0,
        )
        self.canvas.pack()
        self.preview = LetterboxCanvas(self.canvas, CANVAS_SIZE)
        self.canvas.bind("<ButtonPress-1>", self.start_rectangle)
        self.canvas.bind("<B1-Motion>", self.draw_rectangle)
        self.canvas.bind("<ButtonRelease-1>", self.end_rectangle)

        ctk.CTkLabel(right_frame, text="HDF5 File:", anchor="w").pack(
            pady=(10, 0),
            padx=10,
            anchor="w",
        )
        ctk.CTkEntry(
            right_frame,
            textvariable=self.file_path_var,
            width=250,
            placeholder_text="Select .h5 file",
        ).pack(padx=10, pady=(0, 5))
        ctk.CTkButton(
            right_frame,
            text="Browse",
            command=self.browse_file,
            width=100,
        ).pack(pady=(0, 20))

        ctk.CTkLabel(
            right_frame,
            text="Rectangle Coordinates",
            font=("Arial", 14, "bold"),
        ).pack(pady=(10, 5))
        self.coords_label = ctk.CTkLabel(right_frame, text="(x1, y1) -> (x2, y2)")
        self.coords_label.pack(pady=(0, 20))

        ctk.CTkLabel(right_frame, text="Threshold", font=("Arial", 14, "bold")).pack(
            pady=(10, 5),
        )
        self.slider_value_label = ctk.CTkLabel(right_frame, text="Threshold: 128")
        self.slider_value_label.pack(pady=(0, 10))
        self.slider = ctk.CTkSlider(
            right_frame,
            from_=0,
            to=255,
            command=self.on_threshold_change,
            width=200,
        )
        self.slider.set(128)
        self.slider.pack(pady=(0, 20))
        CTkToolTip(
            self.slider,
            "Adjust the threshold for pixel selection inside the rectangle.",
        )

        ctk.CTkLabel(
            right_frame,
            text="Sampling Rate (Hz)",
            font=("Arial", 14, "bold"),
        ).pack(pady=(10, 5))
        ctk.CTkEntry(
            right_frame,
            textvariable=self.sampling_rate_var,
            width=100,
        ).pack(pady=(0, 30))

        self.ok_button = ctk.CTkButton(
            right_frame,
            text="OK",
            command=self.return_mask,
            state="disabled",
        )
        self.ok_button.pack(pady=(10, 10))

    def browse_file(self) -> None:
        file_path = filedialog.askopenfilename(filetypes=[("HDF5 files", "*.h5")])
        if file_path:
            self.file_path_var.set(file_path)
            self.load_h5_file(file_path)

    def load_h5_file(self, file_path: str) -> None:
        self.h5_filepath = file_path

        with h5py.File(file_path, "r") as h5_file:
            amplitudes = h5_file["level_0"]["amplitudes"]
            shape = amplitudes.attrs.get("shape_0")
            if shape is None:
                raise ValueError("Shape attribute 'shape_0' not found for first frame.")
            image_array = np.array(amplitudes[0]).reshape(shape)

        if np.issubdtype(image_array.dtype, np.floating):
            normalized = self.normalize_image(image_array)
            self.original_image = normalized
            self.original_color_image = self.apply_colormap(normalized, cmap_name="inferno")
        else:
            self.original_image = image_array.astype(np.uint8)
            self.original_color_image = cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2RGB)

        self.mask = np.zeros_like(self.original_image, dtype=bool)
        self.rect = None
        self.update_display()
        self.ok_button.configure(state="normal")

    def start_rectangle(self, event: tk.Event) -> None:
        if self.original_image is None or self.preview is None:
            return
        self.drag_start = self.preview.canvas_to_image(event.x, event.y)
        self.rect = None
        self.update_display()

    def draw_rectangle(self, event: tk.Event) -> None:
        if self.drag_start is None or self.preview is None:
            return
        current = self.preview.canvas_to_image(event.x, event.y)
        self.rect = self._normalize_rect(self.drag_start, current)
        self.update_display()

    def end_rectangle(self, event: tk.Event) -> None:
        if self.drag_start is None or self.preview is None:
            return
        current = self.preview.canvas_to_image(event.x, event.y)
        self.rect = self._normalize_rect(self.drag_start, current)
        self.drag_start = None
        self._update_coordinate_label()
        self.update_slider_range()
        self.update_display()

    def update_slider_range(self) -> None:
        if self.rect is None or self.original_image is None:
            return
        x1, y1, x2, y2 = self.rect
        region = self.original_image[y1:y2, x1:x2]
        self.slider.configure(to=int(region.max()) if region.size > 0 else 255)

    def on_threshold_change(self, value: float) -> None:
        self.slider_value_label.configure(text=f"Threshold: {int(value)}")
        self.update_display()

    def update_display(self) -> None:
        if self.original_color_image is None or self.original_image is None or self.preview is None:
            return

        color_img = self.original_color_image.copy()
        threshold = int(self.slider.get())

        if self.rect is not None:
            x1, y1, x2, y2 = self.rect
            region = self.original_image[y1:y2, x1:x2]
            if region.size > 0:
                selected = region > threshold
                self.mask = np.zeros_like(self.original_image, dtype=bool)
                self.mask[y1:y2, x1:x2] = selected
                color_img[self.mask] = [0, 255, 0]

        self.preview.show_rgb(color_img)
        if self.rect is not None:
            rect_canvas = self.preview.image_rect_to_canvas(self.rect)
            self.canvas.create_rectangle(*rect_canvas, outline="red", width=2)

    def normalize_image(self, img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float32)
        img = img - np.min(img)
        max_value = np.max(img)
        if max_value != 0:
            img = img / max_value
        return (img * 255).astype(np.uint8)

    def apply_colormap(self, gray_img: np.ndarray, cmap_name: str = "hot") -> np.ndarray:
        cmap = cm.get_cmap(cmap_name)
        colored = cmap(gray_img / 255.0)
        return (colored[:, :, :3] * 255).astype(np.uint8)

    def return_mask(self) -> None:
        if self.h5_filepath is None or self.mask is None:
            return
        try:
            sampling_rate = float(self.sampling_rate_var.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Sampling rate must be a number.")
            return

        np.save("mask_array.npy", self.mask)
        self.riesz.analyze(self.h5_filepath, self.mask, sampling_rate)

    def _update_coordinate_label(self) -> None:
        if self.rect is None:
            self.coords_label.configure(text="(x1, y1) -> (x2, y2)")
            return
        x1, y1, x2, y2 = self.rect
        self.coords_label.configure(text=f"({x1}, {y1}) -> ({x2}, {y2})")

    @staticmethod
    def _normalize_rect(
        start: tuple[int, int],
        end: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        x1, y1 = start
        x2, y2 = end
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
