from __future__ import annotations

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Callable

import cv2
import customtkinter as ctk
from CTkToolTip import CTkToolTip

from motion_algorithms import (
    AlgorithmName,
    RieszConfig,
    SteerableConfig,
    create_algorithm,
)
from preview_canvas import LetterboxCanvas
from thresholdWindow import ThresholdPopup


CANVAS_SIZE = (640, 480)
SCALE_VALUES = {
    "1": 1.0,
    "1/2": 0.5,
    "1/4": 0.25,
    "1/8": 0.125,
}


class App(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        self.title("Motion Amplification")
        self.geometry("1000x700")
        self.resizable(False, False)

        self.file_path_var = ctk.StringVar()
        self.algorithm_var = ctk.StringVar(value=AlgorithmName.RIESZ.value)
        self.scale_var = ctk.StringVar(value="1/2")
        self.pyramid_type_var = ctk.StringVar(value="halfOctave")
        self.freq_spectrum_var = ctk.BooleanVar(value=False)
        self.attenuate_var = ctk.BooleanVar(value=False)
        self.total_frames = 0
        self.amplitude_data: str | None = None

        self._build_layout()
        self._set_default_values()
        self._update_algorithm_controls(self.algorithm_var.get())

    def _build_layout(self) -> None:
        ctk.CTkLabel(
            self,
            text="Motion Amplification",
            font=("Arial", 20, "bold"),
        ).pack(pady=10)

        top_frame = ctk.CTkFrame(self)
        top_frame.pack(pady=10, padx=10, fill="both", expand=True)

        image_frame = ctk.CTkFrame(top_frame, width=CANVAS_SIZE[0], height=CANVAS_SIZE[1])
        image_frame.pack(side="left", padx=10, pady=10)

        self.canvas = tk.Canvas(
            image_frame,
            width=CANVAS_SIZE[0],
            height=CANVAS_SIZE[1],
            bg="black",
            highlightthickness=0,
        )
        self.canvas.pack()
        self.preview = LetterboxCanvas(self.canvas, CANVAS_SIZE)

        controls_frame = ctk.CTkFrame(top_frame)
        controls_frame.pack(side="left", padx=10, pady=10, fill="both", expand=True)

        self._build_file_controls(controls_frame)
        self.inputs_frame = ctk.CTkFrame(controls_frame)
        self.inputs_frame.pack(pady=10, fill="x")
        self._build_parameter_controls()
        self._build_action_controls(controls_frame)

    def _build_file_controls(self, parent: ctk.CTkFrame) -> None:
        file_frame = ctk.CTkFrame(parent, fg_color="transparent")
        file_frame.pack(pady=10, fill="x", padx=10)

        ctk.CTkLabel(file_frame, text="File:", width=140, anchor="w").pack(
            side="left",
            padx=(0, 10),
        )
        ctk.CTkButton(file_frame, text="Browse", command=self.browse_file, width=70).pack(
            side="right",
        )
        ctk.CTkEntry(
            file_frame,
            textvariable=self.file_path_var,
            width=180,
            placeholder_text="File path",
        ).pack(side="right", padx=(0, 5))

    def _build_parameter_controls(self) -> None:
        self.algorithm_dropdown = self.add_labeled_dropdown(
            "Algorithm:",
            [AlgorithmName.RIESZ.value, AlgorithmName.STEERABLE.value],
            self.algorithm_var,
            command=self._update_algorithm_controls,
        )
        self.sampling_rate = self.add_labeled_entry("Sampling Rate (Hz):")
        self.low_cutoff = self.add_labeled_entry("Low Cutoff Frequency (Hz):")
        self.high_cutoff = self.add_labeled_entry("High Cutoff Frequency (Hz):")
        self.alpha = self.add_labeled_entry("Alpha:")
        self.scale_dropdown = self.add_labeled_dropdown(
            "Scale:",
            list(SCALE_VALUES.keys()),
            self.scale_var,
        )
        self.start_frame = self.add_labeled_entry("Start Frame:")
        self.end_frame = self.add_labeled_entry("End Frame:")

        self.riesz_frame = ctk.CTkFrame(self.inputs_frame, fg_color="transparent")
        self.riesz_frame.pack(fill="x")
        self.levels = self.add_labeled_entry("Levels:", parent=self.riesz_frame)
        self.gaussian_sd = self.add_labeled_entry(
            "Gaussian Kernel SD:",
            parent=self.riesz_frame,
        )

        self.steerable_frame = ctk.CTkFrame(self.inputs_frame, fg_color="transparent")
        self.steerable_frame.pack(fill="x")
        self.pyramid_dropdown = self.add_labeled_dropdown(
            "Pyramid Type:",
            ["halfOctave", "octave"],
            self.pyramid_type_var,
            parent=self.steerable_frame,
        )
        self.sigma = self.add_labeled_entry("Sigma:", parent=self.steerable_frame)
        self.pyfftw_threads = self.add_labeled_entry("FFTW Threads:", parent=self.steerable_frame)
        self.attenuate_checkbox = ctk.CTkCheckBox(
            self.steerable_frame,
            text="Attenuate Other Frequencies",
            variable=self.attenuate_var,
        )
        self.attenuate_checkbox.pack(pady=6, padx=10, anchor="w")

        self.freq_checkbox = ctk.CTkCheckBox(
            self.inputs_frame,
            text="Frequency Spectrum",
            variable=self.freq_spectrum_var,
        )
        self.freq_checkbox.pack(pady=10, padx=10, anchor="w")
        self._attach_tooltips()

    def _build_action_controls(self, parent: ctk.CTkFrame) -> None:
        action_frame = ctk.CTkFrame(parent, fg_color="transparent")
        action_frame.pack(pady=(10, 20), fill="x", padx=10)

        self.start_button = ctk.CTkButton(
            action_frame,
            text="Start",
            command=self.start_processing,
        )
        self.start_button.pack(fill="x", pady=(0, 10))

        self.progress_bar = ctk.CTkProgressBar(
            action_frame,
            orientation="horizontal",
            mode="determinate",
        )
        self.progress_bar.pack(fill="x")
        self.progress_bar.set(0)

        ctk.CTkButton(
            action_frame,
            text="Load Spectrum File",
            command=self.open_spectrum_window,
        ).pack(fill="x", pady=(15, 10))

    def _attach_tooltips(self) -> None:
        CTkToolTip(self.algorithm_dropdown, "Select the motion amplification algorithm.")
        CTkToolTip(self.sampling_rate, "Frame rate of the input video in Hertz.")
        CTkToolTip(self.low_cutoff, "Lower temporal bandpass cutoff frequency in Hertz.")
        CTkToolTip(self.high_cutoff, "Upper temporal bandpass cutoff frequency in Hertz.")
        CTkToolTip(self.alpha, "Motion magnification factor.")
        CTkToolTip(self.gaussian_sd, "Riesz amplitude-weighted blur sigma.")
        CTkToolTip(self.levels, "Number of Riesz pyramid levels.")
        CTkToolTip(self.sigma, "Steerable amplitude-weighted blur sigma.")
        CTkToolTip(self.pyfftw_threads, "Number of FFTW threads used by the Steerable algorithm.")
        CTkToolTip(self.freq_checkbox, "Save amplitude data for frequency spectrum analysis.")

    def add_labeled_entry(
        self,
        label_text: str,
        parent: ctk.CTkFrame | None = None,
    ) -> ctk.CTkEntry:
        frame = ctk.CTkFrame(parent or self.inputs_frame, fg_color="transparent")
        frame.pack(pady=4, padx=10, fill="x")
        ctk.CTkLabel(frame, text=label_text, width=140, anchor="w").pack(
            side="left",
            padx=(0, 10),
        )
        entry = ctk.CTkEntry(frame, width=180)
        entry.pack(side="right", padx=(0, 10), anchor="e")
        return entry

    def add_labeled_dropdown(
        self,
        label_text: str,
        options: list[str],
        variable: ctk.StringVar,
        parent: ctk.CTkFrame | None = None,
        command: Callable[[str], None] | None = None,
    ) -> ctk.CTkOptionMenu:
        frame = ctk.CTkFrame(parent or self.inputs_frame, fg_color="transparent")
        frame.pack(pady=4, padx=10, fill="x")
        ctk.CTkLabel(frame, text=label_text, width=140, anchor="w").pack(
            side="left",
            padx=(0, 10),
        )
        dropdown = ctk.CTkOptionMenu(
            frame,
            values=options,
            variable=variable,
            width=180,
            command=command,
        )
        dropdown.pack(side="right", padx=(0, 10), anchor="e")
        return dropdown

    def _set_default_values(self) -> None:
        defaults = {
            self.sampling_rate: "30",
            self.low_cutoff: "0.4",
            self.high_cutoff: "3.0",
            self.alpha: "10",
            self.levels: "4",
            self.gaussian_sd: "2",
            self.sigma: "3",
            self.pyfftw_threads: "8",
            self.start_frame: "0",
            self.end_frame: "-1",
        }
        for entry, value in defaults.items():
            entry.insert(0, value)

    def _update_algorithm_controls(self, selected: str) -> None:
        if selected == AlgorithmName.STEERABLE.value:
            self.riesz_frame.pack_forget()
            self.steerable_frame.pack(fill="x", before=self.freq_checkbox)
        else:
            self.steerable_frame.pack_forget()
            self.riesz_frame.pack(fill="x", before=self.freq_checkbox)

    def browse_file(self) -> None:
        file_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")],
        )
        if file_path:
            self.file_path_var.set(file_path)
            self.load_first_frame(file_path)

    def load_first_frame(self, file_path: str) -> None:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            messagebox.showerror("Video Error", "Failed to open video.")
            return

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.total_frames = total
        self._replace_entry_value(self.start_frame, "0")
        self._replace_entry_value(self.end_frame, str(max(total - 1, 0)))

        ret, frame = cap.read()
        cap.release()
        if not ret:
            messagebox.showerror("Video Error", "Failed to load video frame.")
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.preview.show_rgb(frame_rgb)

    def start_processing(self) -> None:
        self.start_button.configure(state="disabled")
        self.progress_bar.set(0)
        threading.Thread(target=self.processing, daemon=True).start()

    def processing(self) -> None:
        try:
            config = self._build_config()
            algorithm = create_algorithm(AlgorithmName(self.algorithm_var.get()))
            self.amplitude_data = algorithm.amplify(config, self.update_progress_bar)
            if config.freq_spectrum and self.amplitude_data:
                self.after(
                    0,
                    lambda: ThresholdPopup(self, self.amplitude_data, config.sampling_rate),
                )
            self.update_progress_bar(1.0)
        except Exception as exc:
            self.after(0, lambda: messagebox.showerror("Processing Error", str(exc)))
        finally:
            self.after(0, lambda: self.start_button.configure(state="normal"))

    def _build_config(self) -> RieszConfig | SteerableConfig:
        video_path = self.file_path_var.get().strip()
        if not video_path:
            raise ValueError("Please select a video file first.")

        base = {
            "video_path": video_path,
            "output_dir": os.path.dirname(video_path),
            "sampling_rate": float(self.sampling_rate.get()),
            "low_cutoff": float(self.low_cutoff.get()),
            "high_cutoff": float(self.high_cutoff.get()),
            "amplification_factor": float(self.alpha.get()),
            "scale": SCALE_VALUES[self.scale_var.get()],
            "start_frame": int(self.start_frame.get()),
            "end_frame": int(self.end_frame.get()),
            "freq_spectrum": self.freq_spectrum_var.get(),
        }

        if self.algorithm_var.get() == AlgorithmName.STEERABLE.value:
            return SteerableConfig(
                **base,
                pyramid_type=self.pyramid_type_var.get(),
                sigma=int(self.sigma.get()),
                attenuate_other_frequencies=self.attenuate_var.get(),
                pyfftw_threads=int(self.pyfftw_threads.get()),
            )

        return RieszConfig(
            **base,
            levels=int(self.levels.get()),
            gaussian_kernel_sd=int(self.gaussian_sd.get()),
        )

    def update_progress_bar(self, percent: float) -> None:
        percent = min(max(percent, 0.0), 1.0)
        self.after(0, lambda: self.progress_bar.set(percent))

    def open_spectrum_window(self) -> None:
        ThresholdPopup(self)

    @staticmethod
    def _replace_entry_value(entry: ctk.CTkEntry, value: str) -> None:
        entry.delete(0, "end")
        entry.insert(0, value)
