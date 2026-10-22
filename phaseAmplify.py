import os
import numpy as np
import cv2
from pyramidTools import PyramidTools
from typing import Callable, Optional
import h5py

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fft import fft, fftfreq
from scipy.signal import butter

class RieszMotionAmplifier:
    def __init__(self):
        self.pyramid_tools = PyramidTools()
        
    def amplify(self,
        amplification_factor: float,
        low_cutoff: float,
        high_cutoff: float,
        sampling_rate: float,
        video_path: str,
        output_dir: str,
        num_levels: int =4,
        gaussian_kernel_sd: int =2,
        scale: float = 1.0,
        update_progress: Optional[Callable[[float], None]] = None,
        start_frame: int = 0,
        end_frame: int = -1,
        freq_spectrum: bool = False):
        """
        Performs Eulerian video magnification using the Riesz pyramid representation
        to amplify subtle motion in a specific frequency band.

        This function applies phase-based motion magnification by computing the Riesz pyramid 
        for each frame, estimating phase differences, applying IIR filtering on phase, and 
        reconstructing the motion-magnified video.

        Parameters:
            amplification_factor (float): Factor by which to amplify detected motion.
            low_cutoff (float): Low temporal cutoff frequency (Hz).
            high_cutoff (float): High temporal cutoff frequency (Hz).
            sampling_rate (float): Frame rate of the input video (Hz).
            video_path (str): Path to the input video file.
            output_dir (str): Directory to store the output video file.
            num_levels (int): Number of levels in the Laplacian/Riesz pyramid.
            gaussian_kernel_sd (int): Standard deviation of the Gaussian kernel used for amplitude-weighted blurring.
            scale (float): Scale factor to resize frames for processing.
            update_progress (Callable[[float], None], optional): Callback function to report progress (0.0 to 1.0).
            start_frame (int): Frame index to start processing from.
            end_frame (int): Frame index to stop processing at (-1 means until the end of the video).
            freq_spectrum (bool): If True, saves the amplitude date for each pyramid level to a file for further processing

        Raises:
            RuntimeError: If the video cannot be opened or read.

        Returns:
            Optional[str]: If `freq_spectrum` is True, returns the file path (str) to the saved HDF5 file
            containing the phase amplitude data. Otherwise, returns None.
        """

        # === Compute temporal IIR filter coefficients ===
        nyquist_frequency = sampling_rate / 2
        temporal_filter_order = 1
        B, A = butter(
            temporal_filter_order,
            [low_cutoff / nyquist_frequency, high_cutoff / nyquist_frequency],
            btype='bandpass',
        )

        # === Open input video ===
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Cannot open video file")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # === Prepare output video writer ===
        os.makedirs(output_dir, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        # Generate output file name based on parameters
        writeTag = os.path.splitext(os.path.basename(video_path))[0]
        out_name = (f"{writeTag}-band{low_cutoff:.2f}-{high_cutoff:.2f}Hz-sr{sampling_rate}Hz-"
                f"alpha{amplification_factor}-levels{num_levels}-"
                f"gausian_sd{gaussian_kernel_sd}-scale{scale:.2f}")

        out_path = os.path.join(output_dir, out_name + ".avi")
        out = cv2.VideoWriter(out_path, fourcc, fps, (int(frame_width), int(frame_height)))

        # === Read and preprocess the first frame ===
        i = 0
        while True:
            ret, previous_frame = cap.read()
            if not ret:
                raise RuntimeError("Cannot read first frame from video")
            if i == start_frame:
                break
            i += 1

        # Convert first frame to grayscale and scale
        # Change color channels, because openCV uses BGR and gram the luma (Y) channel from the image
        # only apply scaling to the gray component
        previous_frame_gray = cv2.resize(cv2.cvtColor(previous_frame, cv2.COLOR_BGR2YCrCb)[:,:,0], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)  
        previous_frame_gray = previous_frame_gray.astype(np.float32) / 255.0  

        # === Compute initial Riesz pyramid ===
        previous_laplacian_pyramid, previous_riesz_x, previous_riesz_y = self.pyramid_tools.ComputeRieszPyramid(previous_frame_gray, num_levels)
        number_of_levels = len(previous_laplacian_pyramid) - 1  # exclude lowpass residual

        # === Initialize phase estimates and filter registers ===
        phase_cos = [np.zeros_like(previous_laplacian_pyramid[k]) for k in range(number_of_levels)]
        phase_sin = [np.zeros_like(previous_laplacian_pyramid[k]) for k in range(number_of_levels)]

        register0_cos = [np.zeros_like(previous_laplacian_pyramid[k]) for k in range(number_of_levels)]
        register1_cos = [np.zeros_like(previous_laplacian_pyramid[k]) for k in range(number_of_levels)]
        register0_sin = [np.zeros_like(previous_laplacian_pyramid[k]) for k in range(number_of_levels)]
        register1_sin = [np.zeros_like(previous_laplacian_pyramid[k]) for k in range(number_of_levels)]

        # === Create a list, that saves the phase information in one pixel ===
        #phase_levels_cos = [[] for _ in range(number_of_levels)]
        #phase_levels_sin = [[] for _ in range(number_of_levels)]
        #phase_levels_amp = [[] for _ in range(number_of_levels)]
        if freq_spectrum:
            h5_file = h5py.File(os.path.join(output_dir, out_name + "_spek.h5"), "w")
                # Prepare datasets for each level if not exist
            datasets = []
            for level in range(num_levels):
                group_name = f'level_{level}'
                if group_name not in h5_file:
                    grp = h5_file.create_group(group_name)
                    # Create a dataset to store variable length arrays (flattened)
                    dt = h5py.special_dtype(vlen=np.dtype('float32'))
                    dset = grp.create_dataset(
                        'amplitudes', shape=(0,), maxshape=(None,), dtype=dt, chunks=True)
                else:
                    dset = h5_file[group_name]['amplitudes']
                datasets.append(dset)

    # === Main processing loop ===
        while True:
            ret, current_frame = cap.read()
            if not ret:
                break  # end of video
            if (end_frame != -1) and (i == end_frame):
                break
            
            # Convert to grayscale and resize
            current_frame_gray = cv2.resize(cv2.cvtColor(current_frame, cv2.COLOR_BGR2YCrCb)[:,:,0], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            current_frame_gray = current_frame_gray.astype(np.float32) / 255.0

            # Compute Riesz pyramid of current frame
            current_laplacian_pyramid, current_riesz_x, current_riesz_y = self.pyramid_tools.ComputeRieszPyramid(current_frame_gray, num_levels)

            motion_magnified_laplacian_pyramid = []

            for k in range(number_of_levels):
                # === Phase difference and amplitude computation ===
                phase_diff_cos, phase_diff_sin, amplitude = self.pyramid_tools.ComputePhaseDifferenceAndAmplitude(
                    current_laplacian_pyramid[k], current_riesz_x[k], current_riesz_y[k],
                    previous_laplacian_pyramid[k], previous_riesz_x[k], previous_riesz_y[k]
                )

                # === Accumulate phase over time ===
                phase_cos[k] += phase_diff_cos
                phase_sin[k] += phase_diff_sin

                # === Temporal filtering ===
                phase_filtered_cos, register0_cos[k], register1_cos[k] = self.pyramid_tools.IIRTemporalFilter(
                    B, A, phase_cos[k], register0_cos[k], register1_cos[k]
                )
                phase_filtered_sin, register0_sin[k], register1_sin[k] = self.pyramid_tools.IIRTemporalFilter(
                    B, A, phase_sin[k], register0_sin[k], register1_sin[k]
                )

                # === Save the phase-information ===
                if freq_spectrum:
                    #phase_levels_cos[k].append(phase_diff_cos[int(((0.5)**k)*317),int(((0.5)**k)*334)])
                    #phase_levels_sin[k].append(phase_diff_sin[int(((0.5)**k)*317),int(((0.5)**k)*334)])
                    #phase_levels_amp[k].append(amplitude)
                    amplitude = amplitude.astype(np.float32)
                    # Flatten the amplitude array for storage
                    flat_amp = amplitude.flatten()
                    dset = datasets[k]

                    # Resize dataset by 1 and append
                    dset.resize((dset.shape[0] + 1,))
                    dset[-1] = flat_amp

                    # Save original shape as attribute named after index
                    attr_name = f'shape_{dset.shape[0]-1}'
                    dset.attrs[attr_name] = amplitude.shape


                # === Spatial blur with amplitude weighting ===
                phase_filtered_cos = self.pyramid_tools.AmplitudeWeightedBlur(phase_filtered_cos, amplitude, gaussian_kernel_sd)
                phase_filtered_sin = self.pyramid_tools.AmplitudeWeightedBlur(phase_filtered_sin, amplitude, gaussian_kernel_sd)

                # === Phase amplification ===
                phase_magnified_filtered_cos = amplification_factor * phase_filtered_cos
                phase_magnified_filtered_sin = amplification_factor * phase_filtered_sin

                # === Reconstruct motion-magnified pyramid level ===
                motion_mag_lap_k = self.pyramid_tools.PhaseShiftCoefficientRealPart(
                    current_laplacian_pyramid[k], current_riesz_x[k], current_riesz_y[k],
                    phase_magnified_filtered_cos, phase_magnified_filtered_sin
                )

                motion_magnified_laplacian_pyramid.append(motion_mag_lap_k)

            # Append the lowpass residual unchanged
            motion_magnified_laplacian_pyramid.append(current_laplacian_pyramid[number_of_levels])

            # Collapse pyramid to get magnified frame
            motion_magnified_frame = self.pyramid_tools.CollapseLaplacianPyramid(motion_magnified_laplacian_pyramid)
            motion_magnified_frame = cv2.resize(motion_magnified_frame, None, fx=1.0/scale, fy=1.0/scale, interpolation=cv2.INTER_CUBIC)

            # Build output frame
            original_frame = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2YCrCb)
            out_frame = np.zeros_like(original_frame)
            out_frame[:, :, 0] = np.clip(motion_magnified_frame * 255.0, 0, 255).astype(np.uint8)
            out_frame[:, :, 1:] = original_frame[:, :, 1:]

            # Convert back to 8-bit grayscale for output
            motion_magnified_frame_8bit = cv2.cvtColor(out_frame, cv2.COLOR_YCrCb2BGR)

            # Write output frame
            out.write(motion_magnified_frame_8bit)

            # === Prepare for next frame ===
            previous_laplacian_pyramid = current_laplacian_pyramid
            previous_riesz_x = current_riesz_x
            previous_riesz_y = current_riesz_y
            i += 1

            # === Update progress if callback is provided ===
            if update_progress is not None:
                update_progress(float(i - start_frame)/float(end_frame - start_frame))

        # === Cleanup ===
        cap.release()
        out.release()
        if freq_spectrum:
            h5_file.close()
            return os.path.join(output_dir, out_name + "_spek.h5")
    

    def analyze(self, h5_filepath: str, mask: np.ndarray, sampling_rate: float = 30.0, filepath_plot: str = ""):
        """
        Analyze and visualize average amplitude signals from spatially masked regions in a multiscale HDF5 dataset.

        This function processes a multilevel amplitude dataset stored in an HDF5 file, applying a binary mask 
        (which defines a region of interest) to each level. It computes the average amplitude over time within 
        the masked region, then visualizes:
        - Temporal traces of the average amplitudes per level.
        - Frequency spectra (via FFT) of the same signals.

        Parameters:
            h5_filepath (str): Path to the HDF5 file containing amplitude datasets organized by level (e.g., 'level_0', 'level_1', ...).
            mask (np.ndarray): A 2D boolean mask (same spatial dimensions as original full-resolution amplitude arrays).
            sampling_rate (float, optional): Frame rate (in Hz) used to determine time axis and FFT frequency scale. Defaults to 30.0.
        """

        # Suppress specific UserWarnings (e.g., from seaborn/matplotlib or h5py)
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        import pandas as pd
        
        # Set seaborn style for plots
        sns.set(style="whitegrid")

        level_averages = []

        # --- Load HDF5 data and compute masked averages ---
        with h5py.File(h5_filepath, 'r') as f:
            # Sort levels by numerical order (e.g., 'level_0', 'level_1', ...)
            levels = sorted(f.keys(), key=lambda x: int(x.split('_')[1]))

            for level_key in levels:
                group = f[level_key]
                amplitudes = group['amplitudes']
                n_frames = amplitudes.shape[0]

                # Get frame shape from dataset attribute
                frame_shape = amplitudes.attrs.get('shape_0')
                if frame_shape is None:
                    raise ValueError(f"Missing shape attribute in {level_key} dataset.")

                level_h, level_w = frame_shape
                mask_h, mask_w = mask.shape

                # Resize input mask to match current level's resolution
                resized_mask = cv2.resize(
                    mask.astype(np.uint8),
                    (level_w, level_h),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)

                if np.any(resized_mask):
                    

                    # Compute average amplitude within the masked region for each frame
                    averages = []
                    for i in range(n_frames):
                        arr = np.array(amplitudes[i]).reshape(frame_shape)
                        masked_vals = arr[resized_mask]
                        avg = masked_vals.mean() if masked_vals.size > 0 else np.nan
                        averages.append(avg)

                    level_averages.append(np.array(averages))
                else:
                    print(f"Skipping level {level_key}")

        n_levels = len(level_averages)
        time = np.arange(len(level_averages[0])) / sampling_rate

        # --- Prepare FFT data for seaborn ---
        fft_data = []
        sum_magnitude = None
        freqs = None

        for lvl_idx, avg_trace in enumerate(level_averages):
            # Remove DC component and handle NaNs
            signal = avg_trace - np.nanmean(avg_trace)
            signal = np.nan_to_num(signal)

            # Compute FFT
            spectrum = np.abs(fft(signal))
            n = len(signal)
            xf = fftfreq(n, d=1 / sampling_rate)
            pos_mask = xf >= 0

            freqs = xf[pos_mask]
            mags = spectrum[pos_mask]

            # Sum FFT magnitudes across levels
            if sum_magnitude is None:
                sum_magnitude = mags.copy()
            else:
                sum_magnitude += mags

            # Store FFT data for seaborn plotting
            for freq, mag in zip(freqs, mags):
                fft_data.append({
                    "Frequency (Hz)": freq,
                    "Magnitude": mag,
                    "Level": f"Level {lvl_idx}"
                })

        # Convert FFT data to DataFrame for seaborn
        fft_df = pd.DataFrame(fft_data)

        # --- Plotting ---

        fig, (ax_time, ax_fft) = plt.subplots(
            nrows=2,
            figsize=(14, 8),
            constrained_layout=True
        )

        # Top: Time series
        for lvl_idx, avg_trace in enumerate(level_averages):
            ax_time.plot(time, avg_trace, label=f'Level {lvl_idx}')
        ax_time.set_title('Average Amplitude Over Time (Masked Region)')
        ax_time.set_xlabel('Time (seconds)')
        ax_time.set_ylabel('Average Amplitude')
        ax_time.legend(loc='upper right')

        # Bottom: Overlaid FFTs using seaborn
        sns.lineplot(
            data=fft_df,
            x="Frequency (Hz)",
            y="Magnitude",
            hue="Level",
            ax=ax_fft,
            palette="tab10"
        )

        # Plot the summed FFT magnitude over all levels
        ax_fft.plot(freqs, sum_magnitude, color='black', linewidth=1.5, linestyle='--', label='Sum')

        ax_fft.set_title('FFT Magnitude (All Levels + Sum)')
        ax_fft.set_xlim(0, sampling_rate / 2)
        ax_fft.set_xlabel("Frequency (Hz)")
        ax_fft.set_ylabel("Magnitude")
        ax_fft.legend()
        if not (filepath_plot == ""):
            plt.savefig(filepath_plot)
        plt.show()


PhaseAmplifiy = RieszMotionAmplifier
