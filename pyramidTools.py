import numpy as np
from scipy.ndimage import convolve
import cv2

# The following functions are taken from the pseudo code
# provided with Wadhwa et. al (2014) and were optimized where possible

class PyramidTools:
    def __init__(self):
        pass
    def IIRTemporalFilter(self, B, A, phase, register0, register1):
        """
        Assumes all inputs are contiguous NumPy arrays of same shape.
        """
        filtered_phase = B[0] * phase + register0
        new_register0 = B[1] * phase + register1 - A[1] * filtered_phase
        new_register1 = B[2] * phase - A[2] * filtered_phase
        return filtered_phase, new_register0, new_register1

    def AmplitudeWeightedBlur(self, filtered_phase: np.ndarray, amplitude: np.ndarray, sigma: float) -> np.ndarray:
        """
        Applies an amplitude-weighted spatial Gaussian blur to a filtered phase signal.

        This function is used to spatially smooth the phase signal while preserving 
        spatial structure by weighting the blur based on local amplitude (signal strength).
        It performs a weighted Gaussian blur: 
            smoothed = GaussianBlur(phase * amplitude) / GaussianBlur(amplitude)

        Parameters:
            filtered_phase (np.ndarray):
                2D array of temporally filtered phase values.
            amplitude (np.ndarray):
                2D array representing local amplitude or confidence in the phase values.
            sigma (float):
                Standard deviation of the Gaussian kernel used for spatial smoothing (in pixels).

        Returns:
            np.ndarray:
                A smoothed phase array of the same shape as the input, with spatial consistency
                guided by amplitude weighting.

        Notes:
            - Uses OpenCV's efficient `GaussianBlur` function, because it is way faster than scipy
            - Handles division by zero gracefully by setting invalid outputs to zero.
        """
        # === Step 1: Ensure inputs are in float32 format for OpenCV compatibility ===
        filtered_phase = filtered_phase.astype(np.float32)
        amplitude = amplitude.astype(np.float32)

        # === Step 2: Compute numerator of the weighted blur: amplitude * phase ===
        weighted_phase = filtered_phase * amplitude

        # === Step 3: Apply Gaussian blur to numerator and denominator separately ===
        # OpenCV computes kernel size automatically from sigma if ksize=(0, 0)
        numerator = cv2.GaussianBlur(weighted_phase, ksize=(0, 0), sigmaX=sigma, borderType=cv2.BORDER_REFLECT)
        denominator = cv2.GaussianBlur(amplitude, ksize=(0, 0), sigmaX=sigma, borderType=cv2.BORDER_REFLECT)

        # === Step 4: Divide with care (avoid division by zero) ===
        with np.errstate(divide='ignore', invalid='ignore'):
            smoothed_phase = numerator / denominator
            smoothed_phase[denominator == 0] = 0 # Set undefined regions to zero

        return smoothed_phase

    def ComputePhaseDifferenceAndAmplitude(self, 
        current_real, current_x, current_y,
        previous_real, previous_x, previous_y
    ):
        """
        Computes the quaternionic phase difference and amplitude between two Riesz pyramid frames.

        It estimates the local phase difference between consecutive frames by computing 
        the quaternionic conjugate product of Riesz coefficients, and extracts both 
        directional phase differences and local amplitude.

        Parameters:
            current_real (np.ndarray): 
                Real part of the current frame's Riesz coefficient.
            current_x (np.ndarray): 
                X-direction (i-component) of the current frame's Riesz coefficient.
            current_y (np.ndarray): 
                Y-direction (j-component) of the current frame's Riesz coefficient.
            previous_real (np.ndarray): 
                Real part of the previous frame's Riesz coefficient.
            previous_x (np.ndarray): 
                X-direction of the previous frame's Riesz coefficient.
            previous_y (np.ndarray): 
                Y-direction of the previous frame's Riesz coefficient.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - phase_diff_cos: Phase difference component along the x-direction.
                - phase_diff_sin: Phase difference component along the y-direction.
                - amplitude: Local amplitude estimate for phase weighting.
        """

        # === Step 1: Compute the quaternion conjugate product ===
        # This is equivalent to q1 * conj(q0), where conj(q0) inverts the imaginary parts.
        q_conj_prod_real = current_real * previous_real + current_x * previous_x + current_y * previous_y
        q_conj_prod_x = -current_real * previous_x + previous_real * current_x
        q_conj_prod_y = -current_real * previous_y + previous_real * current_y

        # === Step 2: Compute amplitude of quaternion conjugate product ===
        q_conj_prod_amplitude = np.sqrt(
            q_conj_prod_real**2 + q_conj_prod_x**2 + q_conj_prod_y**2
        )

        # === Step 3: Prevent division by zero ===
        eps = 1e-12
        safe_amplitude = np.maximum(q_conj_prod_amplitude, eps)

        # === Step 4: Compute phase angle using the real part of the quaternion product ===
        # Quaternionic phase difference (imaginary part of log)
        phase_difference = np.arccos(np.clip(q_conj_prod_real / safe_amplitude, -1.0, 1.0))

        # === Step 5: Compute orientation unit vector (imaginary part direction) ===
        orientation_magnitude = np.sqrt(q_conj_prod_x**2 + q_conj_prod_y**2)
        safe_orientation_magnitude = np.maximum(orientation_magnitude, eps)

        cos_orientation = q_conj_prod_x / safe_orientation_magnitude
        sin_orientation = q_conj_prod_y / safe_orientation_magnitude

        # === Step 6: Decompose phase difference along orientation vector ===
        phase_diff_cos = phase_difference * cos_orientation
        phase_diff_sin = phase_difference * sin_orientation

        # === Step 7: Estimate amplitude (used to weight phase confidence) ===
        # Amplitude estimate (sqrt of quaternion amplitude)
        amplitude = np.sqrt(q_conj_prod_amplitude)

        return phase_diff_cos, phase_diff_sin, amplitude

    def PhaseShiftCoefficientRealPart(self, riesz_real: np.ndarray, riesz_x: np.ndarray, riesz_y: np.ndarray, phase_cos: np.ndarray, phase_sin: np.ndarray) -> np.ndarray:
        """
        Applies a quaternionic phase shift to a Riesz coefficient and returns the real part.

        In the context of phase-based video magnification, Riesz pyramids represent 
        local image structures as quaternion-valued coefficients. This function applies
        a quaternion rotation (phase shift) using the exponential of a purely imaginary 
        quaternion defined by (phase_cos, phase_sin), and returns only the real part 
        of the result.

        Parameters:
            riesz_real (np.ndarray):
                The real part of the Riesz coefficient at a pyramid level.
            riesz_x (np.ndarray):
                The x-component (first imaginary part) of the Riesz coefficient.
            riesz_y (np.ndarray):
                The y-component (second imaginary part) of the Riesz coefficient.
            phase_cos (np.ndarray):
                x-component of the imaginary phase shift (cosine part).
            phase_sin (np.ndarray):
                y-component of the imaginary phase shift (sine part).

        Returns:
            np.ndarray:
                The real part of the phase-shifted coefficient.
        """

        # === Step 1: Compute magnitude of the phase vector ===
        # This is used as the angle in quaternion exponential (like Euler's formula)
        phase_magnitude = np.sqrt(phase_cos**2 + phase_sin**2)

        # === Step 2: Avoid division by zero ===
        eps = 1e-12
        safe_magnitude = np.maximum(phase_magnitude, eps)

        # === Step 3: Compute quaternion exponential components ===
        # Quaternion exponentiation (Euler's formula in quaternion space)
        exp_phase_real = np.cos(phase_magnitude)
        exp_phase_x = (phase_cos / safe_magnitude) * np.sin(phase_magnitude)
        exp_phase_y = (phase_sin / safe_magnitude) * np.sin(phase_magnitude)

        # === Step 4: Multiply quaternion exp(phase) with Riesz coefficient ===
        # We compute only the real part of:
        # (a + bi + cj) * (d + ei + fj) = ad - be - cf (real part)
        result = (
            exp_phase_real * riesz_real
            - exp_phase_x * riesz_x
            - exp_phase_y * riesz_y
        )

        return result

    def ComputeLaplacianPyramid(self, image: np.ndarray, num_levels: int):
        """
        Constructs a Laplacian pyramid from a grayscale image.

        A Laplacian pyramid is a multi-resolution representation that stores
        bandpass filtered versions of the image at each level, capturing different
        spatial frequency components.

        Parameters:
            image (np.ndarray): 
                Input grayscale image as a 2D NumPy array (dtype float32 or convertible).
            num_levels (int): 
                Number of pyramid levels to construct (excluding the lowpass residual).

        Returns:
            List[np.ndarray]: 
                A list of 2D arrays representing the Laplacian pyramid levels.
                - laplacian_pyramid[0] is the highest resolution level (finest detail).
                - laplacian_pyramid[-1] is the lowpass residual (lowest resolution image).

        Notes:
            - Input image will be converted to float32 for precision.
            - Each level is one octave (factor of 2) smaller in width and height.
            - The final level is a lowpass image (not a difference).
        """
        # === Step 1: Build Gaussian pyramid ===
        # This will be used to compute differences between blurred and upsampled images
        gaussian_pyramid = [image.astype(np.float32)]
        for _ in range(num_levels):
            image = cv2.pyrDown(image)
            gaussian_pyramid.append(image.astype(np.float32))

        # === Step 2: Construct Laplacian pyramid ===
        laplacian_pyramid = []
        for i in range(num_levels):
            size = (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0])  # width, height
            expanded = cv2.pyrUp(gaussian_pyramid[i + 1], dstsize=size)
            lap = gaussian_pyramid[i] - expanded
            laplacian_pyramid.append(lap)
        
        # === Step 3: Add lowpass residual ===
        # The lowest level of the Gaussian pyramid becomes the residual
        laplacian_pyramid.append(gaussian_pyramid[-1])
        return laplacian_pyramid

    def ComputeRieszPyramid(self, grayscale_frame: np.ndarray, num_levels: int = 4):
        """
        Computes the Riesz pyramid of a 2D grayscale image frame.

        The Riesz pyramid includes:
        - A Laplacian pyramid (multi-resolution representation of the image)
        - First-order Riesz transform components (in x and y directions)
        for each pyramid level except the lowest (lowpass residual).

        Parameters:
            grayscale_frame (np.ndarray):
                2D array representing the grayscale input frame.
            num_levels (int):
                Number of pyramid levels (default is 4). Must be >= 1.

        Returns:
            Tuple containing three lists of np.ndarray:
                - laplacian_pyramid (List[np.ndarray]): 
                    Multi-resolution Laplacian pyramid including lowpass residual.
                - riesz_x (List[np.ndarray]):
                    Riesz transform in the x-direction for each level (except residual).
                - riesz_y (List[np.ndarray]):
                    Riesz transform in the y-direction for each level (except residual).

        Notes:
            - Assumes input is already normalized (e.g., float32 in range [0, 1]).
            - The number of Riesz levels is one less than the number of Laplacian levels
            because the residual is not transformed.
        """

        # Compute Laplacian pyramid
        laplacian_pyramid = self.ComputeLaplacianPyramid(grayscale_frame, num_levels)
        number_of_levels = len(laplacian_pyramid) - 1  # Exclude lowpass residual

        # Define Riesz transform approximation kernels
        kernel_x = np.array([
            [0.0,  0.0,  0.0],
            [0.5,  0.0, -0.5],
            [0.0,  0.0,  0.0]
        ])

        kernel_y = np.array([
            [ 0.0,  0.5,  0.0],
            [ 0.0,  0.0,  0.0],
            [ 0.0, -0.5,  0.0]
        ])

        riesz_x = []
        riesz_y = []

        for k in range(number_of_levels):
            riesz_x_k = convolve(laplacian_pyramid[k], kernel_x, mode='reflect')
            riesz_y_k = convolve(laplacian_pyramid[k], kernel_y, mode='reflect')
            riesz_x.append(riesz_x_k)
            riesz_y.append(riesz_y_k)

        return laplacian_pyramid, riesz_x, riesz_y

    def CollapseLaplacianPyramid(self, laplacian_pyramid):
        """
        Reconstructs an image from its Laplacian pyramid representation.

        This function collapses a Laplacian pyramid by successively upsampling the 
        lowpass residual and adding back the high-frequency details stored at each level.

        Parameters:
            laplacian_pyramid (List[np.ndarray]): 
                A list of NumPy arrays representing the Laplacian pyramid.
                - laplacian_pyramid[0] is the highest resolution level (most detailed).
                - laplacian_pyramid[-1] is the lowest resolution level (lowpass residual).

        Returns:
            np.ndarray:
                The reconstructed image at the original resolution, matching the shape
                of laplacian_pyramid[0].

        Notes:
            - Assumes each level is one octave lower resolution than the previous.
            - Uses bilinear interpolation for upsampling.
        """

        # Start from the smallest level (lowpass residual)
        current = laplacian_pyramid[-1]

        # Iterate from smallest to largest level
        for level in reversed(range(len(laplacian_pyramid) - 1)):
            # Upsample current image to next higher resolution level size
            size = (laplacian_pyramid[level].shape[1], laplacian_pyramid[level].shape[0])  # width, height
            current = cv2.resize(current, size, interpolation=cv2.INTER_LINEAR)

            # Add the Laplacian image at this level
            current = current + laplacian_pyramid[level]

        return current

