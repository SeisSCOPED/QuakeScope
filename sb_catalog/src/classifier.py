from typing import Any

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, filtfilt
from seisbench.models.base import WaveformModel


class QuakeXNet(WaveformModel):
    _annotate_args = WaveformModel._annotate_args.copy()
    # Set default stride in samples
    _annotate_args["stride"] = (_annotate_args["stride"][0], 1000)

    def __init__(
        self,
        sampling_rate=50,
        classes=4,
        output_type="point",
        labels=["eq", "px", "no", "su"],
        pred_sample=19,
        num_channels=3,
        num_classes=4,
        dropout_rate=0.4,
        **kwargs
    ):

        citation = (
            "Kharita, Akash, Marine Denolle, Alexander Hutko, and J. Renate Hartog."
            "A comprehensive machine learning and deep learning exploration for seismic event classification in the Pacific Northwest."
            " AGU24 (2024)."
        )

        super().__init__(
            citation=citation,
            output_type="point",
            component_order="ENZ",
            in_samples=5000,
            pred_sample=pred_sample,
            labels=labels,
            sampling_rate=sampling_rate,
            **kwargs
        )

        # Define the layers of the CNN architecture
        self.conv1 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=8,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=8, out_channels=8, kernel_size=(3, 3), stride=2, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1, padding=1
        )
        self.conv4 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=(3, 3), stride=2, padding=1
        )
        self.conv5 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding=1
        )
        self.conv6 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1
        )
        self.conv7 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=1
        )

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Batch-normalization layers
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(8)
        self.bn3 = nn.BatchNorm2d(16)
        self.bn4 = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(32)
        self.bn6 = nn.BatchNorm2d(32)
        self.bn7 = nn.BatchNorm2d(64)

        # Calculate the input size for the fully connected layer dynamically
        self.fc_input_size = self._get_conv_output_size(num_channels, (129, 38))
        self.fc1 = nn.Linear(self.fc_input_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.fc1_bn = nn.BatchNorm1d(128)
        self.fc2_bn = nn.BatchNorm1d(num_classes)

        # Define dropout
        self.dropout = nn.Dropout(dropout_rate)

    def _get_conv_output_size(self, num_channels, input_dims):
        # Forward pass a dummy input through the conv layers to get the output size
        dummy_input = torch.randn(1, num_channels, *input_dims)
        with torch.no_grad():
            x = F.relu(self.bn1(self.conv1(dummy_input)))
            x = self.pool1(F.relu(self.bn2(self.conv2(x))))
            x = F.relu(self.bn3(self.conv3(x)))
            x = self.pool1(F.relu(self.bn4(self.conv4(x))))
            x = F.relu(self.bn5(self.conv5(x)))
            x = F.relu(self.bn6(self.conv6(x)))
            x = F.relu(self.bn7(self.conv7(x)))
            # print(f"Output shape after conv layers: {x.shape}")
        return x.numel()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # output size: (8, 129, 38)
        x = self.pool1(F.relu(self.bn2(self.conv2(x))))  # output size: (8, 64, 19)
        x = self.dropout(x)

        x = F.relu(self.bn3(self.conv3(x)))  # output size: (16, 64, 19)
        x = self.pool1(F.relu(self.bn4(self.conv4(x))))  # output size: (16, 32, 10)
        x = self.dropout(x)

        x = F.relu(self.bn5(self.conv5(x)))  # output size: (32, 32, 10)
        x = F.relu(self.bn6(self.conv6(x)))  # output size: (32, 16, 5)
        x = self.dropout(x)

        x = F.relu(self.bn7(self.conv7(x)))  # output size: (64, 16, 5)

        x = x.view(x.size(0), -1)  # Flatten before fully connected layer
        x = self.dropout(x)

        x = F.relu(self.fc1_bn(self.fc1(x)))  # classifier
        x = self.fc2_bn(self.fc2(x))  # classifier

        # Do not apply softmax here, as it will be applied in the loss function
        return x

    def linear_detrend(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply linear detrending similar to ObsPy.
        """
        # Time indices
        time = torch.arange(tensor.shape[-1], dtype=tensor.dtype, device=tensor.device)

        # Calculate linear fit coefficients using least squares
        time_mean = time.mean()
        time_variance = ((time - time_mean) ** 2).sum()
        slope = (
            (tensor * (time - time_mean)).sum(dim=-1, keepdim=True)
        ) / time_variance
        intercept = tensor.mean(dim=-1, keepdim=True) - slope * time_mean

        # Compute the trend
        trend = slope * time + intercept

        # Remove the trend from the original tensor
        return tensor - trend

    # Apply the filter using filtfilt
    def bandpass_filter(
        self, batch: torch.Tensor, fs: float, lowcut: float, highcut: float, order=4
    ) -> torch.Tensor:
        # Convert tensor to numpy array
        input_numpy = batch.numpy()  # Shape: (batch_size, num_channels, window_length)

        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype="band")

        # Apply the bandpass filter to each batch and each channel
        filtered_numpy = np.zeros_like(input_numpy)
        for i in range(input_numpy.shape[0]):  # Iterate over batch size
            for j in range(input_numpy.shape[1]):  # Iterate over channels
                filtered_numpy[i, j, :] = filtfilt(b, a, input_numpy[i, j, :])

        # Convert back to tensor
        filtered_tensor = torch.tensor(filtered_numpy)
        return filtered_tensor

    def annotate_batch_pre(
        self, batch: torch.Tensor, argdict: dict[str, Any]
    ) -> torch.Tensor:
        batch = batch.cpu()

        # Detrend each component
        batch = self.linear_detrend(batch)

        # Create a Tukey window using scipy
        tukey_window = scipy.signal.windows.tukey(batch.shape[-1], alpha=0.1)

        # Convert the Tukey window to a PyTorch tensor
        taper = torch.tensor(tukey_window, device=batch.device)

        # Apply the Tukey window to the batch
        batch = batch * taper  # Broadcasting over last axis

        # Apply bandpass filter (1-20 Hz) using torchaudio for filtfilt behavior
        batch = self.bandpass_filter(
            batch, lowcut=1, highcut=20, fs=argdict["sampling_rate"]
        )

        # Normalize each component by the standard deviation of their absolute values
        batch_abs = torch.abs(batch)
        std_abs = batch_abs.std(dim=-1, keepdim=True)
        batch = batch / (std_abs + 1e-10)  # Avoid division by zero

        # Convert the processed waveforms to spectrograms
        spec, f, t = self.extract_spectrograms(batch, fs=argdict["sampling_rate"])

        # print(spec.shape)
        return spec

    def annotate_batch_post(
        self, batch: torch.Tensor, piggyback: Any, argdict: dict[str, Any]
    ) -> torch.Tensor:
        return torch.softmax(batch, dim=-1)

    def classify_aggregate(self, annotations, argdict) -> list:
        window_labels = np.argmax(np.array(annotations), axis=0)

        lb = [self.labels[i] for i in window_labels]
        t = [annotations[0].stats.starttime + i for i in annotations[0].times()]

        return [i for i in zip(lb, t) if i[0] != "no"]

    def extract_spectrograms(self, waveforms, fs, nperseg=256, overlap=0.5):
        """
        Extract spectrograms, time segments, and frequency bins from waveforms.

        Parameters:
            waveforms: Tensor of shape (n_waveforms, n_channels, n_samples)
            fs: Sampling rate (Hz)
            nperseg: Number of FFT points
            overlap: Fractional overlap between segments

        Returns:
            spectrograms: Tensor of shape (n_waveforms, n_channels, frequencies, time_segments)
            frequencies: Array of frequency bins (Hz)
            time_segments: Array of time segment centers (seconds)
        """
        noverlap = int(nperseg * overlap)  # Calculate overlap
        hop_length = nperseg - noverlap  # Calculate hop length

        # Compute frequencies
        frequencies = torch.fft.rfftfreq(nperseg, d=1 / fs)

        # Compute time segments
        time_segments = (
            torch.arange(0, waveforms.shape[-1] - nperseg + 1, hop_length) / fs
        )

        # Example spectrogram to get dimensions
        example_spectrogram = torch.stft(
            waveforms[0, 0],
            n_fft=nperseg,
            hop_length=hop_length,
            win_length=nperseg,
            return_complex=True,
            center=False,
        )
        freq_bins, time_bins = (
            example_spectrogram.shape[-2],
            example_spectrogram.shape[-1],
        )

        # Initialize tensor for spectrograms
        spectrograms = torch.zeros(
            (waveforms.shape[0], waveforms.shape[1], freq_bins, time_bins),
            dtype=torch.complex64,
        )

        # Compute spectrograms
        for i in range(waveforms.shape[0]):  # For each waveform
            for j in range(waveforms.shape[1]):  # For each channel
                Sxx = torch.stft(
                    waveforms[i, j],
                    n_fft=nperseg,
                    hop_length=hop_length,
                    win_length=nperseg,
                    return_complex=True,
                    center=False,
                )
                spectrograms[i, j] = Sxx  # Fill the tensor

        # Convert complex spectrogram to magnitude
        spectrograms = torch.abs(spectrograms)

        return spectrograms, frequencies, time_segments
