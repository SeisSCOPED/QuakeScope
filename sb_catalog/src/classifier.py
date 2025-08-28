import numpy as np
import scipy
import torch
import torch.nn as nn

import torch.nn.functional as F
import matplotlib.pyplot as plt

from typing import Any
from scipy.signal import butter, filtfilt
#import torchaudio.functional as AF
#import torchvision.transforms as transforms

from seisbench.models.base import WaveformModel


class WaveformPreprocessor:
    def __init__(self, input_fs=100, target_fs=50, lowcut=1, highcut=20, order=4, taper_alpha=0.1):
        self.input_fs = input_fs
        self.target_fs = target_fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order
        self.taper_alpha = taper_alpha

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Preprocess a single waveform: detrend, taper, filter, resample, normalize.
        Input:  (C, T) or (1, C, T)
        Output: (C, T_new)
        """
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)  # → (1, C, T)

        x = waveform.clone()

        x = self._linear_detrend(x)
        
        # saving the detrending for comparison
        #torch.save(x, "/home/ak287/PNW_Seismic_Event_Classification/deep_learning/seisbench_output/detrended_seisbench.pt")
        

        
        x = self._taper_tukey(x, alpha=self.taper_alpha)
        #torch.save(x, "/home/ak287/PNW_Seismic_Event_Classification/deep_learning/seisbench_output/tapered_seisbench.pt")
        
        
        x = self._bandpass_filter(x, fs=self.input_fs, lowcut=self.lowcut, highcut=self.highcut, order=self.order)
        #torch.save(x, "/home/ak287/PNW_Seismic_Event_Classification/deep_learning/seisbench_output/filtered_seisbench.pt")
        
        
        x = self._resample(x, self.input_fs, self.target_fs)
        #torch.save(x, "/home/ak287/PNW_Seismic_Event_Classification/deep_learning/seisbench_output/resampled_seisbench.pt")
        
        
        x = self._normalize_per_trace(x)
        #torch.save(x, "/home/ak287/PNW_Seismic_Event_Classification/deep_learning/seisbench_output/normalized_seisbench.pt")
        
        return x  
    
    

    def _linear_detrend(self, batch: torch.Tensor) -> torch.Tensor:
        time = torch.arange(batch.shape[-1], dtype=batch.dtype, device=batch.device)
        time_mean = time.mean()
        time_var = ((time - time_mean) ** 2).sum()
        slope = ((batch * (time - time_mean)).sum(dim=-1, keepdim=True)) / time_var
        intercept = batch.mean(dim=-1, keepdim=True) - slope * time_mean
        trend = slope * time + intercept
        return batch - trend

    def _taper_tukey(self, batch: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
        tukey_window = scipy.signal.windows.tukey(batch.shape[-1], alpha=alpha)
        window = torch.tensor(tukey_window, dtype=batch.dtype, device=batch.device)
        return batch * window

    def _bandpass_filter(self, batch: torch.Tensor, fs: float, lowcut: float, highcut: float, order: int) -> torch.Tensor:
        numpy_batch = batch.cpu().numpy()
        nyquist = 0.5 * fs
        b, a = scipy.signal.butter(order, [lowcut / nyquist, highcut / nyquist], btype='band')

        filtered = np.zeros_like(numpy_batch)
        for i in range(numpy_batch.shape[0]):
            for j in range(numpy_batch.shape[1]):
                filtered[i, j] = scipy.signal.filtfilt(b, a, numpy_batch[i, j])

        return torch.tensor(filtered, dtype=batch.dtype, device=batch.device)

    def _resample(self, batch: torch.Tensor, fs_in: float, fs_out: float) -> torch.Tensor:
        orig_len = batch.shape[-1]
        new_len = int(orig_len * fs_out / fs_in)
        return F.interpolate(batch, size=new_len, mode='linear', align_corners=False)

    def _normalize_per_trace(self, batch: torch.Tensor) -> torch.Tensor:
        stds = torch.std(torch.abs(batch.reshape(batch.shape[0], -1)), dim=1, keepdim=True)
        stds = stds.view(-1, 1, 1)
        return batch / (stds + 1e-10)

class QuakeXNet(WaveformModel):
    _annotate_args = WaveformModel._annotate_args.copy()
    # Set default stride in samples
    _annotate_args["stride"] = (_annotate_args["stride"][0], 2500)

    def __init__(self, 
                sampling_rate = 50,
                classes = 4, 
                output_type = "point",
                labels = ['eq','px','no','su'],
                pred_sample = 2500, 
                num_channels = 3,
                num_classes = 4,
                dropout_rate = 0.4,
                 **kwargs):
        
        citation = ("Kharita, Akash, Marine Denolle, Alexander Hutko, and J. Renate Hartog." 
                    "A comprehensive machine learning and deep learning exploration for seismic event classification in the Pacific Northwest."
                    " AGU24 (2024).")
        
        
        super().__init__(
            citation=citation,
            output_type="point",
            component_order = "ENZ",
            in_samples= 5000,
            pred_sample=pred_sample,
            labels= labels,
            sampling_rate=sampling_rate,
            **kwargs
        )
        
   # Define the layers of the CNN architecture
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=8, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=2, padding=1)
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)
        self.conv7 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
        
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
    
    
    

    def annotate_batch_pre(self, batch: torch.Tensor, argdict: dict[str, Any]) -> torch.Tensor:
        
        processor = WaveformPreprocessor(input_fs = 50, target_fs = 50, lowcut = 1, highcut = 20, order = 4, taper_alpha = 0.1)
 
        processed = processor(batch)
        
        spec = self.compute_spectrogram(processed, fs = 50, nperseg = 256, overlap = 0.5)
        #torch.save(spec, "/home/ak287/PNW_Seismic_Event_Classification/deep_learning/seisbench_output/spectrogram_seisbench.pt")
        
        
        norm_spec = self.normalize_spectrogram_minmax(spec[0])
        #torch.save(norm_spec, "/home/ak287/PNW_Seismic_Event_Classification/deep_learning/seisbench_output/normalized_spectrogram_seisbench.pt")  

        return norm_spec

    def annotate_batch_post(
        self, batch: torch.Tensor, piggyback: Any, argdict: dict[str, Any]
    ) -> torch.Tensor:
        return torch.softmax(batch, dim=-1)
    
    def classify_aggregate(self, annotations, argdict) -> list:
        window_labels = np.argmax(np.array(annotations), axis = 0)

        lb = [self.labels[i] for i in window_labels]
        t = [annotations[0].stats.starttime + i for i in annotations[0].times()]

        return [i for i in zip(lb, t) if i[0] != 'no']
    
    def compute_spectrogram(self, batch: torch.Tensor, fs: int = 50, nperseg: int = 256, overlap: float = 0.5):
        """
        Compute PSD spectrogram (B, C, T) → (B, C, F, T_spec)
        """
        B, C, N = batch.shape
        noverlap = int(nperseg * overlap)
        hop = nperseg - noverlap
        win = torch.hann_window(nperseg, periodic=True, dtype=batch.dtype, device=batch.device)

        segs = batch.unfold(-1, nperseg, hop)
        segs = segs - segs.mean(dim=-1, keepdim=True)
        segs = segs * win

        Z = torch.fft.rfft(segs, n=nperseg, dim=-1).permute(0, 1, 3, 2)
        W = win.pow(2).sum()
        psd = (Z.abs() ** 2) / (W * fs)
        if nperseg % 2 == 0:
            psd[..., 1:-1, :] *= 2.0
        else:
            psd[..., 1:, :] *= 2.0

        freqs = torch.fft.rfftfreq(nperseg, 1 / fs).to(batch.device)
        times = (torch.arange(psd.shape[-1], dtype=batch.dtype, device=batch.device) * hop + nperseg // 2) / fs

        return psd, freqs, times



    def normalize_spectrogram_minmax(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Normalize each trace spectrogram (B, C, F, T) independently to [0, 1]
        """
        B, C, F, T = spectrogram.shape
        spec_flat = spectrogram.reshape(B, C, -1)
        min_vals = spec_flat.min(dim=-1, keepdim=True)[0].view(B, C, 1, 1)
        max_vals = spec_flat.max(dim=-1, keepdim=True)[0].view(B, C, 1, 1)
        return (spectrogram - min_vals) / (max_vals - min_vals + 1e-10)

    
    def plot_waveforms(waveform, title = 'Normalized waveforms'):
        components = ['Z', 'N', 'E']
        fs = 50  # sampling rate in Hz
        time = np.arange(waveform.shape[1]) / fs  # convert sample index to seconds

        plt.figure(figsize=(12, 8))

        for i in range(3):
            plt.subplot(3, 1, i + 1)
            plt.plot(time, waveform[i], linewidth=1.2)
            plt.title(f"{components[i]} Component", fontsize=14)
            plt.ylabel("Amplitude", fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.5)
            if i == 2:
                plt.xlabel("Time (s)", fontsize=12)
            else:
                plt.xticks([])  # hide x-ticks for top plots to reduce clutter

        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # adjust layout to fit suptitle
        plt.show()


# In addition either classify or classify_aggregate