"""
Mel module
"""
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


class MelCompute(torch.nn.Module):
    "Mel compute"

    def __init__(self, device: Any, mel_filters_path: str) -> None:
        super().__init__()
        self.SAMPLE_RATE = 16000
        self.N_FFT = 400
        self.N_MELS = 80
        self.HOP_LENGTH = 160
        self.CHUNK_LENGTH = 30
        self.N_SAMPLES = self.CHUNK_LENGTH * self.SAMPLE_RATE
        self.N_FRAMES = self.exact_div()
        self.mel_filters_path = mel_filters_path
        self.mel_filter = self.__load_mel_filters(
            device, mel_filters_path=mel_filters_path
        )

    def exact_div(self) -> float:
        assert self.N_SAMPLES % self.HOP_LENGTH == 0
        return self.N_SAMPLES // self.HOP_LENGTH

    def __load_mel_filters(
        self, device: Any, mel_filters_path: str
    ) -> torch.Tensor:
        """
        load the mel filterbank matrix for projecting STFT into a
        Mel spectrogram.

        Allows decoupling librosa dependency; saved using:

            np.savez_compressed(
                "mel_filters.npz",
                mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
            )
        """
        with np.load(
            mel_filters_path,
            allow_pickle=True,
        ) as f:
            return torch.from_numpy(f[f"mel_{self.N_MELS}"]).to(device)

    def pad_or_trim(self, array: Any, axis: int = -1) -> Any:
        """
        Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
        """
        pad_widths = [(0, 0)] * array.ndim
        pad_widths[axis] = (0, self.N_FRAMES - array.shape[axis])
        array = F.pad(
            array, [pad for sizes in pad_widths[::-1] for pad in sizes]
        )

        return array

    def log_mel_spectrogram(self, audio: torch.Tensor) -> Any:
        """
        Compute the log-Mel spectrogram of

        Parameters
        ----------
        audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
            The path to audio or either a NumPy array or Tensor containing
            the audio waveform in 16 kHz

        n_mels: int
            The number of Mel-frequency filters, only 80 is supported

        Returns
        -------
        torch.Tensor, shape = (80, n_frames)
            A Tensor that contains the Mel spectrogram
        """
        window = torch.hann_window(self.N_FFT).to(audio.device)
        stft = torch.stft(
            audio,
            self.N_FFT,
            self.HOP_LENGTH,
            window=window,
            return_complex=True,
        )
        magnitudes = stft[:, :-1].abs() ** 2

        mel_spec = self.mel_filter @ magnitudes
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec

    def forward(self, audio: Any) -> Any:
        "forward"
        mel = self.log_mel_spectrogram(audio)
        content_frames = mel.shape[-1]
        return self.pad_or_trim(mel), content_frames
