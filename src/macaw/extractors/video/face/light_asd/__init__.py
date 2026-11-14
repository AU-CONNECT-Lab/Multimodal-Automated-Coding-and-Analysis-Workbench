"""Light-ASD, a model that tells from a face and the sound whether that face is speaking.

The network is the published one, laid out under the same names so its weights load; see LICENSE beside this file.
Nothing here imports the rest of the package, so a check can load this file on its own."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# the AVA-ActiveSpeaker weights, at a fixed commit so a run always reads the same ones
WEIGHTS_URL = "https://raw.githubusercontent.com/Junhua-Liao/Light-ASD/ed38c232de5efe0261dbd68627c0ade7cdfe14eb/weight/pretrain_AVA_CVPR.model"
WEIGHTS_FILE = "light_asd_pretrain_AVA_CVPR.model"

FPS = 25
SAMPLE_RATE = 16000
CROP_SCALE = 0.40
SMOOTH_FRAMES = 13
# frames without a face that a track bridges, and the shortest track scored, as in the model's own demo
MAX_GAP_FRAMES = 10
MIN_TRACK_FRAMES = 10
# window lengths (s) whose scores are averaged, as in the model's own demo
DURATIONS = (1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6)
# a multiple of every window length, so scoring a track in pieces gives what scoring it whole gives
CHUNK_FRAMES = 60 * FPS
# windows of one length sent to the model at once
_BATCH = 8
# where the detector's box sits relative to the 68 face points, fitted on faces that have both
BOX_SIZE = 1.283
BOX_RAISE = 0.315


def face_boxes(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(centre x, centre y, half size) of the box a face detector draws, from each row of 68 points."""
    left, right = np.nanmin(xs, axis=1), np.nanmax(xs, axis=1)
    top, bottom = np.nanmin(ys, axis=1), np.nanmax(ys, axis=1)
    span = np.maximum(right - left, bottom - top) / 2
    return (left + right) / 2, (top + bottom) / 2 - BOX_RAISE * span, BOX_SIZE * span


@dataclass
class Track:
    """One unbroken run of a face at 25 moments a second: the recording time, the file frame shown, and the crop."""

    times: np.ndarray
    frames: np.ndarray
    x: np.ndarray
    y: np.ndarray
    s: np.ndarray


def plan_tracks(t: np.ndarray, x: np.ndarray, y: np.ndarray, s: np.ndarray, *, fps: float, offset_s: float) -> list[Track]:
    """A face seen at times t (a file of fps frames placed at offset_s) as runs at 25 a second: short gaps are bridged
    and the crop smoothed as the model's demo does, and each moment shows the file frame nearest it."""
    from scipy.ndimage import median_filter

    order = np.argsort(t)
    t, x, y, s = (np.asarray(values, dtype=float)[order] for values in (t, x, y, s))
    if t.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(t) > (MAX_GAP_FRAMES + 1) / FPS) + 1
    tracks = []
    for part in np.split(np.arange(t.size), breaks):
        times = t[part[0]] + np.arange(int(math.floor((t[part[-1]] - t[part[0]]) * FPS + 1e-6)) + 1) / FPS
        if times.size <= MIN_TRACK_FRAMES:
            continue
        # held at the ends rather than padded with zeros, which would pull the first and last crops towards the corner
        smooth = [median_filter(np.interp(times, t[part], values[part]), size=SMOOTH_FRAMES, mode="nearest") for values in (x, y, s)]
        frames = np.round((times - offset_s) * fps).astype(int)
        tracks.append(Track(times, frames, *smooth))
    return tracks


def crop(image: np.ndarray, x: float, y: float, s: float) -> np.ndarray:
    """The demo's crop of a face, on the picture padded with grey 110, as a 112 px grey square."""
    import cv2

    pad = int(s * (1 + 2 * CROP_SCALE))
    top, bottom = int(y + pad - s), int(y + pad + s * (1 + 2 * CROP_SCALE))
    left, right = int(x + pad - s * (1 + CROP_SCALE)), int(x + pad + s * (1 + CROP_SCALE))
    out = np.full((max(bottom - top, 1), max(right - left, 1), 3), 110, dtype=np.uint8)
    h, w = image.shape[:2]
    y0, y1 = max(top - pad, 0), min(bottom - pad, h)
    x0, x1 = max(left - pad, 0), min(right - pad, w)
    if y1 > y0 and x1 > x0:
        out[y0 - (top - pad): y1 - (top - pad), x0 - (left - pad): x1 - (left - pad)] = image[y0:y1, x0:x1]
    face = cv2.cvtColor(cv2.resize(out, (224, 224)), cv2.COLOR_BGR2GRAY)
    return face[56:168, 56:168]


def sound_for(samples: np.ndarray, start_s: float, n_frames: int) -> np.ndarray:
    """The 16-bit samples under n_frames moments from start_s (seconds into the track), and one frame more, so the last
    moment keeps its sound; silence where the track has none."""
    first = int(round(start_s * SAMPLE_RATE))
    length = (n_frames + 1) * SAMPLE_RATE // FPS
    out = np.zeros(length, dtype=np.int16)
    lo, hi = max(first, 0), min(first + length, samples.size)
    if hi > lo:
        out[lo - first: hi - first] = samples[lo:hi]
    return out


def score(net, sound: np.ndarray, faces: np.ndarray, device: str) -> np.ndarray:
    """Per moment, how much the face is the one speaking (above 0 reads as speaking): the mean over windows of
    several lengths, as the model's demo scores a track. No longer than faces, or the sound under them."""
    import python_speech_features
    import torch

    mfcc = python_speech_features.mfcc(sound, SAMPLE_RATE, numcep=13, winlen=0.025, winstep=0.010)
    length = min((mfcc.shape[0] - mfcc.shape[0] % 4) / 100, faces.shape[0] / FPS)
    mfcc, faces = mfcc[: int(round(length * 100))], faces[: int(round(length * FPS))]
    n_faces = faces.shape[0]
    if n_faces == 0:
        return np.zeros(0)

    def ask(a: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Windows of one length at once; each is read on its own, so batching them changes nothing."""
        a = torch.as_tensor(a, dtype=torch.float32, device=device)
        v = torch.as_tensor(v, dtype=torch.float32, device=device)
        out = net.model.forward_audio_visual_backend(net.model.forward_audio_frontend(a), net.model.forward_visual_frontend(v))
        return net.lossAV.FC(out.squeeze(1))[:, 1].cpu().numpy()

    every = []
    with torch.no_grad():
        for duration in DURATIONS:
            whole = n_faces // (duration * FPS)
            found = []
            for first in range(0, whole, _BATCH):
                last = min(first + _BATCH, whole)
                a = mfcc[first * duration * 100: last * duration * 100].reshape(last - first, duration * 100, -1)
                v = faces[first * duration * FPS: last * duration * FPS].reshape(last - first, duration * FPS, *faces.shape[1:])
                found.append(ask(a, v))
            if whole * duration * FPS < n_faces:
                found.append(ask(mfcc[whole * duration * 100:][None], faces[whole * duration * FPS:][None]))
            every.append(np.concatenate(found))
    return np.mean(np.array(every), axis=0)


def network(state: dict, device: str):
    """The network with these weights, on device; every one of its parameters must be among them."""
    net = _build()
    own = net.state_dict()
    loaded = 0
    for name, value in state.items():
        key = name if name in own else name.replace("module.", "")
        if key in own and own[key].shape == value.shape:
            own[key].copy_(value)
            loaded += 1
    if loaded != len(own):
        raise ValueError(f"the Light-ASD weights fill {loaded} of the network's {len(own)} parameters")
    return net.to(device).eval()


def _build():
    """The published layout: a visual and an audio encoder, two GRUs, and the linear heads its weights carry."""
    import torch
    from torch import nn

    class AudioBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.relu = nn.ReLU()
            self.m_3 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), padding=(1, 0), bias=False)
            self.bn_m_3 = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)
            self.t_3 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1), bias=False)
            self.bn_t_3 = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)
            self.m_5 = nn.Conv2d(in_channels, out_channels, kernel_size=(5, 1), padding=(2, 0), bias=False)
            self.bn_m_5 = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)
            self.t_5 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 5), padding=(0, 2), bias=False)
            self.bn_t_5 = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)
            self.last = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1), padding=(0, 0), bias=False)
            self.bn_last = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)

        def forward(self, x):
            x_3 = self.relu(self.bn_t_3(self.t_3(self.relu(self.bn_m_3(self.m_3(x))))))
            x_5 = self.relu(self.bn_t_5(self.t_5(self.relu(self.bn_m_5(self.m_5(x))))))
            return self.relu(self.bn_last(self.last(x_3 + x_5)))

    class VisualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, is_down=False):
            super().__init__()
            self.relu = nn.ReLU()
            stride = (1, 2, 2) if is_down else (1, 1, 1)
            self.s_3 = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3), stride=stride, padding=(0, 1, 1), bias=False)
            self.bn_s_3 = nn.BatchNorm3d(out_channels, momentum=0.01, eps=0.001)
            self.t_3 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False)
            self.bn_t_3 = nn.BatchNorm3d(out_channels, momentum=0.01, eps=0.001)
            self.s_5 = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 5, 5), stride=stride, padding=(0, 2, 2), bias=False)
            self.bn_s_5 = nn.BatchNorm3d(out_channels, momentum=0.01, eps=0.001)
            self.t_5 = nn.Conv3d(out_channels, out_channels, kernel_size=(5, 1, 1), padding=(2, 0, 0), bias=False)
            self.bn_t_5 = nn.BatchNorm3d(out_channels, momentum=0.01, eps=0.001)
            self.last = nn.Conv3d(out_channels, out_channels, kernel_size=(1, 1, 1), padding=(0, 0, 0), bias=False)
            self.bn_last = nn.BatchNorm3d(out_channels, momentum=0.01, eps=0.001)

        def forward(self, x):
            x_3 = self.relu(self.bn_t_3(self.t_3(self.relu(self.bn_s_3(self.s_3(x))))))
            x_5 = self.relu(self.bn_t_5(self.t_5(self.relu(self.bn_s_5(self.s_5(x))))))
            return self.relu(self.bn_last(self.last(x_3 + x_5)))

    class VisualEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.block1 = VisualBlock(1, 32, is_down=True)
            self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
            self.block2 = VisualBlock(32, 64)
            self.pool2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
            self.block3 = VisualBlock(64, 128)
            self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

        def forward(self, x):
            x = self.block3(self.pool2(self.block2(self.pool1(self.block1(x))))).transpose(1, 2)
            b, t, c, w, h = x.shape
            return self.maxpool(x.reshape(b * t, c, w, h)).view(b, t, c)

    class AudioEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.block1 = AudioBlock(1, 32)
            self.pool1 = nn.MaxPool3d(kernel_size=(1, 1, 3), stride=(1, 1, 2), padding=(0, 0, 1))
            self.block2 = AudioBlock(32, 64)
            self.pool2 = nn.MaxPool3d(kernel_size=(1, 1, 3), stride=(1, 1, 2), padding=(0, 0, 1))
            self.block3 = AudioBlock(64, 128)

        def forward(self, x):
            x = self.block3(self.pool2(self.block2(self.pool1(self.block1(x)))))
            return torch.mean(x, dim=2, keepdim=True).squeeze(2).transpose(1, 2)

    class BGRU(nn.Module):
        def __init__(self, channel):
            super().__init__()
            self.gru_forward = nn.GRU(input_size=channel, hidden_size=channel, num_layers=1, bidirectional=False, bias=True, batch_first=True)
            self.gru_backward = nn.GRU(input_size=channel, hidden_size=channel, num_layers=1, bidirectional=False, bias=True, batch_first=True)
            self.gelu = nn.GELU()

        def forward(self, x):
            x = self.gelu(self.gru_forward(x)[0])
            x = torch.flip(self.gru_backward(torch.flip(x, dims=[1]))[0], dims=[1])
            return self.gelu(x)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.visualEncoder = VisualEncoder()
            self.audioEncoder = AudioEncoder()
            self.GRU = BGRU(128)

        def forward_visual_frontend(self, x):
            b, t, w, h = x.shape
            return self.visualEncoder(((x.view(b, 1, t, w, h) / 255) - 0.4161) / 0.1688)

        def forward_audio_frontend(self, x):
            return self.audioEncoder(x.unsqueeze(1).transpose(2, 3))

        def forward_audio_visual_backend(self, x1, x2):
            return torch.reshape(self.GRU(x1 + x2), (-1, 128))

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.FC = nn.Linear(128, 2)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.model, self.lossAV, self.lossV = Model(), Head(), Head()

    return Net()
