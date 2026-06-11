"""
SPAD Dataset Generation Pipeline

Processes video frames through:
1. RIFE 16x interpolation (1000 FPS -> 16000 FPS) (configurable)
2. Linear 8x interpolation (16000 FPS -> 128000 FPS) (configurable)
3. Color space conversion (sRGB -> linear RGB -> grayscale)
4. Flux calculation with two versions (PPP=1 and PPP=0.1)
5. Poisson sampling to generate binary frames
6. Save flux and binary frames
"""

import os
import sys
import yaml
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Optional, Callable
from functools import wraps
import zarr

import ihpp_data
import utils

DEFAULT_BASE_DIR = Path("data/phantom-synth")
MIN_CHUNK_SIZE = 50  # Minimum chunk size for GPU processing
# Add RIFE directory to path for model import
RIFE_DIR: Path = Path("ECCV2022-RIFE")
sys.path.insert(0, str(RIFE_DIR.absolute()))

RIFE_INTERP_EXP_DEFAULT = 4  # 2^4 = 16x interpolation (1000 FPS -> 16000 FPS)
LINEAR_INTERP_DEFAULT = 8  # 8x linear interpolation (16000 FPS -> 128000 FPS)


def load_video_frames(video_dir: Path) -> np.ndarray:
    """Load all PNG frames from a video directory, sorted by frame number."""
    frame_files = sorted(video_dir.glob("frame_*.png"), key=lambda x: int(x.stem.split('_')[1]))
    frames = []
    for frame_file in frame_files:
        frame = cv2.imread(str(frame_file))
        if frame is None:
            print(f"Warning: Could not load {frame_file}")
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    if len(frames) == 0:
        raise ValueError(f"No frames loaded from {video_dir}")

    return np.array(frames)  # Shape: [T, H, W, 3]


def pad_image(img: torch.Tensor, padding: Tuple[int, int, int, int]) -> torch.Tensor:
    """Pad image for RIFE processing."""
    return F.pad(img, padding)


def load_rife_model(model_dir: str = "train_log"):
    """Load RIFE model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    # Try loading different RIFE model versions
    model = None
    try:
        try:
            try:
                from model.RIFE_HDv2 import Model
                model = Model()
                model.load_model(model_dir, -1)
                print("Loaded RIFE HDv2 model.")
            except:
                from train_log.RIFE_HDv3 import Model
                model = Model()
                model.load_model(model_dir, -1)
                print("Loaded RIFE HDv3 model.")
        except:
            from model.RIFE_HD import Model
            model = Model()
            model.load_model(model_dir, -1)
            print("Loaded RIFE HD model.")
    except:
        from model.RIFE import Model
        model = Model()
        model.load_model(model_dir, -1)
        print("Loaded RIFE ArXiv model.")

    model.eval()
    model.device()
    return model, device


def rife_interpolate(model, device, frames: np.ndarray, exp: int = 4, scale: float = 1.0) -> np.ndarray:
    """
    Interpolate frames using RIFE.

    Args:
        model: RIFE model
        device: torch device
        frames: Input frames [T, H, W, 3] in range [0, 255]
        exp: Interpolation exponent (2^exp frames output per input frame pair)
        scale: Scale factor for processing

    Returns:
        Interpolated frames [T_out, H, W, 3]
    """
    T, H, W, C = frames.shape
    output_frames = []

    # Prepare padding
    tmp = max(32, int(32 / scale))
    ph = ((H - 1) // tmp + 1) * tmp
    pw = ((W - 1) // tmp + 1) * tmp
    padding = (0, pw - W, 0, ph - H)

    # Recursive interpolation function
    def make_inference(I0_t, I1_t, n):
        if n == 0:
            return []
        middle = model.inference(I0_t, I1_t, scale)
        if n == 1:
            return [middle]
        first_half = make_inference(I0_t, middle, n=n//2)
        second_half = make_inference(middle, I1_t, n=n//2)
        if n % 2:
            return [*first_half, middle, *second_half]
        else:
            return [*first_half, *second_half]

    print(f"RIFE interpolation: {T} input frames -> ~{(T-1) * (2**exp) + 1} output frames")

    # Process first frame
    I0_rgb = frames[0].astype(np.float32) / 255.0
    I0 = torch.from_numpy(np.transpose(I0_rgb, (2, 0, 1))).to(device).unsqueeze(0).float()
    I0 = pad_image(I0, padding)
    I0_np = (I0[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)[:H, :W]
    output_frames.append(I0_np)

    # Process frame pairs
    for i in tqdm(range(T - 1), desc="RIFE interpolation"):
        I0_rgb = frames[i].astype(np.float32) / 255.0
        I1_rgb = frames[i + 1].astype(np.float32) / 255.0

        # Convert to torch tensors and pad
        I0 = torch.from_numpy(np.transpose(I0_rgb, (2, 0, 1))).to(device).unsqueeze(0).float()
        I1 = torch.from_numpy(np.transpose(I1_rgb, (2, 0, 1))).to(device).unsqueeze(0).float()
        I0 = pad_image(I0, padding)
        I1 = pad_image(I1, padding)

        # Interpolate recursively to get 2^exp - 1 intermediate frames
        intermediate = make_inference(I0, I1, 2**exp - 1)

        # Convert intermediate frames to numpy and add to output
        for mid in intermediate:
            mid_np = (mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)[:H, :W]
            output_frames.append(mid_np)

        # The last frame of the pair will be added in the next iteration or at the end
        # original code only adds the very first frame and intermediate frames, but not the last frame of the last pair
        # ADD THIS: append I1 at the end of each pair
        I1_np = (I1[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)[:H, :W]
        output_frames.append(I1_np)

    # # Add final frame
    # I1_rgb = frames[-1].astype(np.float32) / 255.0
    # I1 = torch.from_numpy(np.transpose(I1_rgb, (2, 0, 1))).to(device).unsqueeze(0).float()
    # I1 = pad_image(I1, padding)
    # I1_np = (I1[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)[:H, :W]
    # output_frames.append(I1_np)

    return np.array(output_frames)  # [T_rife, H, W, 3]


def linear_temporal_interpolation(frames: np.ndarray, factor: int = 8) -> np.ndarray:
    """
    Linearly interpolate frames temporally on CPU (fast enough, avoids GPU memory issues).

    Args:
        frames: Input frames [T, H, W, 3] as numpy array
        factor: Interpolation factor (8x means generate 8 frames between each pair)

    Returns:
        Interpolated frames [T_out, H, W, 3] as numpy array
    """
    T, H, W, C = frames.shape

    # Total output: 1 + (T-1) * factor
    total_output = 1 + (T - 1) * factor
    print(f"Linear interpolation: {T} input frames -> {total_output} output frames")

    # Pre-allocate output array to avoid shape issues
    output_frames = np.zeros((total_output, H, W, C), dtype=np.uint8)

    # Add first frame
    output_frames[0] = frames[0].astype(np.uint8)

    # Fill in interpolated frames
    output_idx = 1
    for i in tqdm(range(T - 1), desc="Linear interpolation"):
        I0 = frames[i].astype(np.float32)
        I1 = frames[i + 1].astype(np.float32)

        # Generate 'factor' evenly-spaced frames between I0 and I1
        for j in range(1, factor + 1):
            alpha = j / factor
            interp_frame = (1 - alpha) * I0 + alpha * I1
            output_frames[output_idx] = interp_frame.astype(np.uint8)
            output_idx += 1

    return output_frames  # [T_linear, H, W, 3]


def linear_temporal_interpolation_torch(frames: torch.Tensor, factor: int = 8, device: torch.device = None) -> torch.Tensor:
    """
    Linearly interpolate frames temporally on GPU.

    Args:
        frames: Input frames [T, H, W, 3] or [T, 3, H, W] as torch.Tensor
        factor: Interpolation factor (8x means generate 8 frames between each pair)
        device: GPU device

    Returns:
        Interpolated frames [T_out, H, W, 3] or [T_out, 3, H, W]
    """
    if device is None:
        device = frames.device

    T = frames.shape[0]

    # Check if channels-last or channels-first
    if frames.shape[1] == 3:  # [T, 3, H, W]
        channels_first = True
        _, C, H, W = frames.shape
    else:  # [T, H, W, 3]
        channels_first = False
        _, H, W, C = frames.shape
        # Convert to channels-first for easier interpolation
        frames = frames.permute(0, 3, 1, 2)  # [T, H, W, 3] -> [T, 3, H, W]

    # Total output: 1 + (T-1) * factor
    total_output = 1 + (T - 1) * factor
    print(f"Linear interpolation: {T} input frames -> {total_output} output frames")

    # Create alpha values for interpolation: [1/factor, 2/factor, ..., factor/factor]
    alphas = torch.linspace(1/factor, 1.0, factor, device=device).view(-1, 1, 1, 1)  # [factor, 1, 1, 1]

    output_frames = [frames[0:1]]  # First frame [1, 3, H, W]

    # Process in batches to save memory
    # Use larger batch size for GPU efficiency, but cap it to avoid memory issues
    # For 512x256 frames: each frame ~1.5MB, so we can fit many in 40GB
    batch_size = min(64, T - 1)  # Process 64 intervals at a time

    for batch_start in tqdm(range(0, T - 1, batch_size), desc="Linear interpolation"):
        batch_end = min(batch_start + batch_size, T - 1)
        batch_indices = torch.arange(batch_start, batch_end, device=device, dtype=torch.long)

        I0_batch = frames[batch_indices]  # [batch_size, 3, H, W]
        I1_batch = frames[batch_indices + 1]  # [batch_size, 3, H, W]

        # Expand for interpolation: [batch_size, 1, 3, H, W]
        I0_expanded = I0_batch.unsqueeze(1)  # [batch_size, 1, 3, H, W]
        I1_expanded = I1_batch.unsqueeze(1)  # [batch_size, 1, 3, H, W]

        # Broadcast alphas: [1, factor, 1, 1, 1]
        alphas_expanded = alphas.view(1, factor, 1, 1, 1)

        # Interpolate: [batch_size, factor, 3, H, W]
        interp_batch = (1 - alphas_expanded) * I0_expanded + alphas_expanded * I1_expanded
        # Reshape to [batch_size * factor, 3, H, W]
        interp_batch = interp_batch.view(-1, C, H, W)

        # Append all frames from this batch at once
        output_frames.append(interp_batch)

    # Concatenate all frames
    output = torch.cat(output_frames, dim=0)  # [T_out, 3, H, W]

    if not channels_first:
        # Convert back to channels-last
        output = output.permute(0, 2, 3, 1)  # [T_out, 3, H, W] -> [T_out, H, W, 3]

    return output


def calculate_flux_torch(grayscale: torch.Tensor, target_ppp: float = 1.0, d: float = 7.74e-4) -> Tuple[torch.Tensor, float]:
    """
    Calculate flux N(x,t) = a*I(x,t) + d on GPU

    Args:
        grayscale: Grayscale frames [T, H, W] in range [0, 1] as torch.Tensor
        target_ppp: Target average photons per pixel
        d: Spurious detections per frame (default 7.74×10^-4)

    Returns:
        flux: Flux array [T, H, W] as torch.Tensor
        a: Scaling factor used
    """
    # Calculate scaling factor a so that average a*I(x,t) = target_ppp
    I_mean = grayscale.mean().item()
    if I_mean > 0:
        a = target_ppp / I_mean
    else:
        a = 1.0

    # Calculate flux: N = a*I + d
    flux = a * grayscale + d

    flux_mean = flux.mean().item()
    print(f"Flux calculation: target_ppp={target_ppp}, a={a:.6e}, I_mean={I_mean:.6f}, flux_mean={flux_mean:.6f}")

    return flux, a


def gpu_chunked_processing(chunk_size_init: int = 1000, min_chunk_size: int = 100, reduction_factor: float = 0.5, cpu_fallback: Optional[Callable] = None):
    """
    Decorator for GPU processing with automatic OOM recovery by reducing chunk size.
    Falls back to CPU if OOM occurs at minimum chunk size (if cpu_fallback provided).

    The decorated function should accept 'chunk_size' as a keyword argument and process
    data in chunks, returning a concatenated result.

    Args:
        chunk_size_init: Initial chunk size to try
        min_chunk_size: Minimum chunk size before giving up or falling back to CPU
        reduction_factor: Factor to reduce chunk size by on OOM (0.5 = halve it)
        cpu_fallback: Optional CPU fallback function with same signature (minus device param)
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, chunk_size: Optional[int] = None, **kwargs):
            if chunk_size is None:
                chunk_size = chunk_size_init

            max_retries = 10
            last_error = None

            for attempt in range(max_retries):
                try:
                    return func(*args, chunk_size=chunk_size, **kwargs)
                except torch.cuda.OutOfMemoryError as e:
                    last_error = e
                    if chunk_size <= min_chunk_size:
                        # Try CPU fallback if available
                        if cpu_fallback is not None:
                            print(f"WARNING: GPU OOM even at minimum chunk_size={chunk_size}. Falling back to CPU...")
                            torch.cuda.empty_cache()
                            # Convert args: torch tensors -> numpy, remove device from kwargs
                            args_cpu = []
                            for arg in args:
                                if isinstance(arg, torch.Tensor):
                                    args_cpu.append(arg.cpu().numpy())
                                else:
                                    args_cpu.append(arg)
                            kwargs_cpu = {k: (v.cpu().numpy() if isinstance(v, torch.Tensor) else v)
                                         for k, v in kwargs.items() if k != 'device'}
                            return cpu_fallback(*args_cpu, chunk_size=chunk_size_init, **kwargs_cpu)
                        else:
                            print(f"ERROR: Chunk size reduced to {chunk_size} (minimum {min_chunk_size}), still OOM. No CPU fallback available.")
                            raise e

                    # Clear cache and reduce chunk size
                    torch.cuda.empty_cache()
                    new_chunk_size = max(int(chunk_size * reduction_factor), min_chunk_size)
                    print(f"WARNING: GPU OOM at chunk_size={chunk_size}. Reducing to {new_chunk_size} and retrying...")
                    chunk_size = new_chunk_size

            # If we exhausted retries, raise the last error
            raise last_error

        return wrapper
    return decorator


def color_convert_cpu(frames: np.ndarray, chunk_size: int = 5000, linearize=True) -> np.ndarray:
    """
    CPU fallback for color conversion.
    Convert frames from sRGB to linear RGB to grayscale on CPU.

    Args:
        frames: Input frames [T, H, W, 3] as numpy array (uint8)
        chunk_size: Number of frames to process at once (for memory efficiency)

    Returns:
        Grayscale frames [T, H, W] as numpy array (float32, range [0, 1])
    """
    T, H, W, C = frames.shape
    grayscale_chunks = []

    for start_idx in tqdm(range(0, T, chunk_size), desc="Color conversion (CPU)"):
        end_idx = min(start_idx + chunk_size, T)
        frames_chunk = frames[start_idx:end_idx].astype(np.float32) / 255.0  # [chunk, H, W, 3] in [0, 1]

        # sRGB to linear RGB (simple gamma approximation)
        if linearize:
            linear_rgb = frames_chunk ** 2.2
        else:
            linear_rgb = frames_chunk

        # RGB to grayscale
        grayscale_chunk = (0.2989 * linear_rgb[:, :, :, 0] +
                          0.5870 * linear_rgb[:, :, :, 1] +
                          0.1140 * linear_rgb[:, :, :, 2])  # [chunk, H, W]

        grayscale_chunks.append(grayscale_chunk)

    grayscale = np.concatenate(grayscale_chunks, axis=0)  # [T, H, W]
    return grayscale


@gpu_chunked_processing(chunk_size_init=1000, min_chunk_size=MIN_CHUNK_SIZE, reduction_factor=0.5, cpu_fallback=color_convert_cpu)
def color_convert_chunked_torch(
    frames: np.ndarray, device: torch.device, chunk_size: int = 5000, linearize=True
) -> torch.Tensor:
    """
    Convert frames from sRGB to linear RGB to grayscale on GPU in chunks.

    Args:
        frames: Input frames [T, H, W, 3] as numpy array (uint8)
        device: GPU device
        chunk_size: Number of frames to process at once

    Returns:
        Grayscale frames [T, H, W] as torch.Tensor (float32, range [0, 1])
    """
    T, H, W, C = frames.shape
    # grayscale_chunks = []
    grayscale_cpu = torch.zeros((frames.shape[0], frames.shape[1], frames.shape[2]), dtype=torch.float32)

    for start_idx in tqdm(range(0, T, chunk_size), desc="Color conversion (GPU)"):
        end_idx = min(start_idx + chunk_size, T)
        frames_chunk = frames[start_idx:end_idx].astype(np.float32) / 255.0  # [chunk, H, W, 3] in [0, 1]

        # Move to GPU
        frames_torch = torch.from_numpy(frames_chunk).to(device)  # [chunk, H, W, 3]

        # sRGB to linear RGB (simple gamma approximation)
        if linearize:
            linear_rgb = frames_torch ** 2.2
        else:
            linear_rgb = frames_torch

        # RGB to grayscale
        grayscale_chunk = (0.2989 * linear_rgb[:, :, :, 0] +
                          0.5870 * linear_rgb[:, :, :, 1] +
                          0.1140 * linear_rgb[:, :, :, 2])  # [chunk, H, W]

        # Move chunk to CPU immediately to save GPU memory
        # grayscale_chunks.append(grayscale_chunk.cpu())
        grayscale_cpu[start_idx:end_idx] = grayscale_chunk.cpu()

        # Clear intermediate tensors and free GPU memory
        del frames_torch, linear_rgb, grayscale_chunk
        torch.cuda.empty_cache()

    # Concatenate on CPU
    # grayscale_cpu = torch.cat(grayscale_chunks, dim=0)  # [T, H, W] on CPU
    # Move final result back to GPU
    # why would you move it back to the GPU if chunking was neccessary in the first place?
    # just leave it on the cpu bro, who programmed this
    grayscale = grayscale_cpu
    # del grayscale_chunks
    torch.cuda.empty_cache()
    return grayscale


def calculate_flux_cpu(grayscale: np.ndarray, target_ppp: float = 1.0,
                       d: float = 7.74e-4, chunk_size: int = 2000) -> Tuple[np.ndarray, float]:
    """
    CPU fallback for flux calculation.
    Calculate flux N(x,t) = a*I(x,t) + d on CPU in chunks.

    Args:
        grayscale: Grayscale frames [T, H, W] in range [0, 1] as numpy array
        target_ppp: Target average photons per pixel
        d: Spurious detections per frame
        chunk_size: Number of frames to process at once

    Returns:
        flux: Flux array [T, H, W] as numpy array
        a: Scaling factor used
    """
    # Calculate scaling factor from full dataset
    I_mean = np.mean(grayscale)
    if I_mean > 0:
        a = target_ppp / I_mean
    else:
        a = 1.0

    # Process in chunks
    T = grayscale.shape[0]
    flux_chunks = []

    for start_idx in tqdm(range(0, T, chunk_size), desc="Flux calculation (CPU)"):
        end_idx = min(start_idx + chunk_size, T)
        chunk = grayscale[start_idx:end_idx]
        flux_chunk = a * chunk + d
        flux_chunks.append(flux_chunk)

    flux = np.concatenate(flux_chunks, axis=0)
    flux_mean = np.mean(flux)
    print(f"Flux calculation (CPU): target_ppp={target_ppp}, a={a:.6e}, I_mean={I_mean:.6f}, flux_mean={flux_mean:.6f}")

    return flux, a


@gpu_chunked_processing(chunk_size_init=2000, min_chunk_size=MIN_CHUNK_SIZE, reduction_factor=0.5, cpu_fallback=calculate_flux_cpu)
def calculate_flux_chunked_torch(
    grayscale: torch.Tensor, target_ppp: float = 1.0,
    d: float = 7.74e-4, chunk_size: int = 2000, device: torch.device="cpu"
) -> Tuple[torch.Tensor, float]:
    """
    Calculate flux N(x,t) = a*I(x,t) + d on GPU in chunks to avoid OOM.

    Args:
        grayscale: Grayscale frames [T, H, W] in range [0, 1] as torch.Tensor
        target_ppp: Target average photons per pixel
        d: Spurious detections per frame
        chunk_size: Number of frames to process at once

    Returns:
        flux: Flux array [T, H, W] as torch.Tensor
        a: Scaling factor used
    """
    # Calculate scaling factor from full dataset (use chunked mean to avoid OOM)
    # Process in chunks to calculate mean
    T = grayscale.shape[0]
    sum_val = 0.0
    count = 0
    # grayscale should be on CPU. why would you have it on GPU in the first place while chunking?

    # Calculate mean in chunks to avoid loading full tensor
    for start_idx in tqdm(range(0, T, chunk_size), desc="Mean flux calculation (GPU)"):
        end_idx = min(start_idx + chunk_size, T)
        chunk = grayscale[start_idx:end_idx].to(device)
        sum_val += chunk.sum().item()
        count += chunk.numel()

    I_mean = sum_val / count if count > 0 else 0.0
    if I_mean > 0:
        a = target_ppp / I_mean
    else:
        a = 1.0

    # Process in chunks
    # flux_chunks = []
    flux = torch.zeros_like(grayscale, dtype=torch.float32, device="cpu")

    for start_idx in tqdm(range(0, T, chunk_size), desc="Flux calculation (GPU)"):
        end_idx = min(start_idx + chunk_size, T)
        chunk = grayscale[start_idx:end_idx].to(device)
        flux_chunk = a * chunk + d
        # flux_chunks.append(flux_chunk.to("cpu"))
        flux[start_idx:end_idx] = flux_chunk.to("cpu")
        # Clear chunk from GPU immediately
        del chunk

    # flux = torch.cat(flux_chunks, dim=0)

    # Calculate flux mean in chunks
    flux_sum = flux.sum().item()
    flux_mean = flux_sum / count if count > 0 else 0.0

    print(f"Flux calculation: target_ppp={target_ppp}, a={a:.6e}, I_mean={I_mean:.6f}, flux_mean={flux_mean:.6f}")

    # Clear chunks from GPU
    # del flux_chunks
    torch.cuda.empty_cache()

    return flux, a


def poisson_sample_binary_cpu(flux: np.ndarray, chunk_size: int = 2000) -> np.ndarray:
    """
    CPU fallback for Poisson binary sampling.
    Generate binary frames using Poisson sampling on CPU in chunks.
    Pr{Φ(x,t)=1} = 1 - e^(-N(x,t))

    Args:
        flux: Flux array [T, H, W] as numpy array
        chunk_size: Number of frames to process at once

    Returns:
        Binary frames [T, H, W] as bool numpy array
    """
    T = flux.shape[0]
    binary_chunks = []

    for start_idx in tqdm(range(0, T, chunk_size), desc="Poisson sampling (CPU)"):
        end_idx = min(start_idx + chunk_size, T)
        flux_chunk = flux[start_idx:end_idx]
        prob = 1.0 - np.exp(-flux_chunk)
        binary_chunk = np.random.random(flux_chunk.shape) < prob
        binary_chunks.append(binary_chunk)

    binary = np.concatenate(binary_chunks, axis=0)
    return binary


@gpu_chunked_processing(chunk_size_init=2000, min_chunk_size=MIN_CHUNK_SIZE, reduction_factor=0.5, cpu_fallback=poisson_sample_binary_cpu)
def poisson_sample_binary_chunked_torch(
    flux: torch.Tensor, chunk_size: int = 2000, device: torch.device="cpu"
) -> torch.Tensor:
    """
    Generate binary frames using Poisson sampling on GPU in chunks to avoid OOM.
    Pr{Φ(x,t)>=1} = 1 - e^(-N(x,t))

    Args:
        flux: Flux array [T, H, W] as torch.Tensor
        chunk_size: Number of frames to process at once

    Returns:
        Binary frames [T, H, W] as bool torch.Tensor
    """
    T = flux.shape[0]
    # binary_chunks = []
    binary = torch.zeros_like(flux, dtype=torch.bool, device="cpu")

    for start_idx in tqdm(range(0, T, chunk_size), desc="Poisson sampling (GPU)"):
        end_idx = min(start_idx + chunk_size, T)
        flux_chunk = flux[start_idx:end_idx].to(device)
        prob = 1.0 - torch.exp(-flux_chunk)
        binary_chunk = torch.rand_like(flux_chunk) < prob
        # binary_chunks.append(binary_chunk.to("cpu"))
        binary[start_idx:end_idx] = binary_chunk.to("cpu")
        # Clear intermediate tensors
        del prob, flux_chunk

    # binary = torch.cat(binary_chunks, dim=0)
    return binary


def binary_from_grayscale(
    grayscale_torch, target_ppp, device: torch.device=None,
):
    # grayscale_torch should be on cpu so it doesn't eat VRAM
    flux_result = calculate_flux_chunked_torch(grayscale_torch, target_ppp=target_ppp, device=device)

    # Clear grayscale from GPU immediately
    torch.cuda.empty_cache()

    # Handle CPU fallback results - keep on CPU if fallback was used
    using_cpu_fallback = isinstance(flux_result[0], np.ndarray)

    if using_cpu_fallback:
        print("  Using CPU fallback for flux calculation - keeping results on CPU...")
        flux_cpu = flux_result[0].astype(np.float32)
        a = flux_result[1]
    else:
        flux_torch, a = flux_result

    # Step 6: Generate binary frames (GPU, chunked, with CPU fallback)
    print("\nStep 6: Generating binary frames (chunked, GPU-accelerated)...")

    if using_cpu_fallback:
        # Use CPU for binary sampling if flux was computed on CPU
        print("  Using CPU for binary sampling (flux was computed on CPU)...")
        binary_cpu = poisson_sample_binary_cpu(flux_cpu)
    else:
        # Try GPU for binary sampling
        binary_result = poisson_sample_binary_chunked_torch(flux_torch, device=device)

        # Handle CPU fallback results for binary sampling
        if isinstance(binary_result, np.ndarray):
            print("  Using CPU fallback for binary sampling...")
            binary_cpu = binary_result
        else:
            # Convert GPU results to CPU for saving
            binary_cpu = binary_result.cpu().numpy()

        # Move flux to CPU for saving
        # flux_cpu = flux_torch.cpu().numpy().astype(np.float32)

        # Clear GPU memory
        del flux_torch
        if not isinstance(binary_result, np.ndarray):
            del binary_result
        torch.cuda.empty_cache()
    return binary_cpu


def process_video(video_idx: int, 
                 separated_videos_dir: Path,
                 manifest: List[dict],
                 rife_model,
                 rife_device,
                 output_dir: Path):
    """Process a single video through the full pipeline with GPU acceleration."""
    
    # Find video info from manifest
    video_info = None
    for vinfo in manifest:
        if vinfo['video_idx'] == video_idx:
            video_info = vinfo
            break
    
    if video_info is None:
        raise ValueError(f"Video index {video_idx} not found in manifest")
    
    src_fps = video_info["srcfps"]
    rife_interp_exp = video_info.get("rife_interp_exp", RIFE_INTERP_EXP_DEFAULT)
    linear_interp = video_info.get("linear_interp", LINEAR_INTERP_DEFAULT)
    linearize_color = video_info.get("linearize", True)
    video_dir = separated_videos_dir / video_info['subfolder']
    print(f"\nProcessing video {video_idx}: {video_info['subfolder']}")
    # print(f"  Frames: {video_info['start_frame']} to {video_info['end_frame']} ({video_info['num_frames']} frames)")

    # Step 1: Load frames (CPU)
    print("\nStep 1: Loading frames...")
    # frames = load_video_frames(video_dir)  # [T, H, W, 3]
    frames = utils.read_imgdir(video_dir, grayscale=False)  # [T, H, W, 3]
    T, H, W = frames.shape[0], frames.shape[1], frames.shape[2]
    print(f"  Loaded {len(frames)} frames of size {W}x{H}")

    # Step 2: RIFE 16x interpolation (GPU)
    if rife_interp_exp == 0 or rife_interp_exp is None:
        print("\nStep 2: Skipping RIFE interpolation (exp=0)")
        frames_rife = frames
    else:
        print(f"\nStep 2: RIFE {2**rife_interp_exp}x interpolation...")
        frames_rife = rife_interpolate(rife_model, rife_device, frames, exp=rife_interp_exp, scale=1.0)
        print(f"  RIFE output: {len(frames_rife)} frames")

        # Clear input frames from CPU memory
        del frames
    
    # Step 3: Linear 8x interpolation (CPU - fast enough, avoids GPU memory issues)
    if linear_interp <= 1 or linear_interp is None:
        print("\nStep 3: Skipping linear interpolation (factor<=1)")
        frames_linear = frames_rife
    else:
        print(f"\nStep 3: Linear {linear_interp}x interpolation (CPU)...")
        frames_linear = linear_temporal_interpolation(frames_rife, factor=linear_interp)
        print(f"  Linear output: {len(frames_linear)} frames")

        # Clear RIFE frames
        del frames_rife
    
    # Step 4: Color space conversion (GPU-accelerated, chunked, with CPU fallback)
    print("\nStep 4: Color space conversion (GPU-accelerated, chunked)...")
    try:
        grayscale_result = color_convert_chunked_torch(frames_linear, device=rife_device, linearize=linearize_color)
        # Check if result is numpy (CPU fallback) or torch tensor (GPU)
        if isinstance(grayscale_result, np.ndarray):
            print("  Using CPU fallback result, converting to CPU tensor...")
            grayscale_torch = torch.from_numpy(grayscale_result).float().to("cpu")
        else:
            grayscale_torch = grayscale_result
    except Exception as e:
        print(f"  Error in GPU color conversion: {e}")
        print("  Falling back to CPU color conversion...")
        grayscale_cpu = color_convert_cpu(frames_linear, linearize=linearize_color)
        grayscale_torch = torch.from_numpy(grayscale_cpu).float().to("cpu")
        del grayscale_cpu
    print(f"  Grayscale shape: {grayscale_torch.shape}")
    
    # Clear input frames from CPU memory
    del frames_linear
    torch.cuda.empty_cache()
    
    # # Step 5: Calculate flux (two versions) (GPU, chunked, with CPU fallback)
    # print("\nStep 5: Calculating flux (chunked, GPU-accelerated)...")
    # print("\nStep 6: Also saving after each ppp value...")
    # # Step 7: Save outputs (already on CPU)
    # print("\nStep 7: Saving outputs...")

    # Step 7: Save outputs
    os.makedirs(output_dir, exist_ok=True)

    T_new = grayscale_torch.shape[0]
    if linear_interp <= 1:
        out_fps_est = src_fps * (2 ** rife_interp_exp)
    else:
        out_fps_est = src_fps * (2 ** rife_interp_exp) * linear_interp
    out_fps = T_new / (T / src_fps)
    print(f"  Projected output FPS: {out_fps_est:.2f}")
    print(f"  Actual output FPS: {out_fps:.2f}")

    # binary_ppp1_cpu = binary_from_grayscale(grayscale_torch, target_ppp=1.0, device=rife_device)
    # zarr_ppp1_path = output_dir / f"{video_dir.stem}_rife{rife_interp_exp}_linear{linear_interp}_ppp1.zarr"
    # ihpp_data.write_quantaframes_zarr(zarr_ppp1_path, binary_ppp1_cpu, fps=out_fps, save_coords=False)
    # print(f"  Saved binary (PPP=1) to {zarr_ppp1_path}")
    # del binary_ppp1_cpu

    binary_ppp01_cpu = binary_from_grayscale(grayscale_torch, target_ppp=0.1, device=rife_device)
    zarr_ppp01_path = output_dir / f"{video_dir.stem}_rife{rife_interp_exp}_linear{linear_interp}_ppp0.1.zarr"
    ihpp_data.write_quantaframes_zarr(zarr_ppp01_path, binary_ppp01_cpu, fps=out_fps, save_coords=False)
    print(f"  Saved binary (PPP=0.1) to {zarr_ppp01_path}")
    del binary_ppp01_cpu

    # binary_ppp001_cpu = binary_from_grayscale(grayscale_torch, target_ppp=0.01, device=rife_device)
    # zarr_ppp001_path = output_dir / f"{video_dir.stem}_rife{rife_interp_exp}_linear{linear_interp}_ppp0.01.zarr"
    # ihpp_data.write_quantaframes_zarr(zarr_ppp001_path, binary_ppp001_cpu, fps=out_fps_est, save_coords=False)
    # print(f"  Saved binary (PPP=0.01) to {zarr_ppp001_path}")
    # del binary_ppp001_cpu

    # Save flux (using PPP=1 version)
    flux_path_zarr = output_dir / f"flux_{video_dir.stem}_rife{rife_interp_exp}_linear{linear_interp}.zarr"
    grayscale_np = grayscale_torch.numpy()
    compressor = zarr.codecs.BloscCodec(cname="zstd", clevel=5, shuffle="shuffle")
    chunks = (50, grayscale_np.shape[1], grayscale_np.shape[2])
    z = zarr.create_array(
        flux_path_zarr,
        shape=grayscale_np.shape,
        dtype=grayscale_np.dtype,
        chunks=chunks,
        compressors=compressor
    )
    z[:] = grayscale_np
    print(f"  Saved flux to {flux_path_zarr}")

    # Save binary frames (both versions)
    # save_binary_data_bitpacked expects [T, W, H] format
    # Our binary frames are [T, H, W], so we transpose: (0, 2, 1) -> [T, W, H]
    # ...why?
    # binary_ppp1_twh = np.transpose(binary_ppp1_cpu, (0, 2, 1))  # [T, H, W] -> [T, W, H]
    # binary_ppp01_twh = np.transpose(binary_ppp01_cpu, (0, 2, 1))  # [T, H, W] -> [T, W, H]
    
    # binary_ppp1_path = output_dir / f"binary_{video_idx}_ppp1.bin"
    # save_binary_data_bitpacked(binary_ppp1_twh, str(binary_ppp1_path), frame_width=W, frame_height=H)
    # print(f"  Saved binary (PPP=1) to {binary_ppp1_path}")
    
    # binary_ppp01_path = output_dir / f"binary_{video_idx}_ppp0.1.bin"
    # save_binary_data_bitpacked(binary_ppp01_twh, str(binary_ppp01_path), frame_width=W, frame_height=H)
    # print(f"  Saved binary (PPP=0.1) to {binary_ppp01_path}")

    print(f"\nCompleted processing video {video_idx}\n")


def main():
    parser = argparse.ArgumentParser(description='Generate SPAD dataset from video frames')
    parser.add_argument('--video-idx', type=int, default=None,
                       help='Process a single video index (for SLURM array jobs). If not provided, processes all videos in upsampling_vid_idxs.yaml')
    parser.add_argument('--base-dir', type=str, default=DEFAULT_BASE_DIR,
                       help='Base directory for data')
    args = parser.parse_args()
    
    # Paths
    base_dir = Path(args.base_dir)
    separated_videos_dir = base_dir / "separated_videos"
    manifest_path = separated_videos_dir / "videos_manifest.yaml"
    upsampling_idxs_path = base_dir / "upsampling_vid_idxs.yaml"
    output_dir = base_dir / "upsampled"
    rife_model_dir = (RIFE_DIR / "train_log").absolute()

    # Load manifest
    print("Loading video manifest...")
    with open(manifest_path, 'r') as f:
        manifest = yaml.safe_load(f)
    print(f"  Loaded {len(manifest)} videos in manifest")
    
    # Determine which videos to process
    if args.video_idx is not None:
        # SLURM array job: process single video
        video_idxs = [args.video_idx]
        print(f"  Processing single video (SLURM array job): {args.video_idx}")
    else:
        # Load video indices to process
        print("Loading upsampling indices...")
        with open(upsampling_idxs_path, 'r') as f:
            upsampling_data = yaml.safe_load(f)
        video_idxs = upsampling_data['idxs']
        print(f"  Processing {len(video_idxs)} videos: {video_idxs}")
    
    # Load RIFE model
    print("\nLoading RIFE model...")
    os.chdir(RIFE_DIR)  # Change to RIFE directory for model loading
    rife_model, rife_device = load_rife_model(str(rife_model_dir))
    os.chdir(Path(__file__).parent)  # Change back
    
    print(f"Using device: {rife_device}")
    # print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Process each video
    for video_idx in video_idxs:
        try:
            process_video(video_idx, separated_videos_dir, manifest, 
                         rife_model, rife_device, output_dir)
        except Exception as e:
            print(f"\nError processing video {video_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\nAll videos processed!")


if __name__ == "__main__":
    main()

