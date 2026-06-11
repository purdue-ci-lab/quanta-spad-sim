"""
General data processing functions.
"""
from pathlib import Path
import re
from typing import Union
import warnings

import numpy as np
from tqdm import tqdm
import zarr

import utils

try:
    import cupy as cp
except ImportError:
    cp = None


to_video = utils.to_video


def thin_events_uniform(events, keep_prob, dcr_rate=None, return_idx=False, seed=None, cuda=False):
    """
    Thin events uniformly with probability keep_prob.
    """
    xp = cp if cuda else np
    rng = xp.random.default_rng(seed=seed)
    mask = rng.random(events.shape[0]) < keep_prob
    if return_idx:
        return xp.nonzero(mask)
    return events[mask]


def write_quantaframes_zarr(path, frames, T_exp=None, fps=None, save_coords=False):
    """
    Write binary quanta frames to a Zarr group.

    Args:
        path (str or Path): Path to the output Zarr file.
        frames (np.ndarray): Array of shape (T, H, W) with binary frames.
        T_exp (float): exposure time in seconds.
        fps (float): if T_exp is None, fps must be provided to calculate T_exp.
        save_coords (bool): if True, also saves the (t, y, x) coordinates of events (for
            speed purposes so they don't have to be recomputed). Set to False to save
            disk space.
    """
    if T_exp is None:
        if fps is not None:
            T_exp = frames.shape[0] / fps
        else:
            raise ValueError("Either T_exp or fps must be provided")
    elif fps is not None and T_exp is not None:
        raise ValueError("Only one of T_exp or fps should be provided")
    elif T_exp is not None and fps is None:
        fps = frames.shape[0] / T_exp
    frames = np.asarray(frames, dtype="uint8")
    npoints = np.count_nonzero(frames)
    npts_persec_perpix = np.count_nonzero(frames, axis=0) / T_exp
    # statistics
    avg_pts_persec_perpixel = npoints / (T_exp * frames.shape[1] * frames.shape[2])
    max_pts_persec_perpixel = np.max(npts_persec_perpix)
    min_pts_persec_perpixel = np.min(npts_persec_perpix)
    med_pts_persec_perpixel = np.median(npts_persec_perpix)
    stddev_pts_persec_perpixel = np.std(npts_persec_perpix, ddof=1)
    fps = fps or (frames.shape[0] / T_exp)
    frame_size_bytes = frames.shape[1] * frames.shape[2]  # H * W for uint8
    max_chunk_bytes = 200_000_000
    max_frames_per_chunk = max(1, max_chunk_bytes // frame_size_bytes)
    if save_coords:
        t, y, x = binframes_to_spt(frames, T_exp=T_exp, normalize=True)
        quantadata = {
            "frames": frames,
            "t": t,
            "y": y,
            "x": x
        }
        chunks = {
            "frames": (max_frames_per_chunk, frames.shape[1], frames.shape[2]),
            "t": (100_000_000,),
            "y": (100_000_000,),
            "x": (100_000_000,),
        }
    else:
        quantadata = {
            "frames": frames
        }
        chunks = {
            "frames": (max_frames_per_chunk, frames.shape[1], frames.shape[2]),
        }
    compressors = {
        "frames": zarr.codecs.BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")
    }
    utils.save_arrs_to_zarr(
        quantadata, path,
        chunks=chunks,
        n_workers=48,
        overwrite=True,
        attrs={
            "fps": fps,
            "T_exp": T_exp,
            "T": frames.shape[0],
            "H": frames.shape[1],
            "W": frames.shape[2],
            "shape": "(T, H, W)",
            "npoints": npoints,
            "avg_pts_persec_perpixel": avg_pts_persec_perpixel,
            "max_pts_persec_perpixel": max_pts_persec_perpixel,
            "min_pts_persec_perpixel": min_pts_persec_perpixel,
            "med_pts_persec_perpixel": med_pts_persec_perpixel,
            "stddev_pts_persec_perpixel": stddev_pts_persec_perpixel,
        },
        compressor_dict=compressors,
    )


def binframes_to_spt(frames: np.ndarray, T_exp=None, fps=None, keep_prob=1.0, normalize=True, cuda=False):
    """
    Convert (T, H, W) binary frames (0/1 per pixel) to spatiotemporal
    events (t, y, x).

    Args:
        frames (np.ndarray): array of shape (nframes, height, width) with
            binary pixel values.
        T_exp (float): exposure time in seconds.
        keep_prob (float): probability of keeping each event (for thinning).
        normalize (bool): if True, normalize x, y, t to [0, 1].

    Returns:
        tuple:
            - t: 1D array of time
            - y: 1D array of y coordinates
            - x: 1D array of x coordinates
    """
    xp = cp if cuda else np
    if T_exp is None:
        if fps is not None:
            T_exp = frames.shape[0] / fps
        else:
            raise ValueError("Either T_exp or fps must be provided")
    elif fps is not None and T_exp is not None:
        raise ValueError("Only one of T_exp or fps should be provided")
    # Get coordinates where pixel == 1
    t_coords, y_coords, x_coords = xp.nonzero(frames)
    t_coords = t_coords.astype("float64")
    y_coords = y_coords.astype("float64")
    x_coords = x_coords.astype("float64")

    # Stack and convert to float
    # x_coords += 0.5  # center of pixel x
    # y_coords += 0.5  # center of pixel y
    t_coords *= T_exp / frames.shape[0]  # convert frame index to time in seconds
    if keep_prob < 1.0:
        nevents_orig = t_coords.shape[0]
        idxs = thin_events_uniform(t_coords, return_idx=True, keep_prob=keep_prob, seed=42)
        t_coords = t_coords[idxs]
        y_coords = y_coords[idxs]
        x_coords = x_coords[idxs]
        print(f"Kept {len(idxs)} of {nevents_orig} points after thinning")
    if normalize:
        # Normalize x, y, t to [0, 1]
        x_coords /= frames.shape[2]  # x
        y_coords /= frames.shape[1]  # y
        t_coords /= T_exp  # t
    return t_coords, y_coords, x_coords
