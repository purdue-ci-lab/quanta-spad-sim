"""
Miscellaneous utility functions.
"""
import atexit
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
from itertools import product
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import weakref

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
import zarr


def dot_clean_dir(path):
    """
    Genuinely genuinely genuinely FUCK YOU MACOS
    Stop making these FUCKING DOT FILES
    """
    path = Path(path)
    result = subprocess.run(
        ["dot_clean", "-mn", str(path.absolute())],
        capture_output=True, text=True, check=True
    )
    return result


def is_json_serializable(x):
    """
    'Easier to ask forgiveness than permission.'
    """
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


def make_json_serializable(value):
    """Convert a value to a JSON-serializable format."""

    # Numpy scalar / 0-d array (singleton)
    if isinstance(value, np.generic):
        return value.item()  # converts to native Python scalar

    # Numpy array
    if isinstance(value, np.ndarray):
        return value.tolist()  # converts to nested Python list

    # Recursively handle dicts
    if isinstance(value, dict):
        return {k: make_json_serializable(v) for k, v in value.items()}

    # Recursively handle lists/tuples
    if isinstance(value, (list, tuple)):
        converted = [make_json_serializable(v) for v in value]
        return converted if isinstance(value, list) else tuple(converted)

    return value  # already serializable (str, int, float, bool, None, etc.)


def save_arrs_to_zarr(
    data_dict,
    zarr_path,
    chunks=None,
    attrs=None,
    compressor_dict=None,
    n_workers=8,
    overwrite=True,
):
    """
    Save multiple numpy arrays (e.g., freqinfo) into a single Zarr group concurrently.
    Uses Group.create_array (newer zarr API).

    Args:
        data_dict : dict
            Mapping of name->numpy array to save.
        zarr_path : path-like
            Path to output Zarr directory (e.g., "freqinfo.zarr").
        chunks : dict or None
            - If dict: mapping name->chunk tuple
            - If None: auto applied to all
        attrs : dict or None
            Global attribute metadata to set on the Zarr group.
        compressor_dict: dict of codec or None
            Codec for compression (default: whatever auto is in zarr).
        n_workers : int
            Number of worker threads to write large arrays concurrently.
        overwrite : bool
            If True, overwrite an existing zarr at zarr_path.
    """
    zarr_path = Path(zarr_path)
    if compressor_dict is None:
        compressor_dict = {}
        for key, arr in data_dict.items():
            # check if arr is suitable for shuffle or bitshuffle
            # i.e. if it's uint8 or boolean, use bitshuffle; otherwise, use shuffle
            if arr.itemsize == 1:
                compressor_dict[key] = zarr.codecs.BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")
            else:
                compressor_dict[key] = zarr.codecs.BloscCodec(cname="zstd", clevel=5, shuffle="shuffle")
    if chunks is None:
        chunks = {}

    try:
        root = zarr.open_group(zarr_path, mode="w" if overwrite else "w-")
    except FileNotFoundError:
        dot_clean_dir(zarr_path)
        root = zarr.open_group(zarr_path, mode="w" if overwrite else "w-")

    if attrs is not None:
        for key, value in attrs.items():
            root.attrs[key] = make_json_serializable(value)

    for key, arr in data_dict.items():
        arr = np.asarray(arr)
        ds_chunks = chunks.get(key, "auto")
        ds_compressor = compressor_dict.get(key, "auto")
        ds = root.create_array(
            name=key,
            shape=arr.shape,
            dtype=arr.dtype,
            chunks=ds_chunks,
            compressors=ds_compressor,
            overwrite=True
        )

        # Generate slice tuples for all chunks
        slices_list = []
        for dim, chunk_size in zip(arr.shape, ds.chunks):
            # slice starts along this dimension
            starts = list(range(0, dim, chunk_size))
            slices_list.append(starts)

        # Cartesian product over all chunk starts -> all chunk positions
        chunk_starts = list(product(*slices_list))

        def write_chunk(start_indices):
            # Build slices for this chunk
            slc = tuple(
                slice(start, min(start + cs, dim))
                for start, cs, dim in zip(start_indices, ds.chunks, arr.shape)
            )
            ds[slc] = arr[slc]

        # Write chunks concurrently
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            list(tqdm(executor.map(write_chunk, chunk_starts), total=len(chunk_starts), desc=f"Writing {key}"))
    print(f"Saved arrays to Zarr at: {zarr_path}")
    return root  # return the zarr group (useful for immediate reading)


def read_video(path, return_framerate=False, grayscale=False):
    """
    Read a video file and return frames as a numpy array of
    shape (num_frames, height, width, 3) and the framerate.

    Args:
        path (str): Path to the video file.

    Returns:
        tuple:
        - frames (np.ndarray): Array of video frames.
        - framerate (float): video FPS.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {path}")

    framerate = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    pbar = tqdm(desc="Number of frames read")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        pbar.update(1)
    pbar.close()

    cap.release()

    frames = np.stack(frames, axis=0)

    if return_framerate:
        return frames, framerate
    return frames


def natural_sort_key(s: str):
    """Splits string into text and integer chunks for natural ordering."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]


def read_imgdir(path, grayscale: bool = False) -> np.ndarray:
    path = Path(path)
    supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    tiff_extensions = {".tiff", ".tif"}

    image_files = sorted(
        [f for f in os.listdir(path) if os.path.splitext(f)[1].lower() in supported_extensions],
        key=natural_sort_key
    )

    images = []
    for filename in tqdm(image_files):
        ext = os.path.splitext(filename)[1].lower()
        img = Image.open(path / filename)

        if ext in tiff_extensions:
            # tiff files may have higher bit depth; preserve it by converting to numpy array directly
            arr = np.array(img)  # preserves native bit depth (e.g. uint16 for 12-bit)
            if not grayscale and arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)  # expand to RGB if needed
        else:
            img = img.convert("L" if grayscale else "RGB")
            arr = np.array(img)  # uint8

        images.append(arr)

    return np.array(images)


def to_video(
    frames: np.ndarray, path, res_scale=1.0, playback_fps=None, gamma=1.0, cmap=None, fileformat=None,
    vmin=None, vmax=None, quantile=None, framenames=None
):
    """
    Saves video frame arrays to a video file or sequence of PNGs. If path has no extension, 
    it is treated as a directory and individual image files are saved.

    Args:
        frames (np.ndarray): (T x H x W x C) (RGB) or (T x H x W) (intensity) video frames.
        path (str or Path): output video file path or directory for image files.
        res_scale (float): resolution scaling factor with nearest neighbor interpolation.
        cmap: ignored if frames are RGB; otherwise, matplotlib colormap name or object.
        fileformat (str or None): video format (e.g., "mp4", "avi"), or image format (e.g., "png");
            if None, inferred from path suffix.
        quantile (float or None): if not None, use quantiles to determine vmin and vmax for normalization
            (ignored if vmin or vmax are specified).
    """
    path = Path(path)
    if cmap is None:
        cmap = "viridis"
    cmap_fn = plt.get_cmap(cmap)
    is_rgb = False
    if frames.ndim == 4:
        if frames.shape[3] == 3:
            is_rgb = True
        else:
            raise ValueError("4D frames array must have shape (T, H, W, 3) for RGB video")
    elif frames.ndim == 3:
        is_rgb = False
    else:
        raise ValueError("frames must be a 3D or 4D numpy array")

    # compute a normalized intensity in [0,1] for colormap input
    if vmax is None:
        if quantile is not None:
            vmax = float(np.quantile(frames, quantile))
        else:
            vmax = float(np.max(frames))
    if vmin is None:
        if quantile is not None:
            vmin = float(np.quantile(frames, 1 - quantile))
        else:
            vmin = float(np.min(frames))
            if vmin >= 0:
                print(f"vmin was not specified and frames have non-negative values, so using vmin=0 for more accurate scaling")
                vmin = 0.0

    H, W = frames.shape[1], frames.shape[2]
    if res_scale != 1.0:
        out_W = int(W * res_scale)
        out_H = int(H * res_scale)
    else:
        out_W = W
        out_H = H
    # if path is a directory, write individual image files
    is_video_file = path.suffix in [".mp4", ".avi", ".mov", ".mkv"]
    if not is_video_file:
        path.mkdir(parents=True, exist_ok=True)
        if fileformat is None:
            fileformat = "png"
    else:
        if playback_fps is None:
            raise ValueError("playback_fps must be specified if saving a video file")
        path.parent.mkdir(parents=True, exist_ok=True)
        if fileformat is None:
            fileformat = path.suffix[1:].lower()
        codec = get_codec_for_format(fileformat)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        vidwriter = cv2.VideoWriter(str(path), fourcc, playback_fps, (out_W, out_H), isColor=True)

    max_frames = len(frames)

    if not is_video_file:
        allpaths = []
    for i in tqdm(range(max_frames), desc="Writing video frames"):
        intensity = (np.clip(frames[i], vmin, vmax) - vmin) / (vmax - vmin)  # normalize to [0,1]
        if gamma != 1:
            intensity = intensity ** gamma
        if is_rgb:
            rgb_mapped = (intensity * 255.0).astype(np.uint8)  # (H,W,3) in RGB
        else:
            # apply matplotlib colormap -> returns RGBA in [0,1]
            rgba_mapped = cmap_fn(intensity)  # shape (H,W,4)
            rgb_mapped = (rgba_mapped[..., :3] * 255.0).astype(np.uint8)  # (H,W,3) in RGB
        bgr_mapped = rgb_mapped[..., ::-1]  # convert to BGR for OpenCV
        if res_scale != 1.0:
            bgr_mapped = cv2.resize(bgr_mapped, (out_W, out_H), interpolation=cv2.INTER_NEAREST)

        if is_video_file:
            vidwriter.write(bgr_mapped)
        else:
            if framenames is None:
                frame_path = path / f"frame_{i:05d}.{fileformat}"
            else:
                frame_path = path / f"{framenames[i]}.{fileformat}"
            if fileformat.lower() == "png":
                # higher compression level because there's thousands of frames
                # reminder for anyone reading here; IT'S LOSSLESS COMPRESSION BECAUSE IT'S A PNG
                cv2.imwrite(str(frame_path), bgr_mapped, [cv2.IMWRITE_PNG_COMPRESSION, 5])
            else:
                cv2.imwrite(str(frame_path), bgr_mapped)
            allpaths.append(frame_path)
    if is_video_file:
        vidwriter.release()
    if not is_video_file:
        return allpaths
    return path


def avi_to_mov(avi_path, mov_path=None):
    """
    Converts an AVI video file to a MOV file using ffmpeg with ProRes codec.
    """
    avi_path = Path(avi_path)
    if mov_path is None:
        mov_path = avi_path.with_suffix('.mov')
    subprocess.run([
        "ffmpeg",
        "-i", str(avi_path),
        "-y",  # override
        "-c:v", "prores_ks",
        "-profile:v", "3",  # 3 = HQ
        "-c:a", "copy",
        str(mov_path)
    ], check=True)
    print(f"Converted {avi_path} to {mov_path}")
    return mov_path


def get_codec_for_format(format: str):
    """
    Get appropriate fourcc codec string for given video format.
    For MP4, tries avc1 first and falls back to mp4v if unavailable.
    """
    format = format.lower()
    if format == "mp4":
        return _get_mp4_codec()
    elif format == "avi":
        return "FFV1"
    elif format == "mov":
        return "avc1"
    else:
        raise ValueError(f"I haven't added the codec for: {format}")


def _get_mp4_codec() -> str:
    """
    Colab silently fails to write MP4 files with avc1 (H.264) codec, but mp4v works fine.
    Check which codec is available in this OpenCV build by trying to write a small test video.

    Test whether avc1 (H.264) is available in this OpenCV build.
    Falls back to mp4v if not. Raises RuntimeError if neither works.
    """
    import tempfile, os
    test_frame = np.zeros((64, 64, 3), dtype=np.uint8)
    for codec in ["avc1", "mp4v"]:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            tmp_path = f.name
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(tmp_path, fourcc, 24, (64, 64), isColor=True)
            writer.write(test_frame)
            writer.release()
            if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                print(f"MP4 codec selected: {codec}")
                return codec
            else:
                print(f"MP4 codec '{codec}' produced no output, trying next...")
        except Exception as e:
            print(f"MP4 codec '{codec}' raised an error: {e}, trying next...")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    raise RuntimeError("No working MP4 codec found (tried avc1, mp4v). Consider using imageio+ffmpeg instead.")


def format_number_unit(value, decimals=2):
    """
    Format a number to a human-friendly string using suffixes up to G (giga, 10^9).
    Examples:
        123        -> "123"
        1234       -> "1.23K"
        1500000    -> "1.5M"
        -2500000000-> "-2.5G"

    Parameters
        value: int or float-like value to format
        decimals: max number of decimal places for the scaled value (default 2)

    Returns
        A string with the value scaled and suffixed appropriately.
    """
    try:
        v = float(value)
    except Exception:
        return str(value)

    if not math.isfinite(v):
        return str(value)

    sign = "-" if v < 0 else ""
    v_abs = abs(v)

    # handle zero explicitly
    if v_abs == 0:
        return "0"

    # determine exponent in steps of 1000 (0 -> "", 1 -> K, 2 -> M, 3 -> G)
    exp = int(math.floor(math.log10(v_abs) / 3)) if v_abs >= 1 else 0
    exp = max(0, min(exp, 3))  # clamp to available hard-coded units

    # hard-coded units via conditionals
    if exp == 0:
        unit = ""
    elif exp == 1:
        unit = "K"
    elif exp == 2:
        unit = "M"
    elif exp == 3:
        unit = "G"
    else:
        unit = "G"

    scaled = v_abs / (1000 ** exp)
    fmt = f"{scaled:.{decimals}f}".rstrip("0").rstrip(".")
    return f"{sign}{fmt}{unit}"


def _norm_slice(s: slice, n: int):
    """Return (start, stop, step) with Python's slice semantics."""
    start, stop, step = s.indices(n)
    if step <= 0:
        raise ValueError("Only positive step slices are supported here.")
    return start, stop, step


def map_chunk_to_compact_selection(
    *,
    chunk_start: int,
    chunk_stop: int,
    inner_start: int,
    inner_stop: int,
    sel: slice,
    n_global: int,
    sel_origin: int,
):
    """
    Intersect:
        [chunk_start, chunk_stop)  (step=1)
      ∩ [inner_start, inner_stop)  (step=1)
      ∩ sel (step=sel_step, in global coords)

    Returns:
        out_slc   : slice into compacted output axis (step=1)
        chunk_slc : slice into chunk-local axis (step=sel_step, relative to chunk_start)
    where compacted output axis is defined by indices in sel:
        out_index = (global_index - sel_start) // sel_step

    Parameters:
      sel_origin: the sel_start used to define the compacted output indexing.
                  Typically sel_origin == sel_start (after .indices()).
                  (Kept explicit so you don't accidentally use chunk_start.)
    """
    sel_start, sel_stop, sel_step = _norm_slice(sel, n_global)

    # First apply the step=1 constraints
    valid_start = max(chunk_start, inner_start)
    valid_stop  = min(chunk_stop,  inner_stop)
    if valid_start >= valid_stop:
        return slice(0, 0, 1), slice(0, 0, 1)

    # Now intersect with sel's bounds (still step=1 for the interval)
    overlap_start = max(valid_start, sel_start)
    overlap_stop  = min(valid_stop,  sel_stop)
    if overlap_start >= overlap_stop:
        return slice(0, 0, 1), slice(0, 0, 1)

    # Find the first index >= overlap_start that satisfies sel congruence:
    # global ≡ sel_start (mod sel_step)
    rem = (overlap_start - sel_start) % sel_step
    first = overlap_start if rem == 0 else overlap_start + (sel_step - rem)
    if first >= overlap_stop:
        return slice(0, 0, 1), slice(0, 0, 1)

    # Find the last index < overlap_stop satisfying the congruence
    last = overlap_stop - 1
    last -= (last - sel_start) % sel_step
    if last < first:
        return slice(0, 0, 1), slice(0, 0, 1)

    # Map to compact output coordinates (step 1)
    out_start = (first - sel_origin) // sel_step
    out_stop  = (last  - sel_origin) // sel_step + 1
    out_slc = slice(out_start, out_stop, 1)

    # Map to chunk-local coordinates (step sel_step)
    chunk_slc = slice(first - chunk_start, last - chunk_start + 1, sel_step)

    return out_slc, chunk_slc


def translate_slice_no_clamp(slc: slice, offset: int):
    """Translate without clamping (clamping is the caller's job)."""
    if offset == 0:
        return slc
    start = None if slc.start is None else slc.start + offset
    stop  = None if slc.stop  is None else slc.stop  + offset
    return slice(start, stop, slc.step)


def to_rgba_str(rgbas):
    """Convert array of RGBA values (0-255) to list of "rgba(r,g,b,a)" strings."""
    rgba_strs = []
    for rgba in rgbas:
        r, g, b, a = rgba
        rgba_strs.append(f"rgba({r},{g},{b},{a/255:.2f})")  # a scaled to [0,1]
    return rgba_strs


def resize_video(frames, w, h, interpolation=cv2.INTER_NEAREST):
    Nt, H, W = frames.shape
    resized_frames = np.zeros((Nt, h, w), dtype=frames.dtype)
    for t in tqdm(range(Nt), desc="Resizing video frames"):
        resized_frames[t] = cv2.resize(frames[t], (w, h), interpolation=interpolation)
    return resized_frames


def vals_to_cmap(vals, cmap="viridis", vmin=None, vmax=None):
    cmap_fn = plt.get_cmap(cmap)
    if vmin is None:
        vmin = np.min(vals)
    if vmax is None:
        vmax = np.max(vals)
    vals = np.interp(vals, (vmin, vmax), (0, 1))
    colors = cmap_fn(vals)
    return colors
