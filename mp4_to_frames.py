import cv2
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm


input_video_path = Path("data/phantom-synth/separated_videos/stirfry-fps1000.mp4")
output_folder = Path("data/phantom-synth/separated_videos") / input_video_path.stem
# Resize: @Sotiris Change Here to Desired Size
factor = 512 / 1920
W = round(1920 * factor)
H = round(1080 * factor)
print(f"Resizing frames to WxH {W}x{H}")

output_folder.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(input_video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frame_idx = 0
pbar = tqdm(total=frame_count)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert from BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if W != frame.shape[1] or H != frame.shape[0]:
        frame_resized = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
    else:
        frame_resized = frame

    # Center crop: @Sotiris Change Here to Desired Size
    crop_width, crop_height = W, H
    x_center = frame_resized.shape[1] // 2
    y_center = frame_resized.shape[0] // 2
    x1 = x_center - crop_width // 2
    x2 = x_center + crop_width // 2
    y1 = y_center - crop_height // 2
    y2 = y_center + crop_height // 2

    frame_cropped = frame_resized[y1:y2, x1:x2, :]

    # Save frame as PNG
    out_path = os.path.join(output_folder, f"frame_{frame_idx:05d}.png")
    # Convert RGB back to uint8 if not already
    frame_cropped_uint8 = np.clip(frame_cropped, 0, 255).astype(np.uint8)
    # Save using PIL for convenience
    from PIL import Image
    Image.fromarray(frame_cropped_uint8).save(out_path, format="PNG", compress_level=5)

    frame_idx += 1
    pbar.update(1)

cap.release()
pbar.close()
print(f"Saved {frame_idx} frames to {output_folder}")
