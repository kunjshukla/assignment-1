import os
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
import time
from concurrent.futures import ThreadPoolExecutor

# Directories
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"
ANNOTATED_FOLDER = "annotated_frames"
SELECTED_FOLDER = "selected_frames"

# Ensure directories exist
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, ANNOTATED_FOLDER, SELECTED_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Load YOLO model once
MODEL_PATH = "yolov8n.pt"
model = YOLO(MODEL_PATH)


def clear_directory(folder):
    """Clears all files in the given directory."""
    for file in os.listdir(folder):
        os.remove(os.path.join(folder, file))


def extract_frames(video_path, frame_count=900):
    """Extracts evenly spaced frames from a video."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total_frames // frame_count)
    frame_paths = []

    for i in range(0, total_frames, interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame_path = os.path.join(OUTPUT_FOLDER, f"frame_{len(frame_paths)}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
    cap.release()
    return frame_paths


def annotate_and_rank_frames(frame_paths, batch_size=16):
    """Annotates frames with YOLO detections and ranks them by confidence."""
    annotated_paths = []
    ranked_frames = []

    for i in range(0, len(frame_paths), batch_size):
        batch = frame_paths[i : i + batch_size]
        frames = [cv2.imread(path) for path in batch]
        
        # Batch inference
        results = model(frames)

        for idx, result in enumerate(results):
            # Save annotated frame
            annotated_frame = result.plot()
            annotated_path = os.path.join(
                ANNOTATED_FOLDER, os.path.basename(batch[idx])
            )
            cv2.imwrite(annotated_path, annotated_frame)
            annotated_paths.append(annotated_path)

            # Calculate ranking score (average confidence)
            if hasattr(result, "boxes") and result.boxes is not None:
                confidences = (
                    result.boxes.conf.cpu().numpy()
                    if hasattr(result.boxes, "conf")
                    else []
                )
                score = np.mean(confidences) if len(confidences) > 0 else 0
            else:
                score = 0
            ranked_frames.append((annotated_path, score))

    # Sort frames by confidence score in descending order
    ranked_frames = sorted(ranked_frames, key=lambda x: x[1], reverse=True)
    return [frame[0] for frame in ranked_frames]


def save_top_frames(frame_paths, top_n=20):
    """Saves the top N ranked frames."""
    for i, frame_path in enumerate(frame_paths[:top_n]):
        selected_path = os.path.join(SELECTED_FOLDER, f"frame_{i}.jpg")
        frame = cv2.imread(frame_path)
        cv2.imwrite(selected_path, frame)


# Streamlit App
st.title("Shoppable Item Detection")
st.write("Upload a video to detect and rank frames based on shoppable items.")

# Video Upload
uploaded_file = st.file_uploader("Upload a video...", type=["mp4", "avi", "mov"])
if uploaded_file:
    video_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("Video uploaded successfully!")

    if st.button("Process Video"):
        overall_start_time = time.time()  # Overall process start time

        # Clear previous outputs
        clear_directory(OUTPUT_FOLDER)
        clear_directory(ANNOTATED_FOLDER)
        clear_directory(SELECTED_FOLDER)

        # Extract frames
        start_time = time.time()
        st.write("Extracting frames...")
        frame_paths = extract_frames(video_path, frame_count=900)
        st.write(f"Extracted {len(frame_paths)} frames.")
        st.write(f"Frame extraction time: {time.time() - start_time:.2f} seconds")

        # Annotate and rank frames
        start_time = time.time()
        st.write("Annotating and ranking frames with YOLO detections...")
        ranked_frames = annotate_and_rank_frames(frame_paths)
        st.success("Frames annotated, ranked, and saved.")
        st.write(f"Annotation and ranking time: {time.time() - start_time:.2f} seconds")

        # Save top frames
        start_time = time.time()
        st.write("Saving top-ranked frames...")
        save_top_frames(ranked_frames, top_n=20)
        st.success("Top 20 frames saved.")
        st.write(f"Saving time: {time.time() - start_time:.2f} seconds")

        # Display top-ranked frames
        st.write("Displaying top-ranked frames:")
        for frame_path in ranked_frames[:20]:
            st.image(frame_path, caption=os.path.basename(frame_path))

        # Display overall processing time
        st.write(f"Total processing time: {time.time() - overall_start_time:.2f} seconds")
