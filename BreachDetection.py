import os
def detect_breaches(video_path, conf_threshold, save_video):
    print("Running breach detection...")
    import cv2
    from ultralytics import YOLO
    import time

    MODEL_PATH = "SturgeonBest.pt"
    SAVE_FRAMES = False
    OUTPUT_VIDEO_PATH = os.path.join("outputVideos", "output_with_boxes.mp4")

    model = YOLO(MODEL_PATH)
    start_time = time.time()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open video {video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))
    cap.release()

    timestamps = []

    for frame_idx, result in enumerate(model.predict(source=video_path, conf=conf_threshold, stream=True, verbose=False)):
        annotated_frame = result.plot()
        if save_video:
            writer.write(annotated_frame)
        if hasattr(result, "boxes") and len(result.boxes) > 0:
            time_sec = frame_idx / fps
            timestamps.append(time_sec)
            if SAVE_FRAMES:
                annotated = result.orig_img
                out_name = f"detection_frame_{frame_idx:06d}.jpg"
                cv2.imwrite(out_name, annotated)

    if save_video:
        writer.release()
        print(f"Saved annotated video to {OUTPUT_VIDEO_PATH}")
    cv2.destroyAllWindows()

    threshold = 2.0
    filtered_ts = []
    last_det = None
    for t in timestamps:
        if last_det is None or (t - last_det) >= threshold:
            filtered_ts.append(t)
        last_det = t

    print("\nBreach detection complete.")
    num = len(filtered_ts)
    print(f"Number of breaches detected: {num}")
    if num > 0:
        # Format timestamps as hours:minutes:seconds
        formatted = []
        for t in filtered_ts:
            h = int(t // 3600)
            m = int((t % 3600) // 60)
            s = int(t % 60)
            formatted.append(f"{h:02d}:{m:02d}:{s:02d}")
        print("Timestamps (h:m:s): " + ", ".join(formatted))

    end_time = time.time()
    elapsed = end_time - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = elapsed % 60
    print(f"Processing time: {hours}h {minutes}m {seconds:.2f}s")

if __name__ == "__main__":
    video_file = input("Insert Video Filename (e.g., SturgeonVideo.mp4): ")
    video_path = os.path.join("videos", video_file)
    confidence = input("Insert Confidence Level (0.0 - 1.0): ")
    try:
        confidence = float(confidence)
    except ValueError:
        print("Invalid confidence level. Using default 0.3.")
        confidence = 0.3
    if not (0 < confidence < 1):
        print("Invalid confidence level. Using default 0.3.")
        confidence = 0.3

    save_input = input("Do you want to save the new video with bounding boxes? (y/n): ").lower()
    save = save_input == "y"

    detect_breaches(video_path, conf_threshold=confidence, save_video=save)
