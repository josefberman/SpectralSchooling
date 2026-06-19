import cv2 as cv
from tqdm import tqdm


def upscale_video(input_path, output_path, scale):
    """
    Upscale the video by `scale` times using Lanczos interpolation.
    Args:
        input_path: Path to the input video.
        output_path: Path to the output video.
        scale: The number of times to upscale the video.

    Returns:
        None
    """
    cap = cv.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")

    fps = cap.get(cv.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    upscaled_size = (width * scale, height * scale)
    writer = cv.VideoWriter(output_path, fourcc, fps, upscaled_size, True)

    with tqdm(total=n_frames, desc="Upscaling video") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            upscaled = cv.resize(frame, None, fx=scale, fy=scale, interpolation=cv.INTER_LANCZOS4)
            writer.write(upscaled)
            pbar.update(1)

    cap.release()
    writer.release()
    print(f"Saved {scale}x upscaled video to {output_path}")


def interpolate_frame(frame1, frame2, alpha):
    """
    Linearly blend frame1 and frame2.

    Args:
        frame1: The first frame.
        frame2: The second frame.
        alpha: Interpolation factor (0: frame1, 1: frame2).

    Returns:
        The interpolated frame.
    """
    return cv.addWeighted(frame1, 1.0 - alpha, frame2, alpha, 0)


def slowmo_video(input_path, output_path, scale):
    """
    Extend video duration by inserting interpolated frames between consecutive frames.

    For each pair of consecutive frames, inserts `scale - 1` blended intermediate
    frames. Output length is N + (N - 1) * (scale - 1) for N input frames.

    Args:
        input_path: Path to the input video.
        output_path: Path to the output video.
        scale: Temporal stretch factor (frames inserted between each pair).

    Returns:
        None
    """
    cap = cv.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")

    fps = cap.get(cv.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    writer = cv.VideoWriter(output_path, fourcc, fps, (width, height), True)

    ret, prev = cap.read()
    if not ret:
        cap.release()
        writer.release()
        raise ValueError(f"No frames in video: {input_path}")

    writer.write(prev)
    with tqdm(total=n_frames, desc="Slowmo video") as pbar:
        pbar.update(1)
        while True:
            ret, next_frame = cap.read()
            if not ret:
                break
            if scale > 1:
                for i in range(1, scale):
                    alpha = i / scale
                    writer.write(interpolate_frame(prev, next_frame, alpha))
            writer.write(next_frame)
            prev = next_frame
            pbar.update(1)

    cap.release()
    writer.release()
    print(f"Saved {scale}x slowmo video to {output_path}")


def blackhat_video(input_path, output_path, kernel_size=9):
    """
    Apply blackhat on the video.
    Args:
        input_path: Path to the input video.
        output_path: Path to the output video.
        kernel_size: The size of the kernel for the blackhat operation.

    Returns:
        None
    """
    cap = cv.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")
    fps = cap.get(cv.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    writer = cv.VideoWriter(output_path, fourcc, fps, (width, height), False)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))

    with tqdm(total=n_frames, desc="Blackhat video") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            blackhat = cv.morphologyEx(gray, cv.MORPH_BLACKHAT, kernel)
            writer.write(blackhat)
            pbar.update(1)

    cap.release()
    writer.release()
    print(f"Saved blackhat video to {output_path}")


def binarize_blackhat(blackhat_frames, output_path, threshold=5, close_kernel=3):
    """
    Binarize a sequence of blackhat video frames using tqdm for progress.
    Args:
        blackhat_frames: An iterable of blackhat video frames (grayscale).
        output_path: Path to the output video.
        threshold: The threshold for the binarization.
        close_kernel: The size of the kernel for the closing operation.
        
    Returns:
        None
    """
    binarized_frames = []
    with tqdm(total=len(blackhat_frames), desc="Binarizing blackhat frames") as pbar:
        for blackhat in blackhat_frames:
            binary = cv.threshold(blackhat, threshold, 255, cv.THRESH_BINARY)[1]
            if close_kernel > 1:
                kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (close_kernel, close_kernel))
                binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)
            binarized_frames.append(binary)
            pbar.update(1)

    if not binarized_frames:
        raise ValueError("No binarized frames to save.")

    height, width = binarized_frames[0].shape
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    writer = cv.VideoWriter(output_path, fourcc, 30.0, (width, height), False)  # False: grayscale

    for frame in binarized_frames:
        writer.write(frame)
    writer.release()
    print(f"Saved binarized blackhat video to {output_path}")