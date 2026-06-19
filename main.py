import argparse
from pathlib import Path

import cv2 as cv

from MOT import binarize_blackhat, blackhat_video, slowmo_video, upscale_video

DEFAULT_INPUT = "schooling-datasets/fish_30.mp4"


def _load_grayscale_frames(path: str) -> list:
    cap = cv.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame.ndim == 3:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError(f"No frames read from {path}")
    return frames


def _default_output(input_path: str, suffix: str) -> str:
    path = Path(input_path)
    return str(path.with_name(f"{path.stem}_{suffix}{path.suffix}"))


def add_io_args(parser: argparse.ArgumentParser, *, default_suffix: str) -> None:
    parser.add_argument(
        "--input",
        "-i",
        default=DEFAULT_INPUT,
        help="Input video path",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output video path (default: derived from input)",
    )
    parser.set_defaults(_default_suffix=default_suffix)


def resolve_output(args: argparse.Namespace) -> str:
    if args.output is not None:
        return args.output
    return _default_output(args.input, args._default_suffix)


def cmd_slowmo(args: argparse.Namespace) -> None:
    slowmo_video(args.input, resolve_output(args), args.scale)


def cmd_upscale(args: argparse.Namespace) -> None:
    upscale_video(args.input, resolve_output(args), args.scale)


def cmd_blackhat(args: argparse.Namespace) -> None:
    blackhat_video(args.input, resolve_output(args), args.kernel_size)


def cmd_binarize(args: argparse.Namespace) -> None:
    frames = _load_grayscale_frames(args.input)
    binarize_blackhat(frames, resolve_output(args), args.threshold, args.close_kernel)


def cmd_pipeline(args: argparse.Namespace) -> None:
    input_path = args.input
    slowmo_path = args.slowmo_output or _default_output(input_path, "slowmo")
    upscaled_path = args.upscaled_output or _default_output(input_path, "upscaled")
    blackhat_path = args.blackhat_output or _default_output(input_path, "blackhat")
    binarized_path = args.binarized_output or _default_output(input_path, "binarized")

    slowmo_video(input_path, slowmo_path, args.temporal_scale)
    upscale_video(slowmo_path, upscaled_path, args.spatial_scale)
    blackhat_video(upscaled_path, blackhat_path, args.kernel_size)
    binarize_blackhat(
        _load_grayscale_frames(blackhat_path),
        binarized_path,
        args.threshold,
        args.close_kernel,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fish video processing pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    slowmo = subparsers.add_parser("slowmo", help="Temporal slow-motion via frame interpolation")
    add_io_args(slowmo, default_suffix="slowmo")
    slowmo.add_argument("--scale", type=int, default=4, help="Temporal stretch factor")
    slowmo.set_defaults(func=cmd_slowmo)

    upscale = subparsers.add_parser("upscale", help="Spatial upscaling with Lanczos")
    add_io_args(upscale, default_suffix="upscaled")
    upscale.add_argument("--scale", type=int, default=4, help="Spatial upscale factor")
    upscale.set_defaults(func=cmd_upscale)

    blackhat = subparsers.add_parser("blackhat", help="Black-hat morphology on grayscale frames")
    add_io_args(blackhat, default_suffix="blackhat")
    blackhat.add_argument("--kernel-size", type=int, default=9, help="Morphology kernel size")
    blackhat.set_defaults(func=cmd_blackhat)

    binarize = subparsers.add_parser("binarize", help="Binarize black-hat video")
    add_io_args(binarize, default_suffix="binarized")
    binarize.add_argument("--threshold", type=int, default=5, help="Binarization threshold")
    binarize.add_argument("--close-kernel", type=int, default=3, help="Morphological close kernel size")
    binarize.set_defaults(func=cmd_binarize)

    pipeline = subparsers.add_parser("pipeline", help="Run all phases in order")
    pipeline.add_argument("--input", "-i", default=DEFAULT_INPUT, help="Input video path")
    pipeline.add_argument("--temporal-scale", type=int, default=4, help="Slowmo factor")
    pipeline.add_argument("--spatial-scale", type=int, default=4, help="Upscale factor")
    pipeline.add_argument("--kernel-size", type=int, default=9, help="Black-hat kernel size")
    pipeline.add_argument("--threshold", type=int, default=5, help="Binarization threshold")
    pipeline.add_argument("--close-kernel", type=int, default=3, help="Morphological close kernel size")
    pipeline.add_argument("--slowmo-output", default=None, help="Intermediate slowmo output path")
    pipeline.add_argument("--upscaled-output", default=None, help="Intermediate upscaled output path")
    pipeline.add_argument("--blackhat-output", default=None, help="Intermediate blackhat output path")
    pipeline.add_argument("--binarized-output", default=None, help="Final binarized output path")
    pipeline.set_defaults(func=cmd_pipeline)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
