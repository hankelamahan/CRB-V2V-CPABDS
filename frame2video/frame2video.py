#!/usr/bin/env python3
"""Convert OpenCOOD visualization frames into videos.

The default encoder is FFmpeg. OpenCV is provided as a fallback backend for
environments where ffmpeg is not installed.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageColor


FRAME_RE = re.compile(r"_frame_(\d+)(?=\.[^.]+$)", re.IGNORECASE)


class Frame2VideoError(RuntimeError):
    """User-facing conversion error."""


@dataclass(frozen=True)
class FrameInfo:
    path: Path
    frame_index: Optional[int]
    size: Tuple[int, int]


@dataclass(frozen=True)
class ResizeOptions:
    size: Optional[Tuple[int, int]]
    mode: str
    pad_color: Tuple[int, int, int]


def natural_key(value: str) -> Tuple[object, ...]:
    """Return a key that sorts digit runs numerically."""
    parts = re.split(r"(\d+)", value)
    key: List[object] = []
    for part in parts:
        key.append(int(part) if part.isdigit() else part.lower())
    return tuple(key)


def extract_frame_index(path: Path) -> Optional[int]:
    match = FRAME_RE.search(path.name)
    if match is None:
        return None
    return int(match.group(1))


def parse_resize(value: Optional[str]) -> Optional[Tuple[int, int]]:
    if not value:
        return None
    match = re.fullmatch(r"(\d+)x(\d+)", value.strip().lower())
    if match is None:
        raise argparse.ArgumentTypeError(
            "--resize must use WIDTHxHEIGHT, for example 1920x1080"
        )
    width, height = int(match.group(1)), int(match.group(2))
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("--resize dimensions must be positive")
    return width, height


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert image frames into MP4/AVI/GIF video files."
    )
    parser.add_argument("--input", required=True, type=Path,
                        help="Input directory containing frame images.")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output video path.")
    parser.add_argument("--pattern", default="*_frame_*.png",
                        help="Glob pattern used to select frames.")
    parser.add_argument("--fps", type=float, default=5.0,
                        help="Playback FPS. Must be positive.")
    parser.add_argument("--recursive", action="store_true",
                        help="Scan input directory recursively.")
    parser.add_argument("--sort", choices=["natural", "name", "mtime"],
                        default="natural",
                        help="Frame sort mode.")
    parser.add_argument("--start-frame", type=int, default=None,
                        help="Keep frames with parsed frame id >= this value.")
    parser.add_argument("--end-frame", type=int, default=None,
                        help="Keep frames with parsed frame id <= this value.")
    parser.add_argument("--stride", type=int, default=1,
                        help="Keep every Nth frame after sorting/filtering.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximum number of frames to use.")
    parser.add_argument("--backend", choices=["ffmpeg", "opencv"],
                        default="ffmpeg",
                        help="Encoding backend.")
    parser.add_argument("--codec", default="libx264",
                        help="FFmpeg video codec for non-GIF outputs.")
    parser.add_argument("--crf", type=int, default=18,
                        help="FFmpeg CRF quality value.")
    parser.add_argument("--preset", default="medium",
                        help="FFmpeg encoder preset.")
    parser.add_argument("--pix-fmt", default="yuv420p",
                        help="FFmpeg output pixel format.")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg",
                        help="FFmpeg executable name or path.")
    parser.add_argument("--resize", type=parse_resize, default=None,
                        help="Normalize output frame size, e.g. 1920x1080.")
    parser.add_argument("--resize-mode", choices=["fit", "stretch", "crop"],
                        default="fit",
                        help="Resize strategy.")
    parser.add_argument("--pad-color", default="black",
                        help="Padding color for fit mode. Named color or hex.")
    parser.add_argument("--allow-size-mismatch", action="store_true",
                        help="Allow different input sizes by normalizing them.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite an existing output file.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print planned frames/command without writing video.")
    return parser.parse_args(argv)


def validate_basic_args(args: argparse.Namespace) -> None:
    if not args.input.exists():
        raise Frame2VideoError(f"Input directory does not exist: {args.input}")
    if not args.input.is_dir():
        raise Frame2VideoError(f"Input path is not a directory: {args.input}")
    if args.fps <= 0:
        raise Frame2VideoError("--fps must be positive")
    if args.stride <= 0:
        raise Frame2VideoError("--stride must be positive")
    if args.limit is not None and args.limit <= 0:
        raise Frame2VideoError("--limit must be positive")
    if (args.start_frame is not None and args.end_frame is not None
            and args.start_frame > args.end_frame):
        raise Frame2VideoError("--start-frame can not exceed --end-frame")
    if args.output.exists() and not args.overwrite and not args.dry_run:
        raise Frame2VideoError(
            f"Output already exists, pass --overwrite to replace it: "
            f"{args.output}"
        )
    try:
        ImageColor.getrgb(args.pad_color)
    except ValueError as exc:
        raise Frame2VideoError(f"Invalid --pad-color: {args.pad_color}") from exc


def collect_frames(input_dir: Path, pattern: str,
                   recursive: bool = False) -> List[Path]:
    iterator = input_dir.rglob(pattern) if recursive else input_dir.glob(pattern)
    frames = [path for path in iterator if path.is_file()]
    if not frames:
        raise Frame2VideoError(
            f"No frames matched pattern {pattern!r} under {input_dir}"
        )
    return frames


def sort_frames(paths: Iterable[Path], mode: str = "natural") -> List[Path]:
    if mode == "mtime":
        return sorted(paths, key=lambda p: (p.stat().st_mtime, str(p)))
    if mode == "name":
        return sorted(paths, key=lambda p: (p.name, str(p)))
    if mode != "natural":
        raise Frame2VideoError(f"Unsupported sort mode: {mode}")

    def frame_aware_key(path: Path) -> Tuple[object, ...]:
        frame_index = extract_frame_index(path)
        if frame_index is not None:
            return (0, frame_index, natural_key(path.name),
                    natural_key(str(path)))
        return (1, natural_key(path.name), natural_key(str(path)))

    return sorted(paths, key=frame_aware_key)


def filter_frames(paths: Sequence[Path], start_frame: Optional[int],
                  end_frame: Optional[int], stride: int,
                  limit: Optional[int]) -> List[Path]:
    filtered: List[Path] = []
    needs_frame_id = start_frame is not None or end_frame is not None
    for path in paths:
        frame_index = extract_frame_index(path)
        if needs_frame_id and frame_index is None:
            continue
        if start_frame is not None and frame_index is not None:
            if frame_index < start_frame:
                continue
        if end_frame is not None and frame_index is not None:
            if frame_index > end_frame:
                continue
        filtered.append(path)

    filtered = filtered[::stride]
    if limit is not None:
        filtered = filtered[:limit]
    if not filtered:
        raise Frame2VideoError("No frames remain after filtering")
    return filtered


def validate_frames(paths: Sequence[Path], resize_size: Optional[Tuple[int, int]],
                    allow_size_mismatch: bool = False) -> List[FrameInfo]:
    infos: List[FrameInfo] = []
    unreadable: List[str] = []
    for path in paths:
        try:
            with Image.open(path) as image:
                image.load()
                size = image.size
        except Exception:
            unreadable.append(str(path))
            continue
        infos.append(FrameInfo(path=path,
                               frame_index=extract_frame_index(path),
                               size=size))

    if unreadable:
        joined = "\n  ".join(unreadable[:20])
        raise Frame2VideoError(f"Unreadable image file(s):\n  {joined}")

    sizes = {info.size for info in infos}
    if len(sizes) > 1 and resize_size is None and not allow_size_mismatch:
        preview = ", ".join(f"{w}x{h}" for w, h in sorted(sizes))
        raise Frame2VideoError(
            "Input frame sizes differ. Use --resize or "
            f"--allow-size-mismatch. Sizes: {preview}"
        )
    return infos


def output_size(infos: Sequence[FrameInfo],
                resize_size: Optional[Tuple[int, int]]) -> Tuple[int, int]:
    if resize_size is not None:
        return resize_size
    return infos[0].size


def resize_image(image: Image.Image, target_size: Tuple[int, int],
                 mode: str, pad_color: Tuple[int, int, int]) -> Image.Image:
    image = image.convert("RGB")
    src_w, src_h = image.size
    dst_w, dst_h = target_size
    if (src_w, src_h) == (dst_w, dst_h):
        return image

    if mode == "stretch":
        return image.resize(target_size, Image.Resampling.LANCZOS)

    if mode == "fit":
        scale = min(dst_w / src_w, dst_h / src_h)
        new_size = (max(1, round(src_w * scale)),
                    max(1, round(src_h * scale)))
        resized = image.resize(new_size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", target_size, pad_color)
        left = (dst_w - new_size[0]) // 2
        top = (dst_h - new_size[1]) // 2
        canvas.paste(resized, (left, top))
        return canvas

    if mode == "crop":
        scale = max(dst_w / src_w, dst_h / src_h)
        new_size = (max(1, round(src_w * scale)),
                    max(1, round(src_h * scale)))
        resized = image.resize(new_size, Image.Resampling.LANCZOS)
        left = max(0, (new_size[0] - dst_w) // 2)
        top = max(0, (new_size[1] - dst_h) // 2)
        return resized.crop((left, top, left + dst_w, top + dst_h))

    raise Frame2VideoError(f"Unsupported resize mode: {mode}")


def needs_preprocess(infos: Sequence[FrameInfo],
                     resize_options: ResizeOptions) -> bool:
    if resize_options.size is not None:
        return True
    first_size = infos[0].size
    if any(info.size != first_size for info in infos):
        return True
    suffixes = {info.path.suffix.lower() for info in infos}
    return len(suffixes) > 1


def link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


def prepare_temp_sequence(infos: Sequence[FrameInfo], temp_dir: Path,
                          resize_options: ResizeOptions) -> str:
    preprocess = needs_preprocess(infos, resize_options)
    target_size = output_size(infos, resize_options.size)
    suffix = ".png" if preprocess else infos[0].path.suffix.lower()

    for idx, info in enumerate(infos):
        dst = temp_dir / f"frame_{idx:06d}{suffix}"
        if preprocess:
            with Image.open(info.path) as image:
                processed = resize_image(image, target_size,
                                         resize_options.mode,
                                         resize_options.pad_color)
                processed.save(dst)
        else:
            link_or_copy(info.path, dst)
    return str(temp_dir / f"frame_%06d{suffix}")


def ffmpeg_base_command(ffmpeg_bin: str, overwrite: bool,
                        fps: float, frame_pattern: str) -> List[str]:
    overwrite_flag = "-y" if overwrite else "-n"
    return [
        ffmpeg_bin,
        overwrite_flag,
        "-framerate",
        format_fps(fps),
        "-i",
        frame_pattern,
    ]


def build_ffmpeg_commands(frame_pattern: str, output: Path,
                          args: argparse.Namespace,
                          temp_dir: Path) -> List[List[str]]:
    suffix = output.suffix.lower()
    if suffix == ".gif":
        palette = temp_dir / "palette.png"
        return [
            ffmpeg_base_command(args.ffmpeg_bin, True, args.fps, frame_pattern)
            + ["-vf", "palettegen", str(palette)],
            ffmpeg_base_command(args.ffmpeg_bin, args.overwrite, args.fps,
                                frame_pattern)
            + ["-i", str(palette), "-lavfi", "paletteuse", str(output)],
        ]
    return [
        ffmpeg_base_command(args.ffmpeg_bin, args.overwrite, args.fps,
                            frame_pattern)
        + [
            "-c:v",
            args.codec,
            "-preset",
            args.preset,
            "-crf",
            str(args.crf),
            "-pix_fmt",
            args.pix_fmt,
            str(output),
        ]
    ]


def run_command(command: Sequence[str]) -> None:
    completed = subprocess.run(command, text=True)
    if completed.returncode != 0:
        raise Frame2VideoError(
            f"Command failed with exit code {completed.returncode}: "
            f"{quote_command(command)}"
        )


def format_fps(fps: float) -> str:
    if float(fps).is_integer():
        return str(int(fps))
    return f"{fps:g}"


def quote_command(command: Sequence[str]) -> str:
    return " ".join(shlex_quote(part) for part in command)


def shlex_quote(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_./:=+,-]+", str(value)):
        return str(value)
    return "'" + str(value).replace("'", "'\"'\"'") + "'"


def write_video_ffmpeg(infos: Sequence[FrameInfo], args: argparse.Namespace,
                       resize_options: ResizeOptions) -> None:
    ffmpeg_path = shutil.which(args.ffmpeg_bin) or (
        args.ffmpeg_bin if Path(args.ffmpeg_bin).exists() else None
    )
    if ffmpeg_path is None:
        raise Frame2VideoError(
            "FFmpeg was not found. Install ffmpeg or use --backend opencv."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="frame2video_") as temp_name:
        temp_dir = Path(temp_name)
        frame_pattern = prepare_temp_sequence(infos, temp_dir, resize_options)
        for command in build_ffmpeg_commands(frame_pattern, args.output,
                                             args, temp_dir):
            run_command(command)


def write_video_opencv(infos: Sequence[FrameInfo], args: argparse.Namespace,
                       resize_options: ResizeOptions) -> None:
    if args.output.suffix.lower() == ".gif":
        raise Frame2VideoError("OpenCV backend can not write GIF; use ffmpeg.")
    try:
        import cv2
        import numpy as np
    except Exception as exc:
        raise Frame2VideoError(
            "OpenCV backend requires cv2 and numpy to be installed."
        ) from exc

    target_size = output_size(infos, resize_options.size)
    suffix = args.output.suffix.lower()
    fourcc_name = "MJPG" if suffix == ".avi" else "mp4v"
    fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), fourcc, args.fps, target_size)
    if not writer.isOpened():
        raise Frame2VideoError(
            f"OpenCV could not open video writer for {args.output}"
        )

    try:
        for info in infos:
            with Image.open(info.path) as image:
                frame = resize_image(image, target_size,
                                     resize_options.mode,
                                     resize_options.pad_color)
            rgb = np.asarray(frame)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
    finally:
        writer.release()


def frame_preview(infos: Sequence[FrameInfo], max_items: int = 12) -> List[str]:
    if len(infos) <= max_items:
        selected = infos
    else:
        head = max_items // 2
        tail = max_items - head
        selected = list(infos[:head]) + list(infos[-tail:])
    preview = []
    for info in selected:
        frame_id = "" if info.frame_index is None else f" [{info.frame_index}]"
        preview.append(f"  {info.path.name}{frame_id}")
    if len(infos) > max_items:
        preview.insert(max_items // 2, "  ...")
    return preview


def print_plan(infos: Sequence[FrameInfo], args: argparse.Namespace,
               resize_options: ResizeOptions) -> None:
    target_size = output_size(infos, resize_options.size)
    duration = len(infos) / args.fps
    print(f"Input dir: {args.input}")
    print(f"Pattern: {args.pattern}")
    print(f"Matched frames: {len(infos)}")
    print(f"First frame: {infos[0].path.name}")
    print(f"Last frame: {infos[-1].path.name}")
    print(f"Resolution: {target_size[0]}x{target_size[1]}")
    print(f"FPS: {format_fps(args.fps)}")
    print(f"Duration: {duration:.2f}s")
    print(f"Output: {args.output}")
    print(f"Backend: {args.backend}")
    print("Frames:")
    print("\n".join(frame_preview(infos)))
    if args.backend == "ffmpeg":
        placeholder = "/tmp/frame2video_xxxxx/frame_%06d.png"
        commands = build_ffmpeg_commands(placeholder, args.output, args,
                                         Path("/tmp/frame2video_xxxxx"))
        print("FFmpeg command:")
        for command in commands:
            print("  " + quote_command(command))


def convert(args: argparse.Namespace) -> None:
    validate_basic_args(args)
    pad_color = ImageColor.getrgb(args.pad_color)
    if len(pad_color) == 4:
        pad_color = pad_color[:3]
    resize_options = ResizeOptions(size=args.resize,
                                   mode=args.resize_mode,
                                   pad_color=pad_color)

    frames = collect_frames(args.input, args.pattern, args.recursive)
    frames = sort_frames(frames, args.sort)
    frames = filter_frames(frames, args.start_frame, args.end_frame,
                           args.stride, args.limit)
    infos = validate_frames(frames, args.resize, args.allow_size_mismatch)
    print_plan(infos, args, resize_options)
    if args.dry_run:
        return

    if args.backend == "ffmpeg":
        write_video_ffmpeg(infos, args, resize_options)
    elif args.backend == "opencv":
        write_video_opencv(infos, args, resize_options)
    else:
        raise Frame2VideoError(f"Unsupported backend: {args.backend}")
    print(f"Wrote video: {args.output}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        convert(args)
    except Frame2VideoError as exc:
        print(f"frame2video: error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("frame2video: interrupted", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
