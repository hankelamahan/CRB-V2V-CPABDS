import tempfile
import unittest
from pathlib import Path

from PIL import Image

from frame2video.frame2video import (
    Frame2VideoError,
    ResizeOptions,
    collect_frames,
    extract_frame_index,
    filter_frames,
    prepare_temp_sequence,
    sort_frames,
    validate_frames,
)


class Frame2VideoTest(unittest.TestCase):
    def test_extract_frame_index(self):
        self.assertEqual(
            extract_frame_index(Path("late_constant_frame_00190.png")),
            190,
        )
        self.assertEqual(
            extract_frame_index(Path("intermediate_intensity_frame_00001.jpg")),
            1,
        )
        self.assertIsNone(extract_frame_index(Path("scenario_1_167.png")))

    def test_sort_frames_prefers_frame_id(self):
        paths = [
            Path("late_constant_frame_00100.png"),
            Path("late_constant_frame_00099.png"),
            Path("late_constant_frame_00005.png"),
            Path("scenario_1_167.png"),
        ]
        self.assertEqual(
            [p.name for p in sort_frames(paths)],
            [
                "late_constant_frame_00005.png",
                "late_constant_frame_00099.png",
                "late_constant_frame_00100.png",
                "scenario_1_167.png",
            ],
        )

    def test_filter_frames_window_stride_limit(self):
        paths = [
            Path(f"late_constant_frame_{idx:05d}.png")
            for idx in range(100, 110)
        ]
        filtered = filter_frames(paths, 102, 108, stride=2, limit=3)
        self.assertEqual(
            [p.name for p in filtered],
            [
                "late_constant_frame_00102.png",
                "late_constant_frame_00104.png",
                "late_constant_frame_00106.png",
            ],
        )

    def test_collect_frames_default_pattern_excludes_non_frame_png(self):
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            self._write_image(root / "late_constant_frame_00001.png")
            self._write_image(root / "scenario_1_167.png")
            frames = collect_frames(root, "*_frame_*.png")
            self.assertEqual([p.name for p in frames],
                             ["late_constant_frame_00001.png"])

    def test_validate_frames_rejects_size_mismatch_without_resize(self):
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            first = root / "late_constant_frame_00001.png"
            second = root / "late_constant_frame_00002.png"
            self._write_image(first, size=(32, 24))
            self._write_image(second, size=(16, 24))
            with self.assertRaises(Frame2VideoError):
                validate_frames([first, second], resize_size=None,
                                allow_size_mismatch=False)

    def test_prepare_temp_sequence_renumbers_sparse_frames(self):
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            first = root / "late_constant_frame_00099.png"
            second = root / "late_constant_frame_00105.png"
            self._write_image(first, size=(32, 24), color=(255, 0, 0))
            self._write_image(second, size=(32, 24), color=(0, 255, 0))
            infos = validate_frames([first, second], resize_size=None)
            with tempfile.TemporaryDirectory() as seq_name:
                pattern = prepare_temp_sequence(
                    infos,
                    Path(seq_name),
                    ResizeOptions(size=None,
                                  mode="fit",
                                  pad_color=(0, 0, 0)),
                )
                self.assertTrue(Path(seq_name, "frame_000000.png").exists())
                self.assertTrue(Path(seq_name, "frame_000001.png").exists())
                self.assertTrue(pattern.endswith("frame_%06d.png"))

    @staticmethod
    def _write_image(path, size=(32, 24), color=(10, 20, 30)):
        Image.new("RGB", size, color).save(path)


if __name__ == "__main__":
    unittest.main()
