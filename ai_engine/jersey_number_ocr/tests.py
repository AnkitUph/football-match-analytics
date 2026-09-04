"""Tests for jersey-number OCR + temporal voting.

These run without a Tesseract installation by injecting a fake ``ocr_func``,
so the voting/association logic is exercised in CI and on any dev machine.
"""
from unittest import mock

from django.test import SimpleTestCase
import numpy as np

from ai_engine.jersey_number_ocr import JerseyNumberReader
from ai_engine.jersey_number_ocr.jersey_number_reader import MIN_OCR_READINGS


def make_frames(count, height=120, width=80):
    return [np.zeros((height, width, 3), dtype=np.uint8) for _ in range(count)]


def make_tracks(count, bbox=(10, 10, 60, 110)):
    players = [{1: {"bbox": list(bbox)}} for _ in range(count)]
    return {"players": players, "goalkeepers": [], "referees": [], "ball": []}


class TemporalVotingTests(SimpleTestCase):
    def _reader(self, readings_sequence):
        """A reader whose OCR returns the supplied readings in order.

        Each sampled frame produces two OCR calls (normal + inverted polarity),
        so the sequence length should be 2 * frames_sampled.
        """
        iterator = iter(readings_sequence)

        def ocr_func(image):
            try:
                return [next(iterator)]
            except StopIteration:
                return []

        return JerseyNumberReader(sample_every=1, ocr_func=ocr_func)

    def test_majority_number_wins(self):
        frames = make_frames(5)
        tracks = make_tracks(5)
        # 8 clean readings of 10, 2 noisy readings of 7.
        reader = self._reader([(10, 80.0)] * 8 + [(7, 90.0)] * 2)
        reader.add_jersey_numbers_to_tracks(frames, tracks)
        self.assertEqual(reader.track_jersey_numbers, {1: 10})

    def test_insufficient_evidence_rejected(self):
        frames = make_frames(1)
        tracks = make_tracks(1)
        # Only 2 readings in total, below MIN_OCR_READINGS.
        reader = self._reader([(10, 80.0)] * 2)
        self.assertLess(2, MIN_OCR_READINGS)
        reader.add_jersey_numbers_to_tracks(frames, tracks)
        self.assertEqual(reader.track_jersey_numbers, {})

    def test_split_vote_rejected(self):
        frames = make_frames(6)
        tracks = make_tracks(6)
        reader = self._reader([(10, 80.0)] * 6 + [(17, 80.0)] * 6)
        reader.add_jersey_numbers_to_tracks(frames, tracks)
        self.assertEqual(reader.track_jersey_numbers, {})

    def test_low_confidence_readings_excluded_from_vote(self):
        frames = make_frames(6)
        tracks = make_tracks(6)
        # 10 is read confidently; 7 is always low-confidence and must not vote.
        reader = self._reader([(10, 85.0)] * 8 + [(7, 10.0)] * 4)
        reader.add_jersey_numbers_to_tracks(frames, tracks)
        self.assertEqual(reader.track_jersey_numbers, {1: 10})

    def test_writes_jersey_number_into_tracks(self):
        frames = make_frames(5)
        tracks = make_tracks(5)
        reader = self._reader([(10, 80.0)] * 10)
        reader.add_jersey_numbers_to_tracks(frames, tracks)
        for frame_tracks in tracks["players"]:
            self.assertEqual(frame_tracks[1]["jersey_number"], 10)

    def test_records_vote_share_confidence(self):
        frames = make_frames(5)
        tracks = make_tracks(5)
        # 8 confident votes of 10, 2 of 7 => 10 wins with ~0.78 share.
        reader = self._reader([(10, 80.0)] * 8 + [(7, 90.0)] * 2)
        reader.add_jersey_numbers_to_tracks(frames, tracks)
        self.assertEqual(reader.track_jersey_numbers, {1: 10})
        self.assertGreater(reader.track_jersey_confidence[1], 0.6)
        self.assertLessEqual(reader.track_jersey_confidence[1], 1.0)
        for frame_tracks in tracks["players"]:
            self.assertEqual(frame_tracks[1]["jersey_confidence"],
                             reader.track_jersey_confidence[1])

    def test_unavailable_reader_is_noop(self):
        frames = make_frames(5)
        tracks = make_tracks(5)
        with mock.patch(
            "ai_engine.jersey_number_ocr.jersey_number_reader._tesseract_available",
            return_value=False,
        ):
            reader = JerseyNumberReader()
        self.assertFalse(reader.available)
        reader.add_jersey_numbers_to_tracks(frames, tracks)
        self.assertEqual(reader.track_jersey_numbers, {})
        for frame_tracks in tracks["players"]:
            self.assertNotIn("jersey_number", frame_tracks[1])

    def test_low_confidence_readings_do_not_count_as_evidence(self):
        frames = make_frames(4)
        tracks = make_tracks(4)
        # Only 2 confident readings (>= MIN_OCR_CONFIDENCE); the rest are
        # low-confidence and must NOT count toward MIN_OCR_READINGS.
        reader = self._reader([(10, 80.0)] * 2 + [(7, 10.0)] * 6)
        reader.add_jersey_numbers_to_tracks(frames, tracks)
        self.assertEqual(reader.track_reading_count[1], 2)
        self.assertEqual(reader.track_jersey_numbers, {})

    def test_crop_rejects_tiny_boxes(self):
        self.assertIsNone(JerseyNumberReader.crop_jersey(make_frames(1)[0], [0, 0, 30, 40]))
