"""
Fixed preset of real-world pitch reference points a user picks from when
calibrating a match, instead of typing meter coordinates by hand (which
is an easy way to silently produce a bad calibration).

Coordinate convention: standard 105m x 68m pitch, origin at the center
circle, x running along the length (positive toward the "right"/home
goal as drawn on the calibration canvas), y across the width. This is
the SAME convention already used by:
  - the original hardcoded calibration in tasks.py (its penalty-box
    corners were exactly (36, +/-20.16), which match right_box_top/
    right_box_bottom below)
  - ai_engine/heatmap.py's PITCH_LENGTH_M/PITCH_WIDTH_M

If this list is ever extended, keep every "y" sign consistent with the
existing points (top touchline = positive y) — mixing sign conventions
here would silently corrupt every future calibration, not just new ones.
"""

PITCH_LANDMARKS = [
    {"id": "center_spot", "label": "Center spot", "x": 0.0, "y": 0.0},
    {"id": "halfway_top", "label": "Halfway line \u2013 top touchline", "x": 0.0, "y": 34.0},
    {"id": "halfway_bottom", "label": "Halfway line \u2013 bottom touchline", "x": 0.0, "y": -34.0},
    {"id": "right_corner_top", "label": "Right corner flag \u2013 top", "x": 52.5, "y": 34.0},
    {"id": "right_corner_bottom", "label": "Right corner flag \u2013 bottom", "x": 52.5, "y": -34.0},
    {"id": "left_corner_top", "label": "Left corner flag \u2013 top", "x": -52.5, "y": 34.0},
    {"id": "left_corner_bottom", "label": "Left corner flag \u2013 bottom", "x": -52.5, "y": -34.0},
    {"id": "right_box_top", "label": "Right penalty box \u2013 top corner", "x": 36.0, "y": 20.16},
    {"id": "right_box_bottom", "label": "Right penalty box \u2013 bottom corner", "x": 36.0, "y": -20.16},
    {"id": "left_box_top", "label": "Left penalty box \u2013 top corner", "x": -36.0, "y": 20.16},
    {"id": "left_box_bottom", "label": "Left penalty box \u2013 bottom corner", "x": -36.0, "y": -20.16},
    {"id": "right_penalty_spot", "label": "Right penalty spot", "x": 41.5, "y": 0.0},
    {"id": "left_penalty_spot", "label": "Left penalty spot", "x": -41.5, "y": 0.0},
]

PITCH_LANDMARKS_BY_ID = {landmark["id"]: landmark for landmark in PITCH_LANDMARKS}