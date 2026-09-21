"""
32-keypoint pitch reference for smart-assist homography calibration,
matching the output layout of the Roboflow `football-field-detection-f07vi/14`
keypoint model (sports.configs.soccer.SoccerPitchConfiguration.vertices).

IMPORTANT — do not copy Roboflow's own numeric constants (their template
pitch is a stylized 120m x 70m pitch with internally inconsistent
measurements, e.g. a ~20.15m penalty box depth that doesn't match FIFA's
real 16.5m). Instead, this file reuses Roboflow's VERTEX FORMULA STRUCTURE
(which vertex is expressed as which offset of penalty-box/goal-box/circle
dimensions) but substitutes this project's own FIFA-based constants and
real 105m x 68m pitch size — the same constants already used by
pitch_landmarks.py and ai_engine/heatmap.py's PITCH_LENGTH_M/PITCH_WIDTH_M.

Coordinate convention: SAME as pitch_landmarks.py — origin at the center
circle, x along the length (positive = "right"/home goal side), y across
the width (positive = top touchline). All 13 points in pitch_landmarks.py
that correspond to one of these 32 keypoints were cross-checked
numerically against this file's output before it was written; they match
exactly.

IMPORTANT — roboflow_index = class_id + 1. Confirmed via live inference
AND a homography reprojection test (halfway line, center circle, and
touchlines all landed correctly on real pitch markings in test_11.mp4)
that this model's array position (class_id, 0-indexed) is the correct
index into config.vertices / this list — matching the documented usage
pattern in sports.common.view.ViewTransformer examples. The "class"
string field (e.g. "14") is a display label in Roboflow's dataset
metadata that has drifted out of sync with the real class_id ordering
and must NOT be used for mapping — using it silently produces a badly
wrong homography (a tiny, misplaced center circle; a halfway line
running diagonally across the frame instead of down the pitch). This was
caught and fixed during calibration validation — see homography_check.py.
"""

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
PENALTY_BOX_LENGTH_M = 16.5
PENALTY_BOX_WIDTH_M = 40.32
GOAL_BOX_LENGTH_M = 5.5
GOAL_BOX_WIDTH_M = 18.32
CENTRE_CIRCLE_RADIUS_M = 9.15
PENALTY_SPOT_DISTANCE_M = 11.0

ROBOFLOW_KEYPOINTS_32 = [
    {"roboflow_index": 1,  "id": "left_corner_top",           "label": "Left corner flag - top",                     "x": -52.5, "y": 34.0},
    {"roboflow_index": 2,  "id": "left_goal_line_pbox_top",    "label": "Left goal line - penalty box width mark (top)",    "x": -52.5, "y": 20.16},
    {"roboflow_index": 3,  "id": "left_goal_line_gbox_top",    "label": "Left goal line - goal box width mark (top)",       "x": -52.5, "y": 9.16},
    {"roboflow_index": 4,  "id": "left_goal_line_gbox_bottom", "label": "Left goal line - goal box width mark (bottom)",    "x": -52.5, "y": -9.16},
    {"roboflow_index": 5,  "id": "left_goal_line_pbox_bottom", "label": "Left goal line - penalty box width mark (bottom)", "x": -52.5, "y": -20.16},
    {"roboflow_index": 6,  "id": "left_corner_bottom",         "label": "Left corner flag - bottom",                  "x": -52.5, "y": -34.0},
    {"roboflow_index": 7,  "id": "left_goal_box_top",          "label": "Left goal box - top corner",                 "x": -47.0, "y": 9.16},
    {"roboflow_index": 8,  "id": "left_goal_box_bottom",       "label": "Left goal box - bottom corner",              "x": -47.0, "y": -9.16},
    {"roboflow_index": 9,  "id": "left_penalty_spot",          "label": "Left penalty spot",                          "x": -41.5, "y": 0.0},
    {"roboflow_index": 10, "id": "left_box_top",               "label": "Left penalty box - top corner",              "x": -36.0, "y": 20.16},
    {"roboflow_index": 11, "id": "left_box_line_gbox_top",     "label": "Left penalty box line - goal box width mark (top)",    "x": -36.0,"y": 9.16},
    {"roboflow_index": 12, "id": "left_box_line_gbox_bottom",  "label": "Left penalty box line - goal box width mark (bottom)", "x": -36.0,"y": -9.16},
    {"roboflow_index": 13, "id": "left_box_bottom",            "label": "Left penalty box - bottom corner",           "x": -36.0, "y": -20.16},
    {"roboflow_index": 14, "id": "halfway_top",                "label": "Halfway line - top touchline",               "x": 0.0,   "y": 34.0},
    {"roboflow_index": 15, "id": "halfway_circle_top",         "label": "Halfway line - center circle edge (top)",    "x": 0.0,   "y": 9.15},
    {"roboflow_index": 16, "id": "halfway_circle_bottom",      "label": "Halfway line - center circle edge (bottom)", "x": 0.0,   "y": -9.15},
    {"roboflow_index": 17, "id": "halfway_bottom",             "label": "Halfway line - bottom touchline",            "x": 0.0,   "y": -34.0},
    {"roboflow_index": 18, "id": "right_box_top",              "label": "Right penalty box - top corner",             "x": 36.0,  "y": 20.16},
    {"roboflow_index": 19, "id": "right_box_line_gbox_top",    "label": "Right penalty box line - goal box width mark (top)",    "x": 36.0, "y": 9.16},
    {"roboflow_index": 20, "id": "right_box_line_gbox_bottom", "label": "Right penalty box line - goal box width mark (bottom)", "x": 36.0, "y": -9.16},
    {"roboflow_index": 21, "id": "right_box_bottom",           "label": "Right penalty box - bottom corner",          "x": 36.0,  "y": -20.16},
    {"roboflow_index": 22, "id": "right_penalty_spot",         "label": "Right penalty spot",                         "x": 41.5,  "y": 0.0},
    {"roboflow_index": 23, "id": "right_goal_box_top",         "label": "Right goal box - top corner",                "x": 47.0,  "y": 9.16},
    {"roboflow_index": 24, "id": "right_goal_box_bottom",      "label": "Right goal box - bottom corner",             "x": 47.0,  "y": -9.16},
    {"roboflow_index": 25, "id": "right_corner_top",           "label": "Right corner flag - top",                    "x": 52.5,  "y": 34.0},
    {"roboflow_index": 26, "id": "right_goal_line_pbox_top",   "label": "Right goal line - penalty box width mark (top)",    "x": 52.5,  "y": 20.16},
    {"roboflow_index": 27, "id": "right_goal_line_gbox_top",   "label": "Right goal line - goal box width mark (top)",       "x": 52.5,  "y": 9.16},
    {"roboflow_index": 28, "id": "right_goal_line_gbox_bottom","label": "Right goal line - goal box width mark (bottom)",    "x": 52.5,  "y": -9.16},
    {"roboflow_index": 29, "id": "right_goal_line_pbox_bottom","label": "Right goal line - penalty box width mark (bottom)", "x": 52.5,  "y": -20.16},
    {"roboflow_index": 30, "id": "right_corner_bottom",        "label": "Right corner flag - bottom",                 "x": 52.5,  "y": -34.0},
    {"roboflow_index": 31, "id": "center_circle_left",         "label": "Center circle - left edge on halfway line",  "x": -9.15, "y": 0.0},
    {"roboflow_index": 32, "id": "center_circle_right",        "label": "Center circle - right edge on halfway line", "x": 9.15,  "y": 0.0},
]

ROBOFLOW_KEYPOINTS_32_BY_INDEX = {
    entry["roboflow_index"]: entry for entry in ROBOFLOW_KEYPOINTS_32
}
