import cv2


def read_video(video_path):
    """
    Reads an entire video into memory as a list of frames.

    Note: only suitable for short clips (matches your stated ~15 min max)
    - for longer video this would be a large memory footprint. If you
    ever need to support longer videos, switch to frame-by-frame
    streaming processing instead of loading everything upfront.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    cap.release()
    return fps


def save_video(output_video_frames, output_video_path, fps=25):
    """
    fps should be the SOURCE video's real fps (see get_video_fps) -
    hardcoding a fixed value here would desync the output from the
    original video's actual playback speed.

    Uses mp4v/.mp4 rather than XVID/.avi - .avi isn't reliably playable
    in browsers, and this output is meant to be served via Django's
    MatchVideo.annotated_video field and played in an HTML5 <video> tag.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    height, width = output_video_frames[0].shape[:2]
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    for frame in output_video_frames:
        out.write(frame)
    out.release()