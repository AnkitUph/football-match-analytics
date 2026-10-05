import json
from pathlib import Path

# Adjust these paths if your HF cache location differs
HF_CACHE = Path('/home/ankit/.cache/huggingface')
SRC_ROOT = HF_CACHE / 'datasets' / 'soccerNet'  # typical HF cache path for SoccerNet
DST_ROOT = Path('/home/ankit/Projects/football-match-analytics/data/fine_tune_annotations')
DST_ROOT.mkdir(parents=True, exist_ok=True)


def convert_one(video_id: str):
    # video_id example: "england_epl/2014-2015/2015-02-21 - 18-00"
    label_path = SRC_ROOT / video_id / 'Labels-v2.json'
    if not label_path.exists():
        print(f'⚠️  Missing label file for {video_id}')
        return
    out_path = DST_ROOT / (video_id.replace('/', '_') + '.json')
    with open(label_path) as fp:
        data = json.load(fp)
    events = []
    # Assume 25 fps (as used in the original code)
    for ev in data.get('labels', []):
        frame = int(ev['timestamp'] * 25)
        events.append({
            'frame': frame,
            'label': ev['label'].upper().replace(' ', '_'),
            'score': 1.0
        })
    out = {'video': video_id, 'events': events}
    out_path.write_text(json.dumps(out, indent=2))
    print(f'✔️  {out_path.name} → {len(events)} events')


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: convert_soccernet_labels.py <video_id1> [<video_id2> ...]')
        sys.exit(1)
    for vid in sys.argv[1:]:
        convert_one(vid)
