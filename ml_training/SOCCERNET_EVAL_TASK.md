# Task: Benchmark `best.pt` + BoT-SORT on SoccerNet-tracking

Read this whole file first, then do the steps in order. Do not skip the checks.

## 1. Goal

Measure how accurate our current detector and tracker are, using the public SoccerNet-tracking test set and the TrackEval scoring tool. We want numbers (HOTA, DetA, AssA, MOTA, IDF1) that we can compare against numbers SoccerNet published.

This is a measurement task. Do not change the Django app or the existing `ai_engine` pipeline.

## 2. Project facts

- Active project folder: `~/Projects/football-match-analytics` (run everything from the project root).
- Model: `models/best.pt`. It is a YOLO11s with 4 classes: `ball`, `goalkeeper`, `player`, `referee`.
- Tracker: ultralytics BoT-SORT (built into the `ultralytics` package).
- Everything runs on CPU (no CUDA). Expect it to be slow.
- Docker Compose is used. Prefer running Python commands inside the `web` container (`docker compose exec web ...`) because the ML libraries are installed there. The project folder is mounted at `/app`, so files appear in both places.
- Known problem in this project: `opencv-python` and `opencv-python-headless` conflict when installing packages. After any `pip install`, check that `import cv2` still works. If it breaks, run:
  `pip uninstall -y opencv-python && pip install --force-reinstall opencv-python-headless`

## 3. Files involved

| File | Status |
|---|---|
| `ml_training/eval_soccernet_tracking.py` | Already written. Runs the model + tracker on SoccerNet clips and writes MOT-format result files. |
| `ml_training/TrackEval/` | You will clone this (scoring tool). |
| `datasets/SoccerNet/` | You will download data here. |
| `runs_eval/` | Results go here. |

If `ml_training/eval_soccernet_tracking.py` is missing, stop and tell the user to copy it there. Do not rewrite it from scratch.

## 4. Steps

### Step 1: Check the environment

Run these and write down the results:

```bash
df -h .                      # free disk space
ls -la models/best.pt        # model exists
docker compose ps            # containers are up
docker compose exec web python -c "import ultralytics, cv2, numpy; print(ultralytics.__version__, cv2.__version__, numpy.__version__)"
```

- If `ultralytics` is missing in the container, tell the user. Do not install big packages without telling them.
- Check free disk space before downloading. If it is under about 20 GB, tell the user before continuing.

### Step 2: Keep data out of git and Docker builds

Add these lines to `.gitignore` (and `.dockerignore` if it exists), only if they are not already there:

```
datasets/
runs_eval/
ml_training/TrackEval/
```

### Step 3: Download the SoccerNet test split

Only the `test` split. It has the ground truth and is the one we need.

```bash
docker compose exec web pip install SoccerNet
docker compose exec web python -c "
from SoccerNet.Downloader import SoccerNetDownloader
SoccerNetDownloader(LocalDirectory='datasets/SoccerNet').downloadDataTask(task='tracking', split=['test'])
"
cd datasets/SoccerNet/tracking && unzip -q test.zip
```

After `pip install SoccerNet`, check `import cv2` still works (see the OpenCV note in section 2).

Check the result:

```bash
ls datasets/SoccerNet/tracking/test | head
ls datasets/SoccerNet/tracking/test/$(ls datasets/SoccerNet/tracking/test | head -1)
```

Each sequence folder (named like `SNMOT-xxx`) should contain `img1/`, `gt/gt.txt` and `seqinfo.ini`. If the folders are one level deeper or named differently, find the folder that directly contains the `SNMOT-*` folders. That folder is the `--split-dir` for later steps. If the download fails, report the exact error and stop.

### Step 4: Get TrackEval

```bash
git clone https://github.com/JonathonLuiten/TrackEval ml_training/TrackEval
docker compose exec web pip install scipy   # only if missing
```

TrackEval uses old NumPy names (`np.float`, `np.int`, `np.bool`) that fail on new NumPy. Fix them only inside the TrackEval clone:

```bash
grep -rlE "np\.(float|int|bool)\b" ml_training/TrackEval --include=*.py \
  | xargs -r sed -i -E 's/np\.(float|int|bool)\b/\1/g'
```

Note that `np.float32`, `np.int32` and similar are not changed by this command. That is correct.

### Step 5: Smoke test on 3 clips

```bash
docker compose exec web python ml_training/eval_soccernet_tracking.py \
  --model models/best.pt \
  --split-dir datasets/SoccerNet/tracking/test \
  --name smoke_640 --limit 3 \
  --trackeval-dir ml_training/TrackEval
```

What to check:

1. The script prints speed in fps for each clip. Write it down. Use it to estimate the time for the full split (about 49 clips of about 750 frames each).
2. `runs_eval/smoke_640/data/` has 3 `.txt` files. Look at the first few lines of one. Each line must look like `frame,id,x,y,w,h,conf,-1,-1,-1`.
3. TrackEval prints a table, and `runs_eval/smoke_640/pedestrian_summary.txt` exists.

If something fails:

- Fix small, obvious problems (for example a wrong path or an ultralytics version difference in `eval_soccernet_tracking.py`). Keep the edit as small as possible and show the user the exact change.
- If TrackEval rejects the benchmark or folder layout, read the error and the TrackEval script `scripts/run_mot_challenge.py`, then adjust the command. Do not change TrackEval's metric code.
- If the same step fails twice, stop and report the full error. Do not keep guessing.

### Step 6: Experiments on the same 3 clips

Use the same `--limit 3` so the numbers can be compared. Only change one thing per run, and use a new `--name` each time.

| Name | Extra flags | Question it answers |
|---|---|---|
| `smoke_640` | none | Baseline |
| `size_1280` | `--imgsz 1280` | Does a bigger input size help? |
| `noball_640` | `--no-ball` | How much does the ball hurt detection accuracy? |
| `nogmc_640` | `--gmc-method none` | Does camera motion compensation help? |

Reuse `smoke_640` for the baseline (do not run it again). Run each with `--trackeval-dir ml_training/TrackEval`.

### Step 7: Report

Read each `runs_eval/<name>/pedestrian_summary.txt` and give the user one table:

| Run | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | fps |
|---|---|---|---|---|---|---|---|

Then add:

- **Reference numbers** (from the SoccerNet-Tracking paper, test set, no ground-truth boxes): ByteTrack about 47.2 HOTA (DetA 44.5, AssA 50.3), DeepSORT about 36.7 HOTA. Say clearly that our run is only on 3 clips, so the comparison is rough.
- **Reading the result:** low DetA means the detector is the weak part (retrain with a bigger model or larger image size). Low AssA with good DetA means the tracker is the weak part (ID switches).
- **Time estimate** for a full test-split run, based on the measured fps.

## 5. Rules

- Do not run the full test split until the user says yes. Ask first and give the time estimate.
- Do not edit the Django app, `ai_engine/`, `config/`, migrations, or the database.
- Do not delete anything. Do not commit or push to git.
- Do not install large packages (PyTorch, CUDA stacks) without asking.
- Only change `eval_soccernet_tracking.py` to fix a real error, and show the user the change.
- Use simple language in your reports. Include the exact commands you ran and any errors you saw.

## 6. Done when

- The smoke test and the 3 experiments have run, with result files in `runs_eval/`.
- The user has the comparison table and a short plain-language summary of what it means.
- The user knows how long a full-split run would take and has been asked whether to start it.
