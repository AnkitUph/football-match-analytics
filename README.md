# Football Match Analytics

A web application for uploading football match footage and turning it into player and ball tracking data, team and player summaries, visualizations, and downloadable reports. The project combines a Django application with an asynchronous computer-vision pipeline.

> **Project status:** This is a final-year engineering project and its analytics are experimental. Detection, tracking, team classification, event inference, and pitch calibration can make mistakes. Review results before using them as match records.

## Features

### Match upload and processing

- Register and sign in to an account, manage a profile, and maintain team and player information.
- Upload a match video with match details, home and away teams, lineups, and match kit information.
- Detect candidate kit colors from the uploaded footage and associate the selected colors with the home and away sides. Team colors are not required to be assigned in advance in the team profile.
- Process uploaded footage in a background Celery task and show processing progress and status.
- Read basic video metadata such as duration, frame rate, and resolution.

### Computer vision

- Detect players, goalkeepers, referees, and the ball.
- Track detected objects across frames and stitch compatible player tracklets into longer trajectories.
- Classify player tracks by their observed jersey colors and the kit colors selected for the match. The pipeline can retain an unknown classification when the available evidence is not strong enough.
- Track and interpolate short gaps in the ball trajectory.
- Generate an annotated video with detection and tracking overlays, team labels where available, and a ball trail.

### Match review and analytics

- Review match results and compare home and away team summaries.
- Calibrate the camera view against pitch landmarks to map image locations onto a football pitch.
- Associate detected player tracks with lineup entries to support player-level summaries.
- View player heatmaps and available movement and event statistics.
- Add or remove match goals and assign scorers.
- Download a match report as PDF, export match data as JSON or CSV, and export a video clip.

## Processing workflow

1. Create an account and upload a match video with match details and lineups.
2. Inspect the kit-color candidates found in the video and select the colors that correspond to each side.
3. The background pipeline runs object detection and tracking, estimates track-level team labels, follows the ball, and saves processing artifacts.
4. Review the annotated footage and match results. For pitch-based measurements, calibrate the camera using visible pitch landmarks.
5. Identify players by linking detected tracks to the lineup, review or enter goals, and export the results.

## Analytics status and limitations

Object detection, tracking, video metadata, and saved tracking artifacts are produced during video processing. Pitch-based statistics are a separate step and depend on a saved camera calibration. Player identity association also depends on the quality of the tracks and manual review.

Some match and player fields may use placeholder/demo values when real computed values are unavailable, especially before calibration. Event and player attribution are heuristic and can be incomplete or incorrect. Automated jersey-number OCR is not enabled by default. Treat the results as analysis aids, not authoritative match data.

Known tracking limitations and current pipeline notes are documented in [Tracking and Visualization Status](docs/TRACKING_AND_VISUALIZATION_STATUS.md). The broader stack analysis and proposed future work are in [Current Stack Analysis and Failures](docs/CURRENT_STACK_ANALYSIS_AND_FAILURES.md) and [Alternative Approaches and SOTA Roadmap](docs/ALTERNATIVE_APPROACHES_AND_SOTA_ROADMAP.md).

## Technology

- **Web application:** Python 3.11, Django, HTML/CSS/JavaScript
- **Background jobs:** Celery with Redis
- **Database:** MySQL 8.4
- **Video and vision:** OpenCV, PyTorch/Ultralytics YOLO, and the project's tracking and team-color modules
- **Runtime:** Docker Compose

## Run locally with Docker Compose

### Requirements

- Docker Engine and the Docker Compose plugin
- The repository's configured `/dev/dri` device is passed into the web and worker containers for Intel media/compute support. If your host does not provide this device, adjust the Compose device mapping and verify the configured inference backend before running the vision pipeline.

### Setup

1. Copy the example environment file and configure it:

   ```bash
   cp .env.example .env
   ```

   For the database service as currently configured in `docker-compose.yml`, set these values in `.env`:

   ```dotenv
   DB_NAME=football_db
   DB_USER=football_user
   DB_PASSWORD=football_password
   DB_HOST=db
   DB_PORT=3306
   ```

   Replace the development `SECRET_KEY`, set `ALLOWED_HOSTS` for your environment, and configure email settings if you want account email flows. Do not commit `.env` or production secrets.

2. Build and start the web app, database, Redis, and Celery worker:

   ```bash
   docker compose up --build
   ```

3. In another terminal, apply Django database migrations:

   ```bash
   docker compose exec web python manage.py migrate
   ```

4. Open [http://localhost:8000](http://localhost:8000) and register an account. The development server is exposed on port `8000`.

Stop the stack with `Ctrl+C`; use `docker compose down` to stop and remove the containers. The MySQL and Redis data are stored in named Docker volumes.

## Useful routes

| Route | Purpose |
| --- | --- |
| `/` | Dashboard |
| `/accounts/register/` | Create an account |
| `/accounts/login/` | Sign in |
| `/matches/upload/` | Upload a match |
| `/matches/detect-kit-colors/` | Kit-color detection endpoint used by the upload flow |
| `/matches/<match-id>/processing/` | Processing status page |
| `/matches/<match-id>/results/` | Match results |
| `/matches/<match-id>/calibrate/` | Pitch calibration |
| `/matches/<match-id>/identify/` | Assign detected tracks to lineup players |
| `/players/` | Player pages |
| `/teams/` | Team pages |

Match-specific routes also provide team comparison, player heatmaps, goal editing, PDF reports, JSON/CSV exports, and clip export. Match IDs in these routes are UUIDs.

## Repository layout

```text
apps/
  accounts/       Registration, login, profile, and account flows
  analytics/      Analytics data and services
  dashboard/      Main dashboard
  matches/        Match upload, processing tasks, results, and exports
  players/        Player pages and data
  reports/        Report generation
  teams/          Team pages and data
ai_engine/
  stage1_detection/   Object detection
  stage2_tracking/    Object tracking and tracklet stitching
  stage3_team_reid/   Kit-color extraction and team classification
  stage4_ball_tracking/ Ball tracking and interpolation
  stage7_visualization/ Annotated video generation
config/           Django project settings and URL configuration
docs/             Engineering notes and project documentation
media/            Uploaded and sample media (may not be included in every checkout)
```

## Configuration and development

- Django settings are in `config/`; environment variables are loaded from `.env`.
- The default Compose stack starts the Django development server. Do not use the development server or the sample credentials for production.
- Python dependencies are listed in the root `requirements.txt`; vision-engine-specific dependencies are also listed in `ai_engine/requirements.txt`.
- To create an administrator account inside the running web container:

  ```bash
  docker compose exec web python manage.py createsuperuser
  ```
