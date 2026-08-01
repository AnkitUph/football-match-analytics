import os

from celery import Celery

# Same settings module your manage.py / wsgi.py already point to.
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

app = Celery("config")

# Reads any CELERY_* setting from settings.py (see config_settings_ADDITIONS.py)
app.config_from_object("django.conf:settings", namespace="CELERY")

# Auto-finds tasks.py in every app listed in INSTALLED_APPS
# (apps.matches.tasks.process_match will be picked up automatically)
app.autodiscover_tasks()