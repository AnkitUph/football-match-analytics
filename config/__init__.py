"""
Add these two lines to config/__init__.py (create the file if it doesn't
exist, or just append if it does — don't remove anything already there).

This makes sure Celery's app is loaded whenever Django starts, so
@shared_task decorators in every app's tasks.py connect to it correctly.
"""

from .celery import app as celery_app

__all__ = ("celery_app",)