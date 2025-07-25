from . import metadata
import os

BASE_DIR = os.path.dirname(__file__)
log_folder = os.path.join(BASE_DIR, "logs")

__all__ = ['metadata', 'log_folder']