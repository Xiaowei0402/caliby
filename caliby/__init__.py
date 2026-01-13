import os

# Set required environment variables for atomworks if they are not already set.
# These are required by atomworks' parser but not used for our in-memory implementation.
if "PDB_MIRROR_PATH" not in os.environ:
    os.environ["PDB_MIRROR_PATH"] = ""
if "CCD_MIRROR_PATH" not in os.environ:
    os.environ["CCD_MIRROR_PATH"] = ""

from .Caliby import Caliby

__all__ = ["Caliby"]
