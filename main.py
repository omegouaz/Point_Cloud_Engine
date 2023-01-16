'''
    This is the main script
    Reconstruction of the mesh and heightmap from RGB+D RealSense.
'''

import sys
import os
import copy
import json
import time
import datetime
import math
import cv2
import open3d as o3d
import numpy as np
import scipy.linalg
import scipy.ndimage
import scipy.ndimage.filters as filters

from tqdm import tqdm

from engine import (
    transform,
    make_fragments,
    register_fragments,
    refine,
    integration,
    initialize,
)
from engine.utility import check_folder_structure