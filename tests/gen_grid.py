import sys
import os
import numpy as np

sys.path.append(os.path.abspath("../src"))

from generate_grid import create_directories

Rmax = 100.7
N = 250

create_directories(Rmax, N, template_file="temp.txt")

