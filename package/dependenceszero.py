from __future__ import print_function
import os
import datetime
import numpy as np
import random
import string
import tensorflow as tf
from six.moves import range
import datetime
import sys


state_dict = ['a','c','e','h']        ##   <<<<<<<<---------------
n_states = len(state_dict)
r = 6
vocabulary_size = n_states
