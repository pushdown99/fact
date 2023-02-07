#!/usr/bin/python

from __future__ import  absolute_import

import os
import sys, platform
from pprint import pprint

def torch_info ():
  import torch
  info = {
    'torch': torch.__version__,
    'python': platform.python_version(),
    'cuda': torch.version.cuda,
    'cudnn': torch.backends.cudnn.version()
  }
  return info

pprint (torch_info ())

