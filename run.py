from __future__ import  absolute_import
import os

import ipdb
import json
import time
import codecs
import matplotlib
import sys, platform
from tqdm import tqdm
from glob import glob
from os.path import join, basename
from pprint import pprint

from msdn import inference, evaluate

class Config:
  data = 'nia'
  sample = 'samples/IMG_0010366_violin(violin).jpg'

  def _parse(self, kwargs):
    state_dict = self._state_dict()
    for k, v in kwargs.items():
      if k not in state_dict:
        raise ValueError('UnKnown Option: "--%s"' % k)
      setattr(self, k, v)

    print('======user config========')
    pprint(self._state_dict())
    print('==========end============')

  def _state_dict(self):
    return {k: getattr(self, k) for k, _ in Config.__dict__.items() if not k.startswith('_')}

opt = Config()


def torch_info ():
  import torch
  info = {
    'torch': torch.__version__,
    'python': platform.python_version(),
    'cuda': torch.version.cuda,
    'cudnn': torch.backends.cudnn.version()
  }
  return info

def eval (**kwargs):
  opt._parse(kwargs)

  evaluate ()

def inf (**kwargs):
  opt._parse(kwargs)

  inference (opt.sample)

#######################################################

if __name__ == '__main__':
  _t = time.process_time()
  print ('[+] Information: {}'.format(torch_info()))

  import fire
  fire.Fire()

  _elapsed = time.process_time() - _t
  print ('')
  print ('elapsed: {:.2f} sec'.format(_elapsed))

