from __future__ import  absolute_import
import os

import os
import torch
import numpy as np
import time
import yaml
import pickle
from lib.utils.image import Display
import json
import sys, platform, codecs
from pprint import pprint

import lib

from lib import network
from models.RPN import RPN # Hierarchical_Descriptive_Model
from lib.utils.timer import Timer
from lib.utils.metrics import check_recall
from lib.network import np_to_variable

from lib.datasets.visual_genome_loader import visual_genome
from lib.datasets.nia_loader import nia
import argparse
from models.RPN import utils as RPN_utils
from torch.autograd import Variable
from datetime import datetime as dt
from lib.utils.cython_bbox import bbox_overlaps, bbox_intersections


import pdb

import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

class Config:
  dataset = 'nia'
  data_dir = 'data/nia'
  id_object = os.path.join(data_dir, 'objects.json')
  
  path_data_opts = 'options/data.yaml'
  path_rpn_opts = 'options/RPN/rpn.yaml'
  dataset_option = 'small'
  batch_size = 1
  workers = 4
  lr = 0.01
  momentum = 0.9
  dump_name = ''

  load_net = 'output/RPN/rpn_011616_nia_81_best.h5'

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

cfg = Config()

def rpn (**kwargs):
  global opts
  global data_opts
  info = {
    'torch': torch.__version__,
    'python': platform.python_version(),
    'cuda': torch.version.cuda,
    'cudnn': torch.backends.cudnn.version()
  }
  print ('[+] Information: {}'.format(info))

  print ('[+] Loading training set and testing set...')
  with open (cfg.path_data_opts, 'r') as f:
    data_opts = yaml.full_load (f)

  data_opts['dir']   = 'data/nia'

  train_set = nia (data_opts, 'train', dataset_option=cfg.dataset_option, batch_size=cfg.batch_size)
  test_set  = nia (data_opts, 'test',  dataset_option=cfg.dataset_option, batch_size=cfg.batch_size)

  with open (cfg.path_rpn_opts, 'r') as f:
    opts = yaml.full_load (f)
    opts['scale'] = train_set.opts['test']['SCALES'][0]

  opts['anchor_dir']     = 'data/nia'
  opts['kmeans_anchors'] = False

  net = RPN (opts)
  train_set._feat_stride = net._feat_stride
  train_set._rpn_opts    = net.opts

  train_loader = torch.utils.data.DataLoader (train_set, batch_size=cfg.batch_size,
    shuffle=False, num_workers=cfg.workers, pin_memory= True, collate_fn=nia.collate)
  test_loader = torch.utils.data.DataLoader  (test_set, batch_size=cfg.batch_size,
    shuffle=False, num_workers=cfg.workers, pin_memory=True, collate_fn=nia.collate)

  print ('[+] Done. (train: {}, test: {})'.format(len(train_loader), len(test_loader)))

  print ('[+] training from scratch.')
  optimizer = torch.optim.SGD (list (net.parameters ())[26:], lr=cfg.lr, momentum=cfg.momentum, weight_decay=0.0005)

  network.set_trainable (net.features, requires_grad=False)
  net.cuda ()

  data_dir = os.path.join ('output', 'RPN')
  filename = cfg.dump_name + '_' + dt.now().strftime('%m%d%H')
  net.eval ()
  evaluate (test_loader, net, path=os.path.join (data_dir, filename), dataset='test')


def evaluate (loader, net, path, dataset='train'):
  network.load_net(cfg.load_net, net)
  recall, rois = test (loader, net)

  print ('[{}]\tRecall: object: {recall: .3f}%'.format (dataset, recall=recall * 100))
  print ('Done.')

def test (test_loader, target_net):
  id_object = json.load(codecs.open(cfg.id_object, 'r', 'utf-8-sig'))

  box_num = 0
  correct_cnt, total_cnt = 0., 0.
  print ('========== Testing =======')

  results = []

  batch_time = network.AverageMeter ()
  end = time.time ()
  im_counter = 0
  for i, sample in enumerate (test_loader):
    correct_cnt_t, total_cnt_t = 0., 0.
    #print (sample['path'])

    # Forward pass
    # hyhwang list =>  tensor (Variable data has to be a tensor, but got list, only one element tensors can be converted to Python scalars)
    # im_data = Variable (sample['visual'], volatile=True)
    im_data     = sample['visual'][0].cuda ()

    Display (os.path.join(data_opts['dir'], 'images', sample['path'][0]))

    im_counter += im_data.size (0)
    im_info     = sample['image_info']
    gt_objects  = sample['objects']

    print ('image:', sample['path'][0])
    print ('-----------------------------------------')
    for o in gt_objects:
      for d in o:
        print ('{:15s} {}'.format(id_object[str(int(d[4]))], list(map(int, d[:4]))))

    print ('-----------------------------------------')

    object_rois = target_net (im_data, im_info)[1]
    rois = object_rois.cpu().data.numpy()

    results.append (object_rois.cpu ().data.numpy ())
    box_num += object_rois.size (0)

    correct_cnt_t, total_cnt_t = check_recall (object_rois, gt_objects, top_N=50)
    correct_cnt += correct_cnt_t
    total_cnt += total_cnt_t
    batch_time.update (time.time () - end)
    end = time.time ()
    if (i + 1) % 100 == 0 and i > 0:
      print ('[{0}/{6}]  Time: {1:2.3f}s/img).\t[object] Avg: {2:2.2f} Boxes/im, Top-50 recall: {3:2.3f} ({4:.0f}/{5:.0f})'.format (
        i + 1, batch_time.avg, box_num / float (im_counter), correct_cnt / float (total_cnt)* 100, correct_cnt, total_cnt, len (test_loader)))

    break

  recall = correct_cnt / float (total_cnt)

  print ('====== Done Testing ====')

  return recall, results

###################################################

if __name__ == '__main__':
  _t = time.process_time()
  import fire
  fire.Fire()

  _elapsed = time.process_time() - _t
  print ('')
  print ('elapsed: {:.2f} sec'.format(_elapsed))
  print ('')

