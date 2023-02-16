import torch
import os
import yaml
import random
from pprint import pprint


from lib import network
import lib.datasets as datasets
import lib.utils.general_utils as utils
import lib.utils.logger as logger

from lib.utils.image import Display
from lib.utils.FN_utils import get_model_name, group_features, get_optimizer

import models
from models.HDN_v2.utils import save_checkpoint, load_checkpoint, save_results, save_detections

from models.modules.dataParallel import DataParallel

def make_meters ():
  meters_dict = {
    'loss': logger.AvgMeter (),
    'loss_rpn': logger.AvgMeter (),
    'loss_cls_obj': logger.AvgMeter (),
    'loss_reg_obj': logger.AvgMeter (),
    'loss_cls_rel': logger.AvgMeter (),
    'loss_cls_cap': logger.AvgMeter (),
    'loss_reg_cap': logger.AvgMeter (),
    'loss_cls_objectiveness': logger.AvgMeter (),
    'batch_time': logger.AvgMeter (),
    'data_time': logger.AvgMeter (),
    'epoch_time': logger.SumMeter (),
    'best_recall': logger.AvgMeter (),
  }
  return meters_dict


def init (pretrained_model = None):
  # Set options
  options = {
    'logs': {
      'model_name':      'msdn',
      'dir_logs':        '',
      'operation':       '',
    },
    'data':{
      'dataset_option':  'normal',
      'batch_size':      torch.cuda.device_count (),
    },
    'optim': {
      'lr':              0.01,
      'epochs':          30,
      'lr_decay_epoch':  2,
      'optimizer':       0,
      'clip_gradient':   True,
    },
    'model':{
      'MPS_iter':        1,
      'dropout':         0,
      'use_loss_weight': True,
    },
  }

  with open ('options/models/msdn.yaml', 'r') as handle:
    options_yaml = yaml.full_load (handle)

  options_yaml['dataset'] = 'nia'
  options = utils.update_values (options, options_yaml)

  with open (options['data']['opts'], 'r') as f:
    data_opts = yaml.full_load (f)

  data_opts['anchor_dir']     = 'data/nia'
  data_opts['kmeans_anchors'] = False

  options['data']['dataset_version'] = data_opts.get ('dataset_version', None)
  options['opts'] = data_opts

  lr = options['optim']['lr']

  options = get_model_name (options)
  pprint(options)
  print ()

  print ('[+] checkpoints are saved to: {}'.format (options['logs']['dir_logs']))

  # To set the random seed
  seed = 1
  random.seed (seed)
  torch.manual_seed (seed + 1)
  torch.cuda.manual_seed (seed + 2)

  print ('[+] loading training set and testing set...')
  train_set = getattr (datasets, options['data']['dataset'])(data_opts, 'train',
              dataset_option = options['data'].get ('dataset_option', None), use_region = options['data'].get ('use_region', False),)
  test_set  = getattr (datasets, options['data']['dataset']) (data_opts, 'test',
              dataset_option = options['data'].get ('dataset_option', None), use_region = options['data'].get ('use_region', False))
  print ('[+] done.')

  # Model declaration
  model = getattr (models, options['model']['arch']) (train_set, opts = options['model'])

  # Pass enough message for anchor target generation
  train_set._feat_stride = model.rpn._feat_stride
  train_set._rpn_opts    = model.rpn.opts
  print ('[+] done.')

  # (Experiment) Enable async data loading
  train_loader = torch.utils.data.DataLoader (train_set, batch_size=options['data']['batch_size'],
              shuffle = True, num_workers = 4, #pin_memory = True, # hyhwang
              collate_fn = getattr (datasets, options['data']['dataset']).collate, drop_last = True,)

  test_loader = torch.utils.data.DataLoader (test_set, batch_size=1,
              shuffle = False, num_workers = 4, pin_memory = True,
              collate_fn = getattr (datasets, options['data']['dataset']).collate)

  # To group up the features
  vgg_features_fix, vgg_features_var, rpn_features, hdn_features, mps_features = group_features (model, has_RPN=True)

  network.set_trainable (model, False)
  exp_logger = None

  #########################################################################
  if pretrained_model is not None:
    print ('Loading pretrained model: {}'.format (pretrained_model))
    train_all = True
    network.load_net (pretrained_model, model)
    optimizer = get_optimizer (lr, 2, options, vgg_features_var, rpn_features, hdn_features, mps_features)

  scheduler = torch.optim.lr_scheduler.StepLR (optimizer, step_size=options['optim']['lr_decay_epoch'], gamma=options['optim']['lr_decay'])

  # Setting the state of the training model
  # hyhwang DataParallel => None
  model = DataParallel (model)
  model.cuda ()
  model.train ()

  for param in model.parameters():
    param.grad = None

  # Set loggers
  if exp_logger is None:
    exp_name = os.path.basename (options['logs']['dir_logs']) # add timestamp
    exp_logger = logger.Experiment (exp_name, options)
    exp_logger.add_meters ('train', make_meters ())
    exp_logger.add_meters ('test', make_meters ())
    exp_logger.info['model_params'] = utils.params_count (model)
    print ('Model has {} parameters'.format (exp_logger.info['model_params']))


  #  network.weights_normal_init (net, dev=0.01)

  return model, train_set, test_set, train_loader, test_loader, options


def evaluate (pretrained_model='pretrained/best_model.h5'):
  top_Ns = [50, 100]

  model, train_set, test_set, train_loader, test_loader, options = init (pretrained_model)
  recall, result = model.module.engines.test (test_loader, model, top_Ns, nms=-1., triplet_nms=0.4, use_gt_boxes=False) # True?
  print ('======= Testing Result =======')
  for idx, top_N in enumerate (top_Ns):
    print ('Top-%d Recall\t[PredCls]: %2.3f%%\t[PhrCls]: %2.3f%%\t[SGCls]: %2.3f%%' % (
      top_N, float(recall[2][idx]) * 100, float(recall[1][idx]) * 100, float(recall[0][idx]) * 100))

  print ('============ Done ============')
  save_results (result, None, options['logs']['dir_logs'], is_testing=True)

def inference (path, pretrained_model='pretrained/best_model.h5'):
  model, train_set, test_set, train_loader, test_loader, options = init (pretrained_model)
  sample = test_set.getsample(path)

  obj_boxes, obj_scores, obj_cls, subject_inds, object_inds, subject_boxes, object_boxes, predicate_inds, sub_assignment, obj_assignment, total_score = \
    model.module.engines.inference (model, sample)

  print ('============ Done ============')
  print (subject_inds[0], object_inds[0], predicate_inds[0], total_score[0])
  sub  = subject_inds[0]
  obj  = object_inds[0]
  pred = predicate_inds[0]
  print (test_set._object_classes[sub], test_set._predicate_classes[pred], test_set._object_classes[obj])
  Display(path)


def main ():
  #inference ('samples/IMG_0010366_violin(violin).jpg')
  evaluate ()

if __name__ == '__main__':
    main ()

