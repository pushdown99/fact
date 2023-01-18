import os
import torch
import numpy as np
import time
import yaml
import pickle
from lib.utils.image import Display
import json
import sys, platform
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

import pdb

parser = argparse.ArgumentParser ('Options for training RPN in pytorch')

####################################################################################################################
 
# training settings
parser.add_argument ('--path_data_opts', type=str, default='options/data.yaml', help='Use options for ' )
parser.add_argument ('--lr', type=float, default=0.01, help='To disable the Lanuage Model ')
parser.add_argument ('--max_epoch', type=int, default=12, metavar='N', help='max iterations for training')
parser.add_argument ('--momentum', type=float, default=0.9, metavar='M', help='percentage of past parameters to store')
parser.add_argument ('--log_interval', type=int, default=500, help='Interval for Logging')
parser.add_argument ('--disable_clip_gradient', action='store_true', help='Whether to clip the gradient')
parser.add_argument ('--use_normal_anchors', action='store_true', help='Whether to use kmeans anchors')
parser.add_argument ('--step_size', type=int, default=3, help='step to decay the learning rate')
parser.add_argument ('--batch_size', type=int, default=1, help='#images per batch')
parser.add_argument ('--workers', type=int, default=4)

# Environment Settings
parser.add_argument ('--dataset', type=str, default='nia', help='The dataset name')
parser.add_argument ('--dataset_option', type=str, default='small', help='The dataset to use (small | normal | fat)')
parser.add_argument ('--output_dir', type=str, default='./output/RPN', help='Location to output the model')
parser.add_argument ('--model_name', type=str, default='rpn', help='model name for snapshot')
parser.add_argument ('--resume', type=str, help='The model we resume')
parser.add_argument ('--path_rpn_opts', type=str, default='options/RPN/rpn.yaml', help='Path to RPN opts')
parser.add_argument ('--evaluate', action='store_true', help='To enable the evaluate mode')
parser.add_argument ('--dump_name', type=str, default='RPN_rois')
args = parser.parse_args ()

####################################################################################################################

def main ():
  global args
  global opts
  global data_opts

  info = {
    'torch': torch.__version__,
    'python': platform.python_version(),
    'cuda': torch.version.cuda,
    'cudnn': torch.backends.cudnn.version()
  }
  print ('[+] Information: {}'.format(info))

  pprint (args)
  print ('')

  ##################################################################

  print ('[+] Loading training set and testing set...')
  with open (args.path_data_opts, 'r') as f:
    data_opts = yaml.full_load (f)

  if args.dataset == 'nia':
    data_opts['dir']       = 'data/nia'
  elif args.dataset == 'visual_genome':
    data_opts['dir']       = 'data/visual_genome'

  pprint (data_opts)
  print ('')

  args.model_name += '_'+dt.now().strftime('%m%d%H')+'_' + args.dataset
  if args.dataset == 'visual_genome':
    train_set = visual_genome (data_opts, 'train', dataset_option=args.dataset_option, batch_size=args.batch_size)
    test_set  = visual_genome (data_opts, 'test',  dataset_option=args.dataset_option, batch_size=args.batch_size)
  else:
    train_set = nia (data_opts, 'train', dataset_option=args.dataset_option, batch_size=args.batch_size)
    test_set  = nia (data_opts, 'test',  dataset_option=args.dataset_option, batch_size=args.batch_size)
  print ('[+] Done. (train: {}, test: {})'.format(len(train_set), len(test_set)))

  with open (args.path_rpn_opts, 'r') as f:
    opts = yaml.full_load (f)
    opts['scale'] = train_set.opts['test']['SCALES'][0]

  if args.dataset == 'nia':
    opts['anchor_dir']     = 'data/nia'
    opts['kmeans_anchors'] = False
  elif args.dataset == 'visual_genome':
    opts['anchor_dir']     = 'data/visual_genome'
    opts['kmeans_anchors'] = True

  pprint (opts)
  print ('')

  ##################################################################

  net = RPN (opts)
  # pass enough message for anchor target generation
  train_set._feat_stride = net._feat_stride
  train_set._rpn_opts    = net.opts

  # in evluate mode, we disable the shuffle
  print ('[+] Loading training loader and testing loader...')

  if args.dataset == 'visual_genome':
    train_loader = torch.utils.data.DataLoader (train_set, batch_size=args.batch_size, 
      shuffle=False if args.evaluate else True, num_workers=args.workers, pin_memory= True, collate_fn=visual_genome.collate)
    test_loader = torch.utils.data.DataLoader  (test_set, batch_size=args.batch_size, 
      shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=visual_genome.collate)
  else:
    train_loader = torch.utils.data.DataLoader (train_set, batch_size=args.batch_size, 
      shuffle=False if args.evaluate else True, num_workers=args.workers, pin_memory= True, collate_fn=nia.collate)
    test_loader = torch.utils.data.DataLoader  (test_set, batch_size=args.batch_size, 
      shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=nia.collate)

  print ('[+] Done. (train: {}, test: {})'.format(len(train_loader), len(test_loader)))

  if args.resume is not None:
    print ('[+] resume training from: {}'.format (args.resume))
    RPN_utils.load_checkpoint (args.resume, net)
    optimizer = torch.optim.SGD ([{'params': list (net.parameters ())[26:]},], lr=args.lr, momentum=args.momentum, weight_decay=0.0005)
  else:
    print ('[+] training from scratch.')
    optimizer = torch.optim.SGD (list (net.parameters ())[26:], lr=args.lr, momentum=args.momentum, weight_decay=0.0005)

  network.set_trainable (net.features, requires_grad=False)
  net.cuda ()

  ##################################################################################################
  #
  # Evaluate
  #
  if args.evaluate:
    # evaluate training set
    data_dir = os.path.join ('output', 'RPN')
    filename = args.dump_name + '_' + dt.now().strftime('%m%d%H')
    net.eval ()
    evaluate (test_loader, net, path=os.path.join (data_dir, filename), dataset='test')

    return


  ##################################################################################################
  #
  # Train
  #
  if not os.path.exists (args.output_dir):
    os.mkdir (args.output_dir)

  best_recall = 0.
  for epoch in range (0, args.max_epoch):
    # Training
    train (train_loader, net, optimizer, epoch)

    # Testing
    net.eval ()
    recall, _ = test (test_loader, net)
    print ('[+] epoch[{epoch:d}]: Recall: object: {recall: .3f}% (Best: {best_recall: .3f}%)'.format (
      epoch  = epoch, 
      recall = recall * 100, 
      best_recall = best_recall * 100))

    # update learning rate
    if epoch % args.step_size == 0 and epoch > 0:
      args.disable_clip_gradient = True
      args.lr /= 10
      for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr

    save_name = os.path.join (args.output_dir, '{}'.format (args.model_name))+'_{:d}'.format(int(recall*100))
    RPN_utils.save_checkpoint (save_name, net, epoch, np.all (recall > best_recall))
    best_recall = recall if recall > best_recall else best_recall


####################################################################################################################

def train (train_loader, target_net, optimizer, epoch):
  batch_time             = network.AverageMeter ()
  data_time              = network.AverageMeter ()
  train_loss             = network.AverageMeter ()
  train_loss_obj_box     = network.AverageMeter ()
  train_loss_obj_entropy = network.AverageMeter ()

  accuracy_obj = network.AccuracyMeter ()

  target_net.train ()

  end = time.time ()

  for i, sample in enumerate (train_loader):
    # measure the data loading time
    data_time.update (time.time () - end)

    im_data    = sample['visual'][0].cuda ()
    im_info    = sample['image_info']
    gt_objects = sample['objects']

    anchor_targets = [
      np_to_variable (sample['rpn_targets']['object'][0][0],is_cuda=True, dtype=torch.LongTensor),
      np_to_variable (sample['rpn_targets']['object'][0][1],is_cuda=True),
      np_to_variable (sample['rpn_targets']['object'][0][2],is_cuda=True),
      np_to_variable (sample['rpn_targets']['object'][0][3],is_cuda=True) ]

    target_net (im_data, im_info, rpn_data=anchor_targets)                                   # Forward pass
    loss = target_net.loss ()                                                                # record loss
    train_loss.update (loss.data.item (), im_data.size (0))                                  # total loss
    train_loss_obj_box.update (target_net.loss_box.data.item (), im_data.size (0))           # object bbox reg
    train_loss_obj_entropy.update (target_net.loss_cls.data.item (), im_data.size (0))       # object score
    accuracy_obj.update (target_net.tp, target_net.tf, target_net.fg_cnt, target_net.bg_cnt) # accuracy

    # hyhwang: modify above pytorch code
    #train_loss.update (loss.data[0], im_data.size (0))
    #train_loss_obj_box.update (target_net.loss_box.data[0], im_data.size (0))
    #train_loss_obj_entropy.update (target_net.loss_cls.data[0], im_data.size (0))

    # backward
    optimizer.zero_grad ()
    torch.cuda.synchronize ()
    loss.backward ()

    if not args.disable_clip_gradient:
      network.clip_gradient (target_net, 10.)

    torch.cuda.synchronize ()
    optimizer.step ()

    # measure elapsed time
    batch_time.update (time.time () - end)
    end = time.time ()

    if  (i + 1) % args.log_interval == 0:
      print ('[+] Epoch: [{0}][{1}/{2}]\tBatch_Time: {batch_time.avg:.3f}s\tlr: {lr: f}\t'
        'Loss: {loss.avg:.4f}\n\t[object]:\ttp: {accuracy_obj.true_pos:.3f}, \t'
        'tf: {accuracy_obj.true_neg:.3f}, \tfg/bg=({accuracy_obj.foreground:.1f}/{accuracy_obj.background:.1f})\t'
        'cls_loss: {cls_loss_object.avg:.3f}\treg_loss: {reg_loss_object.avg:.3f}'.format(
         epoch, i + 1, len (train_loader), batch_time=batch_time,lr=args.lr,
         data_time=data_time, loss=train_loss, cls_loss_object=train_loss_obj_entropy, 
         reg_loss_object=train_loss_obj_box, accuracy_obj=accuracy_obj))



def test (test_loader, target_net):
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

    #Display (os.path.join(data_opts['dir'], 'images', sample['path'][0]))

    im_counter += im_data.size (0)
    im_info     = sample['image_info']
    gt_objects  = sample['objects']

    object_rois = target_net (im_data, im_info)[1]

    results.append (object_rois.cpu ().data.numpy ())
    box_num += object_rois.size (0)

    correct_cnt_t, total_cnt_t = check_recall (object_rois, gt_objects, 50)
    correct_cnt += correct_cnt_t
    total_cnt += total_cnt_t
    batch_time.update (time.time () - end)
    end = time.time ()
    if (i + 1) % 100 == 0 and i > 0:
      print ('[{0}/{6}]  Time: {1:2.3f}s/img).\t[object] Avg: {2:2.2f} Boxes/im, Top-50 recall: {3:2.3f} ({4:.0f}/{5:.0f})'.format (
        i + 1, batch_time.avg, box_num / float (im_counter), correct_cnt / float (total_cnt)* 100, correct_cnt, total_cnt, len (test_loader)))

  recall = correct_cnt / float (total_cnt)

  print ('====== Done Testing ====')

  return recall, results

#######################################################################################################

def evaluate (loader, net, path, dataset='train'):
  network.load_net('output/RPN/rpn_011616_nia_81_best.h5', net)
  recall, rois = test (loader, net)

  print ('[{}]\tRecall: object: {recall: .3f}%'.format (dataset, recall=recall * 100))
  print ('Saving ROIs...'),
  with open (path + '_object_' + dataset + '.pkl', 'wb') as f:
    pickle.dump (rois, f)

  print ('Done.')


#######################################################################################################

if __name__ == '__main__':
  main ()
