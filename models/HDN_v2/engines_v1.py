import pdb
import os.path as osp
import time
import json
import codecs
import numpy as np
from os.path import join, basename

from tqdm import tqdm
import datetime
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

import lib.network as network
from lib.network import np_to_variable
from .utils import nms_detections, build_loss_bbox, build_loss_cls, interpret_relationships, interpret_objects

def current ():
    return  datetime.datetime.now()

def line1_80 ():
    return '--------------------------------------------------------------------------------'

def line1_120 ():
    return '------------------------------------------------------------------------------------------------------------------------'

def line2_80 ():
    return '================================================================================'

def line2_120 ():
    return '========================================================================================================================'

def train (loader, model, optimizer, exp_logger, epoch, train_all, print_freq=100, clip_gradient=True, iter_size=1):

    model.train ()
    meters = exp_logger.reset_meters ('train')
    end = time.time ()

    #print ('[-] HDNv2 train called')

    for i, sample in enumerate (loader): # (im_data, im_info, gt_objects, gt_relationships)
        #print ('[-] HDNv2 training # {}'.format(i))
        # measure the data loading time
        batch_size = len (sample['visual'])

        # measure data loading time
        meters['data_time'].update (time.time () - end, n=batch_size)

        input_visual = [item for item in sample['visual']]
        target_objects = sample['objects']
        target_relations = sample['relations']
        image_info = sample['image_info']
        # RPN targets
        rpn_anchor_targets_obj = [[
                np_to_variable (item[0],is_cuda=False, dtype=torch.LongTensor),
                np_to_variable (item[1],is_cuda=False),
                np_to_variable (item[2],is_cuda=False),
                np_to_variable (item[3],is_cuda=False)
                ] for item in sample['rpn_targets']['object']]

        # compute output
        #try:
        raw_losses = model ( im_data=input_visual, im_info=image_info, gt_objects=target_objects,
                    gt_relationships=target_relations, rpn_anchor_targets_obj=rpn_anchor_targets_obj)
        # Determine the loss function
        def merge_losses (losses):
            for key in losses:
                if isinstance (losses[key], dict) or isinstance (losses[key], list):
                    losses[key] = merge_losses (losses[key])
                elif key.startswith ('loss'):
                    losses[key] = losses[key].mean ()
            return losses 

        losses = merge_losses (raw_losses)

        if train_all:
            loss = losses['loss'] + losses['rpn']['loss'] * 0.5
        else:
            loss = losses['loss']
        # to logging the loss and itermediate values
        meters['loss'].update (losses['loss'].cpu ().item (), n=batch_size)
        meters['loss_cls_obj'].update (losses['loss_cls_obj'].cpu ().item (), n=batch_size)
        meters['loss_reg_obj'].update (losses['loss_reg_obj'].cpu ().item (), n=batch_size)
        meters['loss_cls_rel'].update (losses['loss_cls_rel'].cpu ().item (), n=batch_size)
        meters['loss_rpn'].update (losses['rpn']['loss'].cpu ().item (), n=batch_size)
        meters['batch_time'].update (time.time () - end, n=batch_size)
        meters['epoch_time'].update (meters['batch_time'].val, n=batch_size)

        # add support for iter size
        # special case: last iterations
        optimizer.zero_grad (set_to_none=True)

        # hyhwang for DP
        loss.backward ()
        #loss.mean ().backward ()

        if clip_gradient:
            network.clip_gradient (model, 10.)
        else:
            network.avg_gradient (model, iter_size)
        optimizer.step ()
        
        #except:
        #    #import pdb; pdb.set_trace ()
        #    print ("Error: [{}]".format (i))
        end = time.time ()
        # Logging the training loss
        if  (i + 1) % print_freq == 0:
            print ('Epoch: [{0}][{1}/{2}] '
                  'Batch_Time: {batch_time.avg: .3f}\t'
                  'FRCNN Loss: {loss.avg: .4f}\t'
                  'RPN Loss: {rpn_loss.avg: .4f}\t'.format (
                   epoch, i + 1, len (loader), batch_time=meters['batch_time'],
                   loss=meters['loss'], rpn_loss=meters['loss_rpn']))


            print ('\t[object] loss_cls_obj: {loss_cls_obj.avg:.4f} '
            	  'loss_reg_obj: {loss_reg_obj.avg:.4f} '
            	  'loss_cls_rel: {loss_cls_rel.avg:.4f} '.format (
                  loss_cls_obj = meters['loss_cls_obj'],
                  loss_reg_obj = meters['loss_reg_obj'],
                  loss_cls_rel = meters['loss_cls_rel'], ))

    print ('[-] RPN train ended.')
    exp_logger.log_meters ('train', n=epoch)


def inference (model, sample = None):
    print ('========== Inference ==========')
    model.eval ()

    print (sample['path'])
    im_data = sample['visual'][0].cuda ()
    print (im_data.shape)
    im_info = sample['image_info']
    object_result, predicate_result = model.module.forward_eval (im_data, im_info,)
    cls_prob_object, bbox_object, object_rois, reranked_score = object_result[:4]
    cls_prob_predicate, mat_phrase = predicate_result[:2]
    region_rois_num = predicate_result[2]

    #obj_boxes, obj_scores, obj_cls, subject_inds, object_inds, subject_boxes, object_boxes, predicate_inds, sub_assignment, obj_assignment, total_score = \
    #        interpret_relationships (cls_prob_object, bbox_object, object_rois, cls_prob_predicate, mat_phrase, im_info,
    #                    nms=-1., top_N=1, use_gt_boxes=False, triplet_nms=-1., reranked_score=reranked_score)

    return interpret_relationships (cls_prob_object, bbox_object, object_rois, cls_prob_predicate, mat_phrase, im_info, \
             nms=-1., top_N=1, use_gt_boxes=False, triplet_nms=-1., reranked_score=reranked_score)

    #print (subject_inds[0], object_inds[0], predicate_inds[0], total_score[0])
    #print (subject_inds[1], object_inds[1], predicate_inds[1], total_score[1])
    #print (subject_inds[2], object_inds[2], predicate_inds[2], total_score[2])

def print_sg (top_Ns, rel_cnt, pred_cnt_correct, phrase_cnt_correct, rel_cnt_correct, interim=False, image_name=None):
  if interim == True:
    print ()
    print (current(), '{:20s} {:7s} {:7s} {:7s} {:7s} {:7s}  {:7s}  {:7s}     {}'.format('Recall (Top-N)', 'RelCnt', 'PredCnt', 'PhrCnt', 'SGCnt', 'PredCls', 'PhrCls', 'SGCls', 'Using latest image'))
    print (current(), line1_120())
    for idx, top_N in enumerate (top_Ns):
      print (current(), 'Top {:3d}, Recall      {:7d} {:7d} {:7d} {:7d} {:7.2f}% {:7.2f}% {:7.2f}%    {}'.format(
        int(top_N), 
        int(rel_cnt), 
        int(pred_cnt_correct[idx]), 
        int(phrase_cnt_correct[idx]), 
        int(rel_cnt_correct[idx]),
        pred_cnt_correct[idx]   / float (rel_cnt) * 100,
        phrase_cnt_correct[idx] / float (rel_cnt) * 100,
        rel_cnt_correct[idx]    / float (rel_cnt) * 100,
        image_name))
  else:
    print ()
    print (current(), '* Final PredCls, SGCls for K-Vision dataset')
    print (current(), line2_120())
    print (current(), '{:20s} {:7s} {:7s} {:7s} {:7s} {:7s}  {:7s}  {:7s}'.format('Recall (Top-N)', 'RelCnt', 'PredCnt', 'PhrCnt', 'SGCnt', 'PredCls', 'PhrCls', 'SGCls'))
    print (current(), line1_120())
    for idx, top_N in enumerate (top_Ns):
      print (current(), 'Top {:3d}, Recall      {:7d} {:7d} {:7d} {:7d} {:7.2f}% {:7.2f}% {:7.2f}%'.format(
        int(top_N), 
        int(rel_cnt), 
        int(pred_cnt_correct[idx]), 
        int(phrase_cnt_correct[idx]), 
        int(rel_cnt_correct[idx]),
        pred_cnt_correct[idx]   / float (rel_cnt) * 100,
        phrase_cnt_correct[idx] / float (rel_cnt) * 100,
        rel_cnt_correct[idx]    / float (rel_cnt) * 100))

def print_per_category (statistics, top_Ns):
    print ('')
    print (current(), '* Recall per Categories (K-Vision dataset)')
    print (current(), line2_120())
    print (current(), '{:20s} {:8s} {:8s} {:7s} {:7s} {:7s} {:7s} {:7s}  {:7s}  {:7s}'.format('Category', '#Dataset', 'Top-N', 'RelCnt', 'PredCnt', 'PhrCnt', 'SGCnt', 'PredCls', 'PhrCls', 'SGCls'))
    print (current(), line1_120())
    for k, v in statistics.items():
      for idx, top_N in enumerate (top_Ns):
        _rel_cnt            = v['rel_cnt'][idx]
        _pred_cnt_correct   = v['pred_cnt_correct'][idx]
        _phrase_cnt_correct = v['phrase_cnt_correct'][idx]
        _rel_cnt_correct    = v['rel_cnt_correct'][idx]

        print (current(), '{:20s} {:8d} {:8d} {:7d} {:7d} {:7d} {:7d} {:7.2f}% {:7.2f}% {:7.2f}%'.format(
          k, 
          v['total'], 
          int(top_N), 
          int(_rel_cnt), 
          int(_pred_cnt_correct), 
          int(_phrase_cnt_correct), 
          int(_rel_cnt_correct),
          _pred_cnt_correct / float (_rel_cnt) * 100,
          _phrase_cnt_correct / float (_rel_cnt) * 100,
          _rel_cnt_correct / float (_rel_cnt) * 100))

def test (loader, model, top_Ns, nms=-1., triplet_nms=-1., use_gt_boxes=False, opt=None):

    statistics   = dict ()
    print ('[+] dataset            :', opt['data']['dataset'], opt['data']['nick'])
    nick = opt['data']['nick']
    print ('[+] Loading statistics :', opt['data']['result'])
    result = json.load(codecs.open(opt['data']['result'],  'r', 'utf-8-sig'))

    print ()
    print (current(), '[*] Evaluattion w/ K-vision dataset')
    print (current(), line1_80())

    model.eval ()

    rel_cnt = 0.
    rel_cnt_correct       = np.zeros (2)
    phrase_cnt_correct    = np.zeros (2)
    pred_cnt_correct      = np.zeros (2)
    total_region_rois_num = 0
    max_region_rois_num   = 0
    res                   = []

    batch_time = network.AverageMeter ()
    end        = time.time ()

    for i, sample in enumerate (tqdm(loader)): # (im_data, im_info, gt_objects, gt_relationships)
        assert len (sample['visual']) == 1
        input_visual     = sample['visual'][0].cuda ()
        image_name       = sample['path'][0] # hyhwang
        gt_objects       = sample['objects']
        gt_relationships = sample['relations']
        image_info       = sample['image_info']
        # Forward pass

        if nick == 'vg':
          id = basename(image_name)
        else:
          id = basename(image_name).split('_')[0]+'_'+basename(image_name).split('_')[1]
        #print (image_name, id)
        #print (id, result['evals'][id])

        # call (factorizable_network) evaluate, hyhwang
        total_cnt_t, cnt_correct_t, eval_result_t = model.module.evaluate (
            input_visual, image_info, gt_objects, gt_relationships,
            top_Ns = top_Ns, nms=nms, triplet_nms=triplet_nms,
            use_gt_boxes=use_gt_boxes, image_name=image_name)

        eval_result_t['path'] = sample['path'][0] # for visualization
        rel_cnt              += total_cnt_t

        res.append (eval_result_t)

        # hyhwang
        rel_cnt_correct       += cnt_correct_t[0]
        phrase_cnt_correct    += cnt_correct_t[1]
        pred_cnt_correct      += cnt_correct_t[2]
        total_region_rois_num += cnt_correct_t[3]
        max_region_rois_num    = cnt_correct_t[3] if cnt_correct_t[3] > max_region_rois_num else max_region_rois_num
         
        for c in result ['evals'][id]:
          for idx, top_N in enumerate (top_Ns):
            if not c in statistics:
              statistics [c] = dict ()
              statistics [c]['total'] = result['stats'][c]['total']
              statistics [c]['train'] = result['stats'][c]['train']
              statistics [c]['rel_cnt']            = [0, 0]
              statistics [c]['pred_cnt_correct']   = [0, 0]
              statistics [c]['phrase_cnt_correct'] = [0, 0]
              statistics [c]['rel_cnt_correct']    = [0, 0] 

            statistics [c]['rel_cnt'][idx] += rel_cnt
            statistics [c]['pred_cnt_correct'][idx]    +=  pred_cnt_correct [idx]
            statistics [c]['phrase_cnt_correct'][idx]  +=  phrase_cnt_correct [idx]
            statistics [c]['rel_cnt_correct'][idx]     +=  rel_cnt_correct [idx]

        batch_time.update (time.time () - end)
        end = time.time ()

        # hyhwang
        if (i + 1) % 15 == 0 and i > 0:
            print_sg (top_Ns, rel_cnt, pred_cnt_correct, phrase_cnt_correct, rel_cnt_correct, True, image_name)
#            print (current(), '{:20s} {:7s} {:7s} {:7s} {:7s} {:7s}  {:7s}  {:7s}     {}'.format('Recall (Top-N)', 'RelCnt', 'PredCnt', 'PhrCnt', 'SGCnt', 'PredCls', 'PhrCls', 'SGCls', 'Using latest image'))
#            print (current(), line1_120())
#            for idx, top_N in enumerate (top_Ns):
#              print (current(), 'Top {:3d}, Recall      {:7d} {:7d} {:7d} {:7d} {:7.2f}% {:7.2f}% {:7.2f}%    {}'.format(
#                int(top_N), 
#                int(rel_cnt), 
#                int(pred_cnt_correct[idx]), 
#                int(phrase_cnt_correct[idx]), 
#                int(rel_cnt_correct[idx]),
#                pred_cnt_correct[idx]   / float (rel_cnt) * 100,
#                phrase_cnt_correct[idx] / float (rel_cnt) * 100,
#                rel_cnt_correct[idx]    / float (rel_cnt) * 100,
#                image_name))

#            print ('[Evaluation][%d/%d][%.2fs/img][avg: %d subgraphs, max: %d subgraphs]' % \
#              (i+1, len (loader), batch_time.avg, total_region_rois_num / float (i+1), max_region_rois_num))
#
#            for idx, top_N in enumerate (top_Ns):
#                print ('\tTop-%d Recall(HDN):\t[PredCls] %2.3f%%\t[PhrCls %2.3f%%\t[SGCls] %2.3f%%\t[rel_cnt] %2.3f' % (
#                    top_N, 
#                    pred_cnt_correct[idx]   / float (rel_cnt) * 100,
#                    phrase_cnt_correct[idx] / float (rel_cnt) * 100,
#                    rel_cnt_correct[idx]    / float (rel_cnt) * 100, 
#                    float (rel_cnt)))


    print_per_category (statistics, top_Ns)
    print_sg (top_Ns, rel_cnt, pred_cnt_correct, phrase_cnt_correct, rel_cnt_correct)

#    print ('')
#    print (current(), '* Recall per Categories (K-Vision dataset)')
#    print (current(), line2_120())
#    print (current(), '{:20s} {:8s} {:8s} {:7s} {:7s} {:7s} {:7s} {:7s}  {:7s}  {:7s}'.format('Category', '#Dataset', 'Top-N', 'RelCnt', 'PredCnt', 'PhrCnt', 'SGCnt', 'PredCls', 'PhrCls', 'SGCls'))
#    print (current(), line1_120())
#    for k, v in statistics.items():
#      for idx, top_N in enumerate (top_Ns):
#        _rel_cnt            = v['rel_cnt'][idx]
#        _pred_cnt_correct   = v['pred_cnt_correct'][idx]
#        _phrase_cnt_correct = v['phrase_cnt_correct'][idx]
#        _rel_cnt_correct    = v['rel_cnt_correct'][idx]
#
#        print (current(), '{:20s} {:8d} {:8d} {:7d} {:7d} {:7d} {:7d} {:7.2f}% {:7.2f}% {:7.2f}%'.format(
#          k, 
#          v['total'], 
#          int(top_N), 
#          int(_rel_cnt), 
#          int(_pred_cnt_correct), 
#          int(_phrase_cnt_correct), 
#          int(_rel_cnt_correct),
#          _pred_cnt_correct / float (_rel_cnt) * 100,
#          _phrase_cnt_correct / float (_rel_cnt) * 100,
#          _rel_cnt_correct / float (_rel_cnt) * 100))
      
    print (current(), line2_120())
    
    #recall = [rel_cnt_correct / float (rel_cnt), phrase_cnt_correct / float (rel_cnt), pred_cnt_correct / float (rel_cnt)]
    recall = [[rel_cnt, rel_cnt], pred_cnt_correct, phrase_cnt_correct, rel_cnt_correct, pred_cnt_correct / float (rel_cnt), phrase_cnt_correct / float (rel_cnt), rel_cnt_correct / float (rel_cnt)]
    print (current(), 'Done.')

    return recall, res


def test_object_detection (loader, model, nms=-1., use_gt_boxes=False):
    print ('========== Testing (object detection) ==========')
    model.eval ()
    object_classes = loader.dataset.object_classes
    result = {obj: {} for obj in object_classes}

    for i, sample in enumerate (loader): # (im_data, im_info, gt_objects, gt_relationships)
        input_visual = sample['visual'][0].cuda ()
        path         = sample['path'][0] # hyhwang
        gt_objects   = sample['objects']
        image_info   = sample['image_info']
        # Forward pass
        boxes = model.evaluate_object_detection (input_visual, image_info, gt_objects, nms=nms, use_gt_boxes=use_gt_boxes)
        filename = osp.splitext (sample['path'][0])[0] # for visualization
        assert len (boxes) == len (result), "The two should have same length (object categories)"
        for j, obj_class in enumerate (object_classes):
            if j == 0:
                continue
            result[obj_class][filename] = boxes[j]
        if (i + 1) % 500 == 0 and i > 0:
            print ('[Evaluation][%d/%d] processed.' % (i+1, len (loader)))

    return result
