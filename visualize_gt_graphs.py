import os
import os.path as osp
from shutil import rmtree
import torch
import numpy as np
import random
import numpy.random as npr
import json
import pickle
import yaml
import cv2

from pprint import pprint

# from faster_rcnn.datasets.factory import get_imdb
import lib.datasets as datasets
from lib.visualize_graph.visualize import viz_scene_graph, draw_scene_graph
from lib.visualize_graph.vis_utils import check_recalled_graph

import argparse
import pdb

from PIL import Image

parser = argparse.ArgumentParser('options for Meteor evaluation')

parser.add_argument('--path_data_opts', default='options/data.yaml', type=str, help='path to a data file')
parser.add_argument('--dataset_option', default='normal', type=str, help='path to the evaluation result file')
parser.add_argument('--dataset', default='nia', type=str, help='path to the evaluation result file')
parser.add_argument('--output_dir', default='output/scene_graph/nia', type=str)
parser.add_argument('--path_result', type=str, help='path to the evaluation result file')
parser.add_argument('--topk', default=100, type=int, help='topK detections are used. ')

args = parser.parse_args()
print (args)

#hyhwang
#if args.dataset is not 'nia':
#    args.dataset_option = None

# def prepare_rel_matrix(relationships, object_num):
#     rel_mat = np.zeros()
#     for rel in len(relationships):
#         rel_mat[rel[0], rel[1]] = rel_cls[i]
#     return rel_mat


def draw_graph_pred (im, boxes, obj_ids, pred_relationships, ind_to_class, ind_to_predicate, filename):
    # hyhwang
    print('sample:', pred_relationships)
    #if rel_pred.size < 4:
    #    print('Image Skipped. {} size:{} pred_relationships:{}'.format(filename, rel_pred.size, len(list(pred_relationships)))) #hyhwang?
    #    print (list(pred_relationships))
    #    return

    # indices of predicted boxes
    pred_inds = np.array(pred_relationships)[:, :2].ravel()
    print (pred_inds)
    #print (rel_pred.shape, rel_pred.size, type(rel_pred), type(pred_relationships))
    #pred_inds = np.array(pred_relationships).ravel()

    # draw graph predictions
    print ('obj_ids:',   obj_ids)
    print ('pred_inds:', pred_inds)
    print ('ind_to_class:', ind_to_class)
    print ('ind_to_predicate:', ind_to_predicate)
    graph_dict = draw_scene_graph (obj_ids, pred_inds, pred_relationships, ind_to_class, ind_to_predicate, filename=filename)
    viz_scene_graph(im, boxes, obj_ids, ind_to_class, ind_to_predicate, pred_inds, pred_relationships, filename=filename)
    """
    out_boxes = []
    for box, cls in zip(boxes[pred_inds], cls_pred[pred_inds]):
        out_boxes.append(box[cls*4:(cls+1)*4].tolist())

    graph_dict['boxes'] = out_boxes

    if do_save == 'y':
        scipy.misc.imsave('cherry/im_%i.png' % idx, im)
        fn = open('cherry/graph_%i.json' % idx, 'w+')
        json.dump(graph_dict, fn)
    print(idx)
    """

def visualize():
    high_recall_cases = []
    low_recall_cases = []

    global args
    print('=========== Visualizing Scene Graph =========')


    print('Loading dataset...'),
    with open(args.path_data_opts, 'r') as handle:
        options = yaml.full_load(handle)
    test_set    = getattr(datasets, args.dataset)(options, 'test', dataset_option=args.dataset_option, use_region=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, collate_fn=getattr(datasets, args.dataset).collate)
    print('Done Loading')

    print('Generated graphs will be saved: {}'.format(args.output_dir))
    if osp.isdir(args.output_dir):
        rmtree(args.output_dir)
    os.makedirs(args.output_dir)

    if args.path_result is not None:
        with open(args.path_result, 'rb') as f:
            print('Loading result....'),
            result = pickle.load(f)
            print('Done')
            print('Total: {} images'.format(len(result)))
    else: 
        result = None

    for i, sample in enumerate(test_loader): # (im_data, im_info, gt_objects, gt_relationships)
        gt_boxes = sample['objects'][0][:, :4] / sample['image_info'][0][2]
        #hyhwang
        #gt_cls = sample['objects'][0][:, 4].astype(np.int)
        gt_cls = sample['objects'][0][:, 4].astype(int)
        # hyhwang
        gt_relations = sample['relations'][0]
        gt_relations  = zip(*np.where(gt_relations > 0))
        print (gt_relations)
        imagename = sample['path'][0].split('/')[-1].split('.')[0]
        #print (imagename, sample['relations'])
        filename  = osp.join(args.output_dir, imagename)

        if result is not None:
            if max(result[i]['rel_recall']) > 0.7 and gt_cls.shape[0] >= 3:
                high_recall_cases.append(imagename)
            elif max(result[i]['rel_recall']) < 0.6 and max(result[i]['rel_recall']) > 0.3 and gt_cls.shape[0] >= 4:
                low_recall_cases.append(imagename)
            

            pred_objects = result[i]['objects']
            pred_relationships = result[i]['relationships']
            assert sample['path'][0] == result[i]['path'], 'Image mismatch.'
            gt_objects = np.concatenate([gt_boxes, gt_cls[:, np.newaxis]], axis=1)
            det_objects = np.concatenate([pred_objects['bbox'], pred_objects['class'][:, np.newaxis]], axis=1)
            det_relations = np.zeros([det_objects.shape[0], det_objects.shape[0]])
            for rel in pred_relationships:
                det_relations[rel[0], rel[1]] = rel[2]
            gt_objects, gt_relations = check_recalled_graph (gt_objects, gt_relations, 
                                        det_objects[:args.topk], det_relations[:args.topk, :args.topk])
            if gt_objects.shape[0] == 0:
                print('Skipped: {}'.format(imagename))
                continue
            # hyhwang
            #gt_objects = gt_objects.astype(np.int)
            gt_objects = gt_objects.astype(int)
            gt_boxes   = gt_objects[:, :4]
            gt_cls     = gt_objects[:, 4]

        #rel_ids = gt_relations[gt_relations > 0] # hyhwang
        #temp_relation_ids = np.where(gt_relations > 0)
        #gt_relations = zip(temp_relation_ids[0], temp_relation_ids[1], rel_ids)
        im = cv2.imread(osp.join(test_set._data_path, sample['path'][0]))

        # hyhwang
        print ('gt_relations:', list(gt_relations))
        draw_graph_pred (im, gt_boxes, gt_cls, gt_relations, test_set._object_classes, test_set._predicate_classes, filename=filename)

    if result is not None:
        print('Dumping the high/low recall cases.')
        with open(osp.join(args.output_dir, 'high_recall_cases.txt'), 'w') as f:
            for s in high_recall_cases:
                f.write(s+'\n')
        with open(osp.join(args.output_dir, 'low_recall_cases.txt'), 'w') as f:
            for s in low_recall_cases:
                f.write(s+'\n')

    print ('Done generating scene graphs.')


if __name__ == '__main__':
    visualize()


