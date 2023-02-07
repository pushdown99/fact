import os
import os.path as osp
import torch
import numpy as np
import random
import numpy.random as npr
import json
import pickle
import yaml
import cv2
from tqdm import tqdm

from pprint import pprint

# from faster_rcnn.datasets.factory import get_imdb
import lib.datasets as datasets
from lib.visualize_graph.vis_utils import ground_predictions
from lib.visualize_graph.visualize import viz_scene_graph, draw_scene_graph


import argparse
import pdb

from PIL import Image 

#from eval.evaluator import DenseCaptioningEvaluator


parser = argparse.ArgumentParser('Options for Meteor evaluation')

parser.add_argument('--path_data_opts', default='options/data.yaml', type=str, help='path to a data file')
parser.add_argument('--path_result', default='output/testing_result.pkl', type=str, help='path to the evaluation result file')
parser.add_argument('--output_dir', default='output/graph_results/nia', type=str, help='path to the evaluation result file')
parser.add_argument('--dataset_option', default='normal', type=str, help='path to the evaluation result file')
parser.add_argument('--dataset', default='nia', type=str, help='path to the evaluation result file')

args = parser.parse_args()

def visualize():

    global args
    print ('=========== Visualizing Scene Graph =========')


    print ('Loading dataset...'),
    with open(args.path_data_opts, 'r') as handle:
        options = yaml.full_load(handle)

    test_set    = getattr(datasets, args.dataset)(options, 'test', dataset_option=args.dataset_option, use_region=False) # hyhwang: True => False
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, collate_fn=getattr(datasets, args.dataset).collate)

    with open(args.path_result, 'rb') as f:
        print ('Loading evaluate result: {}'.format(args.path_result)),
        result = pickle.load(f)
        print ('Total: {} images'.format(len(result)))
        print ()

    for i, sample in enumerate(tqdm(test_loader, desc='Genrating scene graph')): # (im_data, im_info, gt_objects, gt_relationships)
        objects       = result[i]['objects']
        relationships = result[i]['relationships']
        gt_boxes      = sample['objects'][0][:, :4] / sample['image_info'][0][2]
        gt_relations  = sample['relations'][0]
        gt_relations  = zip(*np.where(gt_relations > 0))
        gt_to_pred    = ground_predictions (objects['bbox'], gt_boxes, 0.5) # hyhwang 0.5 => 0.2
        #print (sample['path'][0], result[i]['path']) #hyhwang
        assert sample['path'][0] == result[i]['path'], 'Image mismatch.'
        im = cv2.imread(osp.join(test_set._data_path, sample['path'][0]))
        image_name = sample['path'][0].split('/')[-1].split('.')[0]
        image_name = osp.join(args.output_dir, image_name)

        # hyhwang: adjust if condition
        draw_graph_pred (im, objects['bbox'], objects['class'], relationships, gt_to_pred, gt_relations, 
          test_set._object_classes, test_set._predicate_classes, filename=image_name)

    print ('Done generating scene graphs.')


def draw_graph_pred (im, boxes, obj_ids, pred_relationships, gt_to_pred, gt_relations, ind_to_class, ind_to_predicate, filename):
    """
    Draw a predicted scene graph. To keep the graph interpretable, only draw
    the node and edge predictions that have correspounding ground truth
    labels.
    args:
        im: image
        boxes: prediceted boxes
        obj_ids: object id list
        rel_pred_mat: relation classification matrix
        gt_to_pred: a mapping from ground truth box indices to predicted box indices
        idx: for saving
        gt_relations: gt_relationships
    """
    rel_pred = []
    all_rels = []

    for pred_rel in pred_relationships: # result's relationship (sub. obj, pred, score)
        for rel in gt_relations:        # compare w/ dataset relations
            if rel[0] not in gt_to_pred or rel[1] not in gt_to_pred:
                continue

            #print ('found:', pred_rel)
            rel_pred.append(pred_rel)
            all_rels.append([pred_rel[0], pred_rel[1]])
            break

            # discard duplicate grounding, hyhwang: found duplicate grounding bug, so remove this code
            #if pred_rel[0] == gt_to_pred[rel[0]] and pred_rel[1] == gt_to_pred[rel[1]]:
            #    print ('found2:', pred_rel)
            #    rel_pred.append(pred_rel)
            #    all_rels.append([pred_rel[0], pred_rel[1]])
            #    break
    # rel_pred = pred_relationships[:5]  # uncomment to visualize top-5 relationships
    rel_pred = np.array(rel_pred)

    # hyhwang
    if rel_pred.size < 4:
        # print('Image Skipped. {} size:{} pred_relationships:{} gt_relations:{}'.format(filename, rel_pred.size, len(list(pred_relationships)), len(list(gt_relations)))) 
        return
    # indices of predicted boxes
    pred_inds = rel_pred[:, :2].ravel()

    # draw graph predictions
    #graph_dict = draw_scene_graph (obj_ids, pred_inds, rel_pred, ind_to_class, ind_to_predicate, filename=filename)
    draw_scene_graph (obj_ids, pred_inds, rel_pred, ind_to_class, ind_to_predicate, filename=filename)
    viz_scene_graph (im, boxes, obj_ids, ind_to_class, ind_to_predicate, pred_inds, rel_pred, filename=filename)

if __name__ == '__main__':
    visualize()


