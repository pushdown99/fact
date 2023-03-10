# --------------------------------------------------------
# Scene Graph Generation by Iterative Message Passing
# Licensed under The MIT License [see LICENSE for details]
# Written by Danfei Xu
# --------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
from graphviz import Digraph
import cv2

import pdb

"""
Utility for visualizing a scene graph
"""




def draw_scene_graph (labels, inds, rels, ind_to_class, ind_to_predicate, filename):
    """
    draw a graphviz graph of the scene graph topology
    """
    #print ('draw_scene_graph', type(labels), inds, type(inds)) #hyhwang
    viz_labels = labels[inds.astype(int)] #hyhwang: astype(int)
    viz_rels = None
    if rels is not None:
        viz_rels = []
        for rel in rels:
            if rel[0] in inds and rel[1] in inds :
                sub_idx = np.where(inds == rel[0])[0][0]
                obj_idx = np.where(inds == rel[1])[0][0]
                viz_rels.append([sub_idx, obj_idx, rel[2]])
    return draw_graph(viz_labels, viz_rels, ind_to_class, ind_to_predicate, filename)


def draw_graph(labels, rels, ind_to_class, ind_to_predicate, filename):
    u = Digraph('sg', filename=filename)
    u.body.append('size="6,6"')
    u.body.append('rankdir="LR"')
    u.node_attr.update(style='filled')

    out_dict = {'ind_to_class': ind_to_class, 'ind_to_predicate': ind_to_predicate}
    out_dict['labels'] = labels.tolist()
    out_dict['relations'] = rels

    rels = np.array(rels)
    rel_inds = rels[:,:2].ravel().tolist()
    name_list = []
    for i, l in enumerate(labels):
        if i in rel_inds:
            name = ind_to_class[l]
            name_suffix = 1
            obj_name = name
            while obj_name in name_list:
                obj_name = name + '_' + str(name_suffix)
                name_suffix += 1
            name_list.append(obj_name)
            u.node(str(i), label=obj_name, color='lightblue2')

    for rel in rels:
        edge_key = '%s_%s' % (rel[0], rel[1])
        u.node(edge_key, label=ind_to_predicate[rel[2].astype(int)], color='red') #hyhwang: astype

        u.edge(str(rel[0]), edge_key)
        u.edge(edge_key, str(rel[1]))

    u.render(cleanup=True) # save the graph to file and remove the source
    return out_dict


def viz_scene_graph(im, rois, labels, ind_to_class, ind_to_predicate, inds=None, rels=None, filename=None):
    """
    visualize a scene graph on an image
    """
    if inds is None:
        inds = np.arange(rois.shape[0])
    viz_rois = rois[inds.astype(int)] #hyhwang
    viz_labels = labels[inds.astype(int)]
    viz_rels = None
    if rels is not None:
        viz_rels = []
        for rel in rels:
            if rel[0] in inds and rel[1] in inds :
                sub_idx = np.where(inds == rel[0])[0][0]
                obj_idx = np.where(inds == rel[1])[0][0]
                viz_rels.append([sub_idx, obj_idx, rel[2]])
        viz_rels = np.array(viz_rels)
    return _viz_scene_graph(im, viz_rois, viz_labels, ind_to_class, ind_to_predicate, viz_rels, filename)


def _viz_scene_graph(im, rois, labels, ind_to_class, ind_to_predicate, rels, filename):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), aspect='equal')
    if rels.size > 0:
        rel_inds = rels[:,:2].ravel().tolist()
    else:
        rel_inds = []
    # draw bounding boxes
    for i, bbox in enumerate(rois):
        if int(labels[i]) == 0 and i not in rel_inds:
            continue
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        label_str = ind_to_class[int(labels[i])]
        ax.text(bbox[0], bbox[1] - 2,
                label_str,
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    # draw relations
    for i, rel in enumerate(rels):
        if rel[2] == 0: # ignore bachground
            continue
        sub_box = rois[rel[0].astype(int), :] #hyhwang
        obj_box = rois[rel[1].astype(int), :]
        obj_ctr = [obj_box[0], obj_box[1] - 2]
        sub_ctr = [sub_box[0], sub_box[1] - 2]
        line_ctr = [(sub_ctr[0] + obj_ctr[0]) / 2, (sub_ctr[1] + obj_ctr[1]) / 2]
        predicate = ind_to_predicate[int(rel[2])]
        ax.arrow(sub_ctr[0], sub_ctr[1], obj_ctr[0]-sub_ctr[0], obj_ctr[1]-sub_ctr[1], color='green')

        ax.text(line_ctr[0], line_ctr[1], predicate,
                bbox=dict(facecolor='green', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title('Scene Graph Visualization', fontsize=14)
    ax.axis('off')
    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename + '.png')
    plt.close(fig)
