import  os, re, sys, time, math, json, random, codecs, argparse
import json
import pickle
import numpy
import string

from os.path import basename, join, relpath, realpath, abspath, exists
from pprint import pprint
from glob import glob
from tqdm import tqdm
from pathlib import Path
from nltk import flatten


class Option:
  dataset    = 'nia'
  images_dir = 'images'
  info_dir   = 'info'

  #
  objects_json    = join(info_dir, 'objects.json')
  replace_json    = join(info_dir, 'replace.json')
  eng2kor_json    = join(info_dir, 'eng2kor.json')
  kor2eng_json    = join(info_dir, 'kor2eng.json')

  #
  dicts_json      = join(info_dir, 'dict.json')
  categories_json = join(info_dir, 'categories.json')
  inv_weight_json = join(info_dir, 'inverse_weight.json')
  forbidden_json  = join(info_dir, 'forbidden.json')

  # for save file
  files_json      = join(info_dir, '_files.json')
  files_text      = join(info_dir, '_files.txt')
  instances_pkl   = join(info_dir, '_instances.pkl')
  instances_json  = join(info_dir, '_instances.json')
  object_id_json  = join(info_dir, '_object_id.json')
  id_object_json  = join(info_dir, '_id_object.json')
  cls_files_json  = join(info_dir, '_cls_files.json')
  obj_files_json  = join(info_dir, '_obj_files.json')
  cls_stats_json  = join(info_dir, '_cls_stats.json')
  obj_stats_json  = join(info_dir, '_obj_stats.json')
  captions_json   = join(info_dir, 'captions.json')
  unalias_json    = join(info_dir, '_unalias.json')
  pred_dicts_json = join(info_dir, '_pred_dicts.json')
  obj_dicts_json  = join(info_dir, '_obj_dicts.json')
  word_dicts_json = join(info_dir, '_word_dicts.json')

  # for prepare option
  n_caption = 10
  lang      = 'ko'
  using     = 4

  # for build option
  min_per_class = 50
  num_per_class = 60000
  limits = 100000
  split  = [0.8, 0.1]

  # for processing
  _images_dir    = 'download/images'
  _bboxes_dir    = 'download/bbox'
  _captions_dir  = 'download/caption'
  _relations_dir = 'download/relation'

  trainvaltest_text = join(info_dir, 'trainvaltest.txt')
  trainval_text     = join(info_dir, 'trainval.txt')
  train_text        = join(info_dir, 'train.txt')
  val_text          = join(info_dir, 'val.txt')
  test_text         = join(info_dir, 'test.txt')

  c_trainvaltest_json = join(info_dir, 'c_trainvaltest.json')
  c_trainval_json     = join(info_dir, 'c_trainval.json')
  c_train_json        = join(info_dir, 'c_train.json')
  c_val_json          = join(info_dir, 'c_val.json')
  c_test_json         = join(info_dir, 'c_test.json')
  c_text_json         = join(info_dir, 'c_text.json')

  trainvaltest_json  = join(info_dir, 'trainvaltest.json')
  trainval_json      = join(info_dir, 'trainval.json')
  train_json         = join(info_dir, 'train.json')
  val_json           = join(info_dir, 'val.json')
  test_json          = join(info_dir, 'test.json')
  train_fat_json     = join(info_dir, 'train_fat.json')
  val_fat_json       = join(info_dir, 'val_fat.json')
  test_fat_json      = join(info_dir, 'test_fat.json')
  train_small_json   = join(info_dir, 'train_small.json')
  val_small_json     = join(info_dir, 'val_small.json')
  test_small_json    = join(info_dir, 'test_small.json')

  # for management
  update = True

  def _state_dict (self):
    return {k: getattr(self, k) for k, _ in Option.__dict__.items() if not k.startswith('_')}

  def _parse (self, kwargs):
    state_dict = self._state_dict()
    for k, v in kwargs.items():
      if k not in state_dict:
        raise ValueError('unknown option: "--%s"' % k)
      setattr(self, k, v)

    print('----------- configuration -----------')
    pprint(self._state_dict())
    print('--------------- end -----------------')

opt = Option ()

images_pkl    = join(opt.info_dir, '_images.pkl')
bboxes_pkl    = join(opt.info_dir, '_bboxes.pkl')
captions_pkl  = join(opt.info_dir, '_captions.pkl')
relations_pkl = join(opt.info_dir, '_relations.pkl')

############################################################################################################

def get_id_from_file (f):
  return basename(f).split('_')[0]+'_'+basename(f).split('_')[1]

def get_objects ():
  if not os.path.exists(opt.objects_json):
    print ('object file not exist:', opt.objects_json)
    sys.exit()

  id_object = json.load(codecs.open(opt.objects_json, 'r', 'utf-8-sig'))
  object_id = { object: str(id) for (id, object) in id_object.items() }
  obj_dicts = {k:0 for k in object_id.keys()}

  return id_object, object_id, obj_dicts

def get_name_cls_ins (path):
  name = basename(path).split('.')[0]
  cls  = name.split('(')[0].split('_')[2]
  ins  = name.split('(')[1].split(')')[0]

  return name, cls, ins

def load_json (f):
  return json.load(codecs.open(f, 'r', 'utf-8-sig'))

def get_jsons (dir_, desc):
  return {basename(p).split('.')[0].split('_')[0]+'_'+basename(p).split('.')[0].split('_')[1]:p for p in tqdm(glob(join(dir_, '**/*.json'), recursive=True), desc=desc)}

def put_json (d, f):
  json.dump(d, open(f, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

def put_text (d, f):
  fp = open(f, 'w')
  fp.write('\n'.join(d))
  fp.close()

#forbidden = [
#  '이 사진의 주제는',
#  '표지에',
#  '제목의',
#  '제목이'
#]
forbidden = load_json (opt.forbidden_json)

def is_malformed_sentence (s):
  for f in forbidden:
    if s.find (f)>= 0:
      return True
  return False

replaces = load_json (opt.replace_json)
kor2eng  = load_json (opt.kor2eng_json)
eng2kor  = load_json (opt.eng2kor_json)

def word_refine (ko):
  if ko in kor2eng:
    en = kor2eng[ko]
    if en in eng2kor:
      return eng2kor[en]
  return ko

def word_split (ko, offset):
  if offset !=  0:
    w1 = ko [:-offset]
    w2 = ko [-offset:]
  else:
    w1 = ko
    w2 = ''

  return w1, w2

def word_spacing (ko, offset):
  w1, w2 = word_split (ko, offset)

  if w1 in kor2eng:
    return word_refine (w1) + ' ' + word_refine (w2)
  elif w2 in kor2eng:
    return word_refine (w1) + ' ' + word_refine (w2)

  return ko

def sentence_refine (sentence, flag='pre', lang='ko'):
  for r in replaces[flag]:
    sentence = sentence.replace(r['src'], r['dst'])

  if lang == 'ko':
    ps = sentence.split()
    pl = len (ps)

    for i, p in enumerate (ps):
      if not p in kor2eng:
        length = len (p)
        for offset in range (0, length):
          ps[i] = word_spacing (p, offset)
          if ps[i].find (' ') >= 0:
            break
      else:
        ps[i] = word_refine (p)
    sentence = ' '.join(ps)
  else:
    ps = sentence.split()

    table = str.maketrans('', '', string.punctuation)
    desc = [word.lower() for word in ps]
    desc = [w.translate(table) for w in desc]
    desc = [word for word in desc if len(word)>1]
    desc = [word for word in desc if word.isalpha()]
    sentence = ' '.join(desc)

  return sentence

def merge_bbox (box1, box2):
  x1 = min(box1[0], box2[0])
  y1 = min(box1[1], box2[1])
  x2 = max(box1[2], box2[2])
  y2 = max(box1[3], box2[3])

  return [x1, y1, x2, y2]

def convert_box_wh_xy (b):
  b = list(map(int, b))
  return [b[0], b[1], b[0]+b[2], b[1]+b[3]]

def convert_box_xy_wh (b):
  b = list(map(int, b))
  return [b[0], b[1], b[2]-b[0], b[3]+b[1]]

def convert_box_wh_yx (b):
  b = list(map(int, b))
  return [b[1], b[0], b[1]+b[3], b[0]+b[2]]

ENTITY_LENGTH = 14

def relation_verify (r, o, c, i, offset = True):
  if isinstance(r[o], str) and len(r[o]) < ENTITY_LENGTH:
    d = r[o]
    if d in kor2eng and kor2eng[d] in eng2kor and kor2eng[d] in c:
      l = numpy.where(i == kor2eng[d])[0]
      if offset:
        return kor2eng[d], int(l[0])
      else:
        return kor2eng[d], int(l[len(l)-1])

  return r[o], -1

def class_to_relationships (objs, relations):
  # reduce and drop objecti, non-referenced
  classes = [o['class'] for o in objs]
  indexes = numpy.array(classes)
  idx  = list ()

  for i, r in enumerate (relations):
    sub_id, sub_idx = relation_verify (r, 'entity1', classes, indexes)
    obj_id, obj_idx = relation_verify (r, 'entity2', classes, indexes, False)
    if sub_idx >= 0 and obj_idx >= 0:
      idx.append (sub_idx)
      idx.append (obj_idx)

  idx = list(set(idx))
  idx.sort ()

  objects = list ()
  for i in idx:
    objects.append (objs[i])

  classes = [o['class'] for o in objects]
  indexes = numpy.array(classes)
  _l   = list ()
  _ul  = list ()

  relationships = list ()

  for i, r in enumerate (relations):
    sub_id, sub_idx = relation_verify (r, 'entity1', classes, indexes)
    obj_id, obj_idx = relation_verify (r, 'entity2', classes, indexes, False)
    predicate = r['relation'].lower()
    #_ul = list () # hyhwang?

    if sub_idx >= 0 and obj_idx >= 0:
      d = {'sub_id': sub_idx, 'sub': sub_id, 'predicate': predicate, 'obj_id': obj_idx, 'obj': obj_id}
      relationships.append (d)
      _l.append (i)
    if sub_idx < 0: _ul.append (sub_id)
    if obj_idx < 0: _ul.append (obj_id)

  return relationships, objects, _l, _ul

WEIGHT = 100.0

def inverse_weight (dicts):
  weights = [1.0 / 1 if w == 0 else w for w in list(dicts.values())]
  weights = [w / sum(weights) * WEIGHT for w in weights]

  return { k: weights[i] for i, k in enumerate (dicts)}

def build_caption (l, f, to_text=False):
  with open (images_pkl, 'rb') as fp: images = pickle.load (fp)
  with open (opt.captions_json, 'rb') as fp: captions = json.load (fp)

  data = {images[basename(k).split('_')[0]+'_'+basename(k).split('_')[1]]:captions[basename(k).split('_')[0]+'_'+basename(k).split('_')[1]][:opt.n_caption] for k in l}

  if to_text:
    d = list([v for v in data.values()])
    [x for sublist in d for x in sublist]
    put_json (flatten([v for v in data.values()]), opt.c_text_json)

  put_json (data, f)

############################################################################################################

def load_images (update = False):
  print ('\nLoading images:')

  ids   = dict ()
  dups  = dict ()
  added = 0
  drop  = 0

  for f in tqdm (glob (join (opt.images_dir, '*.jpg'), recursive=True), desc='[+] Loading images from '+opt.images_dir+'/*.jpg'):
    id     = get_id_from_file (f)
    ids[id] = f

  print ('[+] symbolic image files, images(s): {:,}'.format (len (ids)))

  if exists (images_pkl):
    _t = time.process_time ()
    print ('[+] previous saved images file:{} found.'.format (images_pkl))

    with open (images_pkl, 'rb') as f: images = pickle.load (f)
    _elapsed = time.process_time () - _t
    print ('[+] loading file:{}, elapsed: {:.2f} sec, # {:,}'.format (images_pkl, _elapsed, len (images)))
    #print (images[list(images.keys())[0]])
  else: images = dict ()

  if len(images) > 0 and not update: return images

  for f in tqdm (glob (join (opt._images_dir, '**/*.jpg'), recursive=True), desc='{+] loading images from '+opt._images_dir+'/**/*.jpg'):
    id     = get_id_from_file (f)
    target = '../' + relpath (f)
    link   = join (opt.images_dir, basename (f))

    if not id in dups: dups[id] = -1
    dups[id] += 1

    if not id in ids: 
      if exists (link): os.remove (link)
      os.symlink (target, link) # symbolic link
    if not id in images: added += 1
    images[id] = link

  print ('[+] ended. images: {:,}, added: {:,}, dup: {:,}'.format (len (images), added, sum(dups.values())))

  if added > 0:
    print ('[+] save files. images: {:,}, added: {:,}'.format (len (images), added))
    with open (images_pkl, 'wb') as f: pickle.dump (images, f)

  return images

def load_bboxes (images, update = False):
  print ('\nLoading bboxes:')
  added = 0
  dups  = dict ()
  drop  = 0

  if exists (bboxes_pkl):
    _t = time.process_time ()
    print ('[+] previous saved bboxes file:{} found.'.format (bboxes_pkl))
    with open (bboxes_pkl, 'rb') as f: bboxes = pickle.load (f)
    _elapsed = time.process_time () - _t
    print ('[+] loading file:{}, elapsed: {:.2f} sec, # {:,}'.format (bboxes_pkl, _elapsed, len (bboxes)))
    #print (bboxes[list(bboxes.keys())[0]])
  else: bboxes = dict ()

  if len(bboxes) > 0 and not update: return bboxes

  for f in tqdm (glob (join (opt._bboxes_dir, '**/*.json'), recursive=True), desc='{+] loading bboxes from '+opt._bboxes_dir+'/**/*.json'):
    id = get_id_from_file (f)

    if not id in dups: dups[id] = -1
    dups[id] += 1

    if not id in images: # image file not found
      drop += 1
      continue

    if not id in bboxes:
      _b_ = json.load (codecs.open (f, 'r', 'utf-8-sig'))

      added += 1
      bboxes[id] = dict ()
      bboxes[id]['images']      = _b_['images']
      bboxes[id]['annotations'] = _b_['annotations']

  print ('[+] ended. bboxes: {:,}, added: {:,}, drop: {:,}, dup: {:,}'.format (len (bboxes), added, drop, sum(dups.values())))

  if added > 0:
    print ('[+] save files. bboxes: {:,}, added: {:,}'.format (len (bboxes), added))
    with open (bboxes_pkl, 'wb') as f: pickle.dump (bboxes, f)

  return bboxes

def load_captions (images, update = False):
  print ('\nLoading captions:')

  added = 0
  dups  = dict ()
  drop  = 0

  if exists (captions_pkl):
    _t = time.process_time ()
    print ('[+] previous saved captions file:{} found.'.format (captions_pkl))
    with open(captions_pkl, 'rb') as f: captions = pickle.load(f)
    _elapsed = time.process_time () - _t
    print ('[+] loading file:{}, elapsed: {:.2f} sec, # {:,}'.format (captions_pkl, _elapsed, len (captions)))
    #print (captions[list(captions.keys())[0]])
  else: captions = dict ()

  if len(captions) > 0 and not update: return captions

  for f in tqdm(glob(join(opt._captions_dir, '**/*.json'), recursive=True), desc='{+] loading captions from '+opt._captions_dir+'/**/*.json'):
    id = get_id_from_file (f)

    if not id in dups: dups[id] = -1
    dups[id] += 1

    if not id in images: # image file not found
      drop += 1
      continue

    if not id in captions:
      _c_ = json.load(codecs.open(f, 'r', 'utf-8-sig'))

      added += 1
      captions[id] = dict ()
      captions[id]['text'] = list([]) if 'annotations' not in _c_ else _c_['annotations'][0]['text']

  print ('[+] ended. captions: {:,}, added: {:,}, drop: {:,}, dup: {:,}'.format (len (captions), added, drop, sum(dups.values())))

  if added > 0:
    print ('[+] save files. captions: {:,}, added: {:,}'.format (len (captions), added))
    with open(captions_pkl, 'wb') as f: pickle.dump (captions, f)

  return captions

def load_relations (images, bboxes, update = False):
  print ('\nLoading relations:')

  added = 0
  dups  = dict ()
  drop  = 0

  if exists (relations_pkl):
    _t = time.process_time ()
    print ('[+] previous saved relations file:{} found.'.format (relations_pkl))
    with open(relations_pkl, 'rb') as f: relations = pickle.load(f)
    _elapsed = time.process_time () - _t
    print ('[+] loading file:{}, elapsed: {:.2f} sec, # {:,}'.format (relations_pkl, _elapsed, len (relations)))
    #print (relations[list(relations.keys())[0]])
  else: relations = dict ()

  if len(relations) > 0 and not update: return relations

  for f in tqdm(glob(join(opt._relations_dir, '**/*.json'), recursive=True), desc='{+] loading relations from '+opt._relations_dir+'/**/*.json'):
    id = get_id_from_file (f)

    if not id in dups: dups[id] = -1
    dups[id] += 1

    if not id in images: # image file not found
      drop += 1
      continue

    if not id in bboxes: # bbox file not found
      drop += 1
      continue

    if not id in relations:
      _r_ = json.load(codecs.open(f, 'r', 'utf-8-sig'))

      added += 1
      relations[id] = dict ()
      relations[id]['text'] = list([]) if 'annotations' not in _r_ else _r_['annotations'][0]['text']

  print ('[+] ended. relations: {:,}, added: {:,}, drop: {:,}, dup: {:,}'.format (len (relations), added, drop, sum(dups.values())))

  if added > 0:
    print ('[+] save files. relations: {:,}, added: {:,}'.format (len (relations), added))
    with open(relations_pkl, 'wb') as f: pickle.dump (relations, f)

  return relations

def load_instances (images, bboxes, captions, relations, update = False):
#  if exists (images_pkl):
#    _t = time.process_time ()
#    print ('[+] previous saved instances file:{} found.'.format (_instances_pkl))
#
#    with open (_instances_pkl, 'rb') as f: instances = pickle.load (f)
#    _elapsed = time.process_time () - _t
#    print ('[+] loading file:{}, elapsed: {:.2f} sec, # {:,}'.format (_instances_pkl, _elapsed, len (instances)))
#    #print (instances[list(instances.keys())[0]])
#  else: instances = dict ()

  instances  = dict ()

  dicts      = dict ()
  dicts['idx2word'] = list ()

  categories = dict ()
  inv_weight = dict ()
  pred_dicts = dict ()
  obj_dicts  = dict ()
  word_dicts = dict ()
  caps       = dict ()
  unalias    = dict ()
  cls_files  = dict ()
  obj_files  = dict ()

  drop1  = 0
  drop2  = 0
  drop3  = 0
  drop4  = 0
  drop5  = 0
  drop6  = 0
  drop7  = 0
  drop8  = 0
  drop9  = 0
  drop10 = 0

  id_object, object_id, obj_dicts = get_objects ()

  print ('id_object:', len(id_object))
  print ('object_id:', len(object_id))
  print ('obj_dicts:', len(obj_dicts))

  for i, k in enumerate(tqdm(bboxes, desc='[+] build instances with bboxes')):
    if not k in images:    
      drop1 += 1
      continue
    if not k in captions:  
      drop2 += 1
      continue
    if not k in relations: 
      drop3 += 1
      continue

    _b_ = bboxes[k]
    _c_ = captions[k]
    _r_ = relations[k]

    if len (_c_ ['text']) < opt.n_caption: 
      drop4 += 1
      continue
    if len (_r_ ['text']) < opt.n_caption: 
      drop5 += 1
      continue


    if not k in instances:
      instances[k] = dict()

      instances[k]['height'] = _b_['images'][0]['height']
      instances[k]['width']  = _b_['images'][0]['width']
      instances[k]['path']   = basename(images[k])

      instances[k]['image'] = images[k]

      instances[k]['relationships'] = list ()
      instances[k]['regions']       = list ()
      instances[k]['objects']       = list ()
      instances[k]['captions']      = dict ()

      instances[k]['_b_'] = [{'box': o['bbox'], 'id': o['category_id'], 'class': id_object[str(o['category_id'])]} for o in _b_['annotations']]
      instances[k]['_c_'] = _c_ ['text']
      instances[k]['_r_'] = _r_ ['text']

      if len (instances[k]['_b_']) == 0:
        del instances[k]
        continue

      if len (instances[k]['_c_']) == 0:
        del instances[k]
        continue

      if len (instances[k]['_r_']) == 0:
        del instances[k]
        continue


    if sum ([ 1 if o['korean']=='' else 0 for o in instances[k]['_c_']]) > 0:
      drop6 += 1
      del instances[k]
      continue
    if sum ([ 1 if o['english']=='' else 0 for o in instances[k]['_c_']]) > 0:
      drop7 += 1
      del instances[k]
      continue

    name, cls, ins = get_name_cls_ins (images[k])

    ## check hyhwang (rel_cnt)
    ## objects
    #instances[k]['objects'] = [{'box': convert_box_wh_xy (o['box']), 'class': o['class']} for o in instances[k]['_b_']]
    instances[k]['relationships'], objects, _l, _ul  = class_to_relationships (instances[k]['_b_'], instances[k]['_r_'])
    instances[k]['objects'] = [{'box': convert_box_wh_xy (o['box']), 'class': o['class']} for o in objects]

    for d in instances[k]['relationships']:
      pred = d['predicate']
      sub  = d['sub']
      obj  = d['obj']
      if not pred in pred_dicts: ## hyhwang
        pred_dicts [pred] = 0

      pred_dicts [pred] += 1
      obj_dicts [sub] += 1
      obj_dicts [obj] += 1

    for u in _ul:
      if not u in unalias:
        unalias[u] = 0
      unalias[u] += 1

    ## captions
    ## relations & regions
    for i, _i in enumerate (_l):
      ko = instances[k]['_r_'][_i]['korean']
      ko = sentence_refine (ko,  'pre', 'ko')
      ko = sentence_refine (ko, 'post', 'ko')

      en = instances[k]['_r_'][_i]['english']
      en = sentence_refine (en,  'pre', 'en')
      en = sentence_refine (en, 'post', 'en')

      region = dict ()
      sub  = instances[k]['relationships'][i]['sub_id']
      obj  = instances[k]['relationships'][i]['obj_id']
      pred = instances[k]['relationships'][i]['predicate']
      box1 = instances[k]['objects'][sub]['box']
      box2 = instances[k]['objects'][obj]['box']
      region['box'] = merge_bbox (box1, box2)

      if opt.lang == 'ko':
        region['phrase']  = ko.split ()
        for p in region['phrase']:
          dicts['idx2word'].append (p)
          if not p in word_dicts: word_dicts[p] = 0
          word_dicts[p] += 1
        region['phrase2'] = en.split ()
      else:
        region['phrase']  = en.split ()
        for p in region['phrase']:
          dicts['idx2word'].append (p)
          if not p in word_dicts: word_dicts[p] = 0
          word_dicts[p] += 1
        region['phrase2'] = ko.split ()

      instances[k]['regions'].append (region)

    if len (instances[k]['relationships']) == 0:
      del instances[k]
      continue

    if len (instances[k]['regions']) == 0:
      del instances[k]
      continue

    ###########################################################
    #
    # normal condition.
    #
    ko43 = [ c['korean']  for c in instances[k]['_c_']]
    en43 = [ c['english'] for c in instances[k]['_c_']]
#    if is_malformed_sentence (' '.join(ko43)):
#      del instances[k]
#      drop8 += 1
#      continue

    ko44 = [ c['korean']  for c in instances[k]['_r_']]
    en44 = [ c['english'] for c in instances[k]['_r_']]
    if is_malformed_sentence (' '.join(ko44)):
      del instances[k]
      drop9 += 1
      continue

    if len(ko43) != opt.n_caption or len(en43) != opt.n_caption or len(ko44) != opt.n_caption or len(en44) != opt.n_caption:
      del instances[k]
      drop10 += 1
      continue

    if not cls in cls_files:
      cls_files[cls] = list()
    cls_files[cls].append(name)

    for obj in [ o['class'] for o in instances[k]['_b_']]:
      if not obj in obj_files:
        obj_files[obj] = list()
      obj_files[obj].append(name)

    ########################################################

    if opt.using == 3:
      korean  = ko43
      english = en43
    else:
      korean  = ko44
      english = en44

    instances[k]['captions']['korean'] = list ()
    for ko in korean:
      ko = sentence_refine (ko,  'pre', 'ko')
      ko = sentence_refine (ko, 'post', 'ko')
      instances[k]['captions']['korean'].append (ko)

    instances[k]['captions']['english'] = list ()
    for en in english:
      en = sentence_refine (en,  'pre', 'en')
      en = sentence_refine (en, 'post', 'en')
      instances[k]['captions']['english'].append (en)

    if opt.lang == 'ko':
      caps[k] = instances[k]['captions']['korean']
    else:
      caps[k] = instances[k]['captions']['english']


  ###############################################################################

  dicts['idx2word'] = list(set(dicts['idx2word']))
  dicts['word2idx'] = { k:i for i, k in enumerate (dicts['idx2word']) }

  files     = [basename(v['image']).split('.')[0] for k, v in instances.items()]
  obj_files = { k:list(set(v)) for k, v in obj_files.items()}
  cls_stats = { k: len(cls_files[k]) for k in cls_files}
  cls_stats = dict(sorted(cls_stats.items(), key=lambda item: item[1], reverse=True))
  obj_stats = { k: len(obj_files[k]) for k in obj_files}
  obj_stats = dict(sorted(obj_stats.items(), key=lambda item: item[1], reverse=True))
  unalias   = dict(sorted(unalias.items(), key=lambda item: item[1], reverse=True))

  print ('instances:', len(instances), 'index:', i, 'sum:', drop1+drop2+drop3+drop4+drop5+drop6+drop7+drop8+drop9+drop10, 'drop:', drop1, drop2, drop3, drop4, drop5, drop6, drop7, drop8, drop9, drop10)

  pred_dicts = dict(sorted(pred_dicts.items(), key=lambda item: item[1], reverse=True))
  obj_dicts  = dict(sorted(obj_dicts.items(), key=lambda item: item[1], reverse=True))
  word_dicts = dict(sorted(word_dicts.items(), key=lambda item: item[1], reverse=True))

  categories['predicate'] = list(pred_dicts.keys())
  categories['object']    = list(object_id.keys())

  inv_weight['predicate'] = inverse_weight (pred_dicts)
  inv_weight['object']    = inverse_weight (obj_dicts)

  put_json (files,      opt.files_json)
  put_text (files,      opt.files_text)
  put_json (instances,  opt.instances_json)
  put_json (object_id,  opt.object_id_json)
  put_json (id_object,  opt.id_object_json)
  put_json (cls_files,  opt.cls_files_json)
  put_json (obj_files,  opt.obj_files_json)
  put_json (cls_stats,  opt.cls_stats_json)
  put_json (obj_stats,  opt.obj_stats_json)
  put_json (caps,       opt.captions_json)
  put_json (unalias,    opt.unalias_json)
  put_json (dicts,      opt.dicts_json)
  put_json (pred_dicts, opt.pred_dicts_json)
  put_json (obj_dicts,  opt.obj_dicts_json)
  put_json (word_dicts, opt.word_dicts_json)
  put_json (inv_weight, opt.inv_weight_json)
  put_json (categories, opt.categories_json)


  return instances

#################################################################################

def prepare (**kwargs):
  opt._parse(kwargs)

  images    = load_images ()
  bboxes    = load_bboxes (images)
  captions  = load_captions (images)
  relations = load_relations (images, bboxes)

  print ('\n-------------------------------------')
  print ('images   : {:,}'.format(len(images)   ))
  print ('bboxes   : {:,}'.format(len(bboxes)   ))
  print ('captions : {:,}'.format(len(captions) ))
  print ('relations: {:,}'.format(len(relations)))
  #pprint(images  [list(images.keys())[0]]  )
  #pprint(bboxes  [list(bboxes.keys())[0]]  )
  #pprint(captions[list(captions.keys())[0]])
  #pprint(relations[list(relations.keys())[0]])

  instances = load_instances (images, bboxes, captions, relations)
  #print ('\n-------------------------------------')
  #print ('instances: {:,}'.format(len(instances)   ))
  #pprint(instances[list(instances.keys())[0]])

def build (**kwargs):
  opt._parse(kwargs)

  cls_files = load_json (opt.cls_files_json)
  instances = load_json (opt.instances_json)

  trainvaltest = [[ l for l in v[:opt.num_per_class] if len(v) >= opt.min_per_class] for k, v in cls_files.items()]
  trainvaltest = [x for sublist in trainvaltest for x in sublist]
  random.shuffle(trainvaltest)

  trainvaltest = trainvaltest[:opt.limits]

  total = len(trainvaltest)
  size1 = int(total * opt.split[0])
  size2 = int(total * (opt.split[0]+opt.split[1]))

  train    = trainvaltest[:size1]
  val      = trainvaltest[size1:size2]
  test     = trainvaltest[size2:]
  trainval = train + val

  put_text (trainvaltest, opt.trainvaltest_text)
  put_text (trainval, opt.trainval_text)
  put_text (train, opt.train_text)
  put_text (val, opt.val_text)
  put_text (test, opt.test_text)

  build_caption (trainvaltest, opt.c_trainvaltest_json, True)
  build_caption (trainval, opt.c_trainval_json)
  build_caption (train, opt.c_train_json)
  build_caption (val, opt.c_val_json)
  build_caption (test, opt.c_test_json)

  _trainvaltest = list ()
  _trainval     = list ()
  _train        = list ()
  _val          = list ()
  _test         = list ()

  _trainvaltest = list([instances[k.split('_')[0]+'_'+k.split('_')[1]] for k in trainvaltest])
  _trainval     = list([instances[k.split('_')[0]+'_'+k.split('_')[1]] for k in trainval])
  _train        = list([instances[k.split('_')[0]+'_'+k.split('_')[1]] for k in train])
  _val          = list([instances[k.split('_')[0]+'_'+k.split('_')[1]] for k in val])
  _test         = list([instances[k.split('_')[0]+'_'+k.split('_')[1]] for k in test])

  fat   = 0.5
  small = 0.1

  _train_fat    = _train[:int(len(train)*fat)  ]
  _val_fat      = _val  [:int(len(val)*fat)    ]
  _test_fat     = _test [:int(len(test)*fat)   ]
  _train_small  = _train[:int(len(train)*small)]
  _val_small    = _val  [:int(len(val)*small)  ]
  _test_small   = _test [:int(len(test)*small) ]

  put_json (_trainvaltest, opt.trainvaltest_json)
  put_json (_trainval, opt.trainval_json)

  put_json (_train, opt.train_json)
  put_json (_val,   opt.val_json  )
  put_json (_test,  opt.test_json )

  put_json (_train_fat, opt.train_fat_json)
  put_json (_val_fat,   opt.val_fat_json  )
  put_json (_test_fat,  opt.test_fat_json )

  put_json (_train_small, opt.train_small_json)
  put_json (_val_small,   opt.val_small_json  )
  put_json (_test_small,  opt.test_small_json )

  print ('trainvaltest(txt) :', len(trainvaltest))
  print ('train       (txt) :', len(train))
  print ('val         (txt) :', len(val))
  print ('trainval    (txt) :', len(trainval))
  print ('test        (txt) :', len(test))

  print ('trainvaltest(json):', len(_trainvaltest))
  print ('train       (json):', len(_train))
  print ('val         (json):', len(_val))
  print ('trainval    (json):', len(_trainval))
  print ('test        (json):', len(_test))

if __name__ == '__main__':
  _t = time.process_time()

  import fire
  fire.Fire()

  _elapsed = time.process_time() - _t

  print ('')
  print ('Total elapsed: {:.2f} sec'.format(_elapsed))

