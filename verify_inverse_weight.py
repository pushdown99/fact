import torch
import json
from os.path import join, basename

dataset = 'nia'
data_path = join(dataset, 'info')

inverse_weight = json.load (open (join (data_path, 'inverse_weight.json')))
cats           = json.load (open (join (data_path, 'categories.json')))
dictionary     = json.load (open (join (data_path, 'dict.json')))

_object_classes    = tuple (['__background__'] + cats['object'])
_predicate_classes = tuple (['__background__'] + cats['predicate'])

print ()
print ('_object_classes:', len(_object_classes), '_predicate_classes:', len(_predicate_classes))

_object_class_to_ind    = dict (zip (_object_classes, range (len(_object_classes))))
_predicate_class_to_ind = dict (zip (_predicate_classes, range (len(_predicate_classes))))

inverse_weight_object = torch.ones (len(_object_classes))
for idx in range (1, len(_object_classes)):
  inverse_weight_object[idx] = inverse_weight['object'][_object_classes[idx]]

inverse_weight_object = inverse_weight_object / inverse_weight_object.min ()
print (inverse_weight_object)
print (inverse_weight_object.min())
print (inverse_weight_object.max())

inverse_weight_predicate = torch.ones (len(_predicate_classes))
for idx in range (1, len(_predicate_classes)):
  inverse_weight_predicate[idx] = inverse_weight['predicate'][_predicate_classes[idx]]
inverse_weight_predicate = inverse_weight_predicate / inverse_weight_predicate.min ()
print (inverse_weight_predicate)
print (inverse_weight_predicate.min())
print (inverse_weight_predicate.max())



if dataset == 'nia':
  _pred_dicts = json.load (open (join (data_path, '_pred_dicts.json')))
  sum = 0
  for k in _pred_dicts:
    sum += _pred_dicts[k]
  print ('predicate sum:', '{:,} {:,}'.format(sum, 2* sum))

  _obj_dicts = json.load (open (join (data_path, '_obj_dicts.json')))
  sum = 0
  for k in _obj_dicts:
    sum += _obj_dicts[k]
  print ('object sum   :', '{:,}'.format(sum))



