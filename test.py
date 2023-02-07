import json

l = [
 {'sub': 'table', 'pred': 'on', 'obj': 'person'},
 {'sub': 'table', 'pred': 'on', 'obj': 'person'},
 {'sub': 'table', 'pred': 'on', 'obj': 'person'},
 {'sub': 'table', 'pred': 'on', 'obj': 'person'},
 {'sub': 'table', 'pred': 'on', 'obj': 'person'},
 {'sub': 'table', 'pred': 'on', 'obj': 'person'},
 {'sub': 'table', 'pred': 'on', 'obj': 'person'}
]

print (l)
l = list(set(list(map(tuple, l))))
print (json.dumps(l))
