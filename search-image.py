from __future__ import  absolute_import
import os
from os.path import join
from pprint import pprint


import time
from flickrapi import FlickrAPI
from six.moves import urllib

from lib.utils.image import Display

FLICKR_PUBLIC = 'c6a2c45591d4973ff525042472446ca2'
FLICKR_SECRET = '202ffe6f387ce29b'

class Config:
  keyword = 'kitty'
  
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
    return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
      if not k.startswith('_')}

opt = Config()

def search (**kwargs):
  opt._parse(kwargs)

  flickr = FlickrAPI(FLICKR_PUBLIC, FLICKR_SECRET, format='parsed-json')
  extras='url_sq,url_t,url_s,url_q,url_m,url_n,url_z,url_c,url_l,url_o'
  res = flickr.photos.search(text=opt.keyword, per_page=5, extras=extras)
  #res = flickr.photos.search(text='kitchen plate', per_page=5, extras=extras)
  photos = res['photos']

  urls = []
  for i, photo in enumerate(photos['photo']):
    url = photo.get('url_c')
    urls.append (url)

  filename = join('samples',opt.keyword+'.jpg')

  print ('Found {} from {}, save to {}'.format(opt.keyword, urls[0], filename))
  urllib.request.urlretrieve(urls[0], filename)
  Display (filename)

# Resize the image and overwrite it
#image = Image.open('00001.jpg')
#image = image.resize((256, 256), Image.ANTIALIAS)
#image.save('00001.jpg')

if __name__ == '__main__':
  _t = time.process_time()

  import fire
  fire.Fire()

  _elapsed = time.process_time() - _t
  print ('')
  print ('elapsed: {:.2f} sec'.format(_elapsed))

