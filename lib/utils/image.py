from IPython.display import Image, display

import numpy as np
import matplotlib.pyplot as plt
import climage

def is_notebook() -> bool:
  try:
    shell = get_ipython().__class__.__name__
    if shell == 'ZMQInteractiveShell':
      return True   # Jupyter notebook or qtconsole
    elif shell == 'TerminalInteractiveShell':
      return False  # Terminal running IPython
    else:
      return False  # Other type (?)
  except NameError:
    return False    # Probably standard Python interpreter

def Display (file):
  if is_notebook():
    display(Image(file))
  else:
    out = climage.convert(file, is_unicode=True)
    print (out)

