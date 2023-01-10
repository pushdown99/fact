import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel as DataParallel_raw
import numpy as np


class DataParallel (DataParallel_raw):
    """
    we do the scatter outside of the DataPrallel.
    input: Scattered Inputs without kwargs.
    """

    def __init__ (self, module):
        # Disable all the other parameters
        #print ('[-] DataParallel __init__')
        super (DataParallel, self).__init__ (module)


    def forward (self, *inputs, **kwargs):
        #print ('[-] DataParallel forward')
        assert len (inputs) == 0, "Only support arguments like [variable_name = xxx]"
        new_inputs = [{} for _ in self.device_ids]

        for key in kwargs:
            if key == 'im_data':
                for i, device in enumerate (self.device_ids):
                    new_inputs[i][key] = kwargs[key][i].to (device)

            elif key.startswith ("rpn_anchor_targets"):
                for i, device in enumerate (self.device_ids):
                    new_inputs[i][key] = [item.to (device) for item in kwargs[key][i]]
                
            else:
                assert isinstance (kwargs[key], list)
                for i in range (len (self.device_ids)):
                    new_inputs[i][key] = [kwargs[key][i], ]

        #replicas = nn.parallel.replicate(module, device_ids)
        #inputs = nn.parallel.scatter(input, device_ids)
        #replicas = replicas[:len(inputs)]
        #outputs = nn.parallel.parallel_apply(replicas, inputs)
        #return nn.parallel.gather(outputs, output_device)

        nones = [[] for _ in self.device_ids]
        replicas = self.replicate (self.module, self.device_ids)
        outputs = self.parallel_apply (replicas, nones, new_inputs)
        #print ('[-] DataParallel device_ids: {}, output_device: {}'.format(self.device_ids, self.output_device))

        #hyhwang
        return self.gather (outputs, self.output_device)
        #UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
        #logit = self.gather (outputs, self.output_device)
        #print ('[-] DataParallel forward end.')
        #return logit
