import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

#from openvino.inference_engine import IECore
from openvino.runtime import Core

class BCC():

    def __init__(self):
        self.compiled_model_ir = self.get_model()

    def get_model(self):

        ir_path = 'bcc_model.xml'
        # Load the network in Inference Engine
        ie = Core()
        model_ir = ie.read_model(model=ir_path)
        compiled_model = ie.compile_model(model=model_ir, device_name="CPU")
        
        return compiled_model

    def get_crowd_count(self, image):
        image = cv2.resize(image, (1280, 720),
               interpolation = cv2.INTER_NEAREST)
       
        input_image = np.expand_dims(np.transpose(image, (2, 0, 1)), 0)
        
        # Get input and output layers
        # output_layer_ir = self.compiled_model_ir.output(0)
        output_layer_ir = self.compiled_model_ir.output(0)

        # Run inference on the input image
        res_ir = self.compiled_model_ir([input_image])[output_layer_ir]

        crowd_count = int(round(torch.sum(torch.from_numpy(res_ir)).item(),0))
        
        return crowd_count

vid_dir = 'dining_hall_np_sp'
save_dir = 'bcc_results_np_sp'

img = cv2.imread('sample.jpeg')

bcc = BCC()
count = bcc.get_crowd_count(img)
print(count)
