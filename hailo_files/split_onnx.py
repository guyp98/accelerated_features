#!/usr/bin/env python3

import onnx
from google.protobuf.json_format import MessageToDict
import onnxruntime
import torch
import numpy as np
import time
from zenlog import log
import argparse


parser = argparse.ArgumentParser(description='Running an ONNX file with ONNXRuntime package')
parser.add_argument('--onnx-path', help="ONNX file path to be ran")
parser.add_argument('--onnx-output', help="In case you want to create a subgraph, insert the wanted output ONNX name")
parser.add_argument('--subgraph-input', nargs='+', help="input layer name of the subgraph")
parser.add_argument('--subgraph-output', nargs='+', help="output layer name of the subgraph")
parser.add_argument('--image-num', help="Number of images to run inference on. Defaults to 1000 images")
args = parser.parse_args()


# -------- Load / Create a sub model --------------
model_path = args.onnx_path
images_num = 50

print(args.onnx_path)

if (args.image_num):
  images_num = int(args.image_num)

if (args.onnx_output):
  if (not args.subgraph_input or not args.subgraph_output):
    raise ValueError('You must define input and ouput nodes to create a subgraph')
# NOTICE: the name of the input\output layer is often the name that is under the INPUTS\OYTOUTS knob in Netron
  output_path = args.onnx_output
  input_names = args.subgraph_input
  output_names = args.subgraph_output
  onnx.utils.extract_model(model_path, output_path, input_names, output_names)
  onnx_model = onnx.load(output_path)
  model_path = args.onnx_output
else:
  onnx_model = onnx.load(model_path)


# -------- Check model validity --------------
try:
    onnx.checker.check_model(onnx_model)
except onnx.checker.ValidationError as e:
    log.info('The model is invalid: %s' % e)
else:
    log.info('The model is valid!')

# -------- Get model input shapes --------------
model_inputs = {}
model_shapes = []
for i, input in enumerate(onnx_model.graph.input):
  m_dict = MessageToDict(input)
  dim_info = m_dict.get("type").get("tensorType").get("shape").get("dim")
  shape = [d.get("dimValue") for d in dim_info]
  log.info(f'Model input {i+1} shape is: {shape}')
  batch_size, channels, height, width = shape
  if (height == None or width == None or channels == None):
    raise ValueError('The model must have a defined resolution')
  elif (batch_size == None):
    model_inputs[input.name] = np.array(torch.rand(1, int(channels), int(height), int(width)))
  else:
    model_inputs[input.name] = np.array(torch.rand(int(batch_size), int(channels), int(height), int(width)))
    

# -------- Run inference --------------    
start_time = time.time()
sess = onnxruntime.InferenceSession(model_path)

print('\n            Running Inference...')
for _ in range(images_num): # simulates inference of the images
  output = sess.run(None, model_inputs)
  
end_time = time.time()
print('            Inference was successful!\n')
log.info('-------------------------------------')
log.info(' Infer Time:      {:.3f} sec'.format(end_time - start_time))
log.info(' Average FPS:     {:.3f}'.format(images_num/(end_time - start_time)))
log.info(' Latency:         {:.3f} ms'.format(1/(images_num/(end_time - start_time))*1000))
log.info('-------------------------------------')

