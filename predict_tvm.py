import numpy as np
import tvm
from tvm import relay
from tvm.contrib import graph_runtime
import torch
from model import AlexNet
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json
import time

data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# load image
img = Image.open("sunflowers.jpeg")
plt.imshow(img)
# [N, C, H, W]
img = data_transform(img)
# expand batch dimension
img = torch.unsqueeze(img, dim=0)
x = np.array(img)
# read class_indict
try:
    json_file = open('class_indices.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

# create model
model = AlexNet(num_classes=5)
# load model weights
model_weight_path = "./AlexNet_weights.pth"
model.load_state_dict(torch.load(model_weight_path))
# model = torch.load('AlexNet.pth')
model.eval()
scripted_model = torch.jit.trace(model, img).eval()
input_name = 'data'
shape_list = [
    (input_name, x.shape)]
mod, params = relay.frontend.from_pytorch(scripted_model, input_shapes=shape_list)

# Build relay
target = 'llvm'
target_host = 'llvm'

ctx = tvm.cpu(0)
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, target_host=target_host, params=params)

dtype = "float32"
m = graph_runtime.GraphModule(lib["default"](ctx))
# Set inputs
m.set_input(input_name, tvm.nd.array(x.astype(dtype)))
# Execute
start = time.perf_counter()
m.run()
elapsed = (time.perf_counter() - start)
print("Time used:",elapsed)
# Get outputs
tvm_output = m.get_output(0)

output_array = tvm_output.asnumpy()[0]
output_array_softmax = torch.softmax(torch.tensor(output_array), dim=0)
print(output_array_softmax)
top_tvm = torch.argmax(output_array_softmax).numpy()
print(class_indict[str(top_tvm)],output_array_softmax[top_tvm])