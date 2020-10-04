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
img = Image.open("images/sunflowers.jpeg")
plt.imshow(img)
# [N, C, H, W]
img = data_transform(img)
# expand batch dimension
img = torch.unsqueeze(img, dim=0)

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
model.eval()

start = time.perf_counter()
with torch.no_grad():
    # predict class
    output = torch.squeeze(model(img))
elapsed = (time.perf_counter() - start)
print("Time used:",elapsed)
print(output)
predict = torch.softmax(output, dim=0)
predict_cla = torch.argmax(predict).numpy()

print(class_indict[str(predict_cla)], predict[predict_cla].item())
