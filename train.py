import torch
from torchvision import transforms, utils, datasets
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from model import AlexNet
import time
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

data_root = os.path.abspath(os.getcwd())  # get data root path
image_path = data_root + "/flower_data/"  # flower data set path
train_dir = image_path + "train"
validation_dir = image_path + "val"

transform = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    "valid": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
}

train_data_set = datasets.ImageFolder(
    train_dir, transform=transform["train"])
train_num = train_data_set.__len__()
flower_list = train_data_set.class_to_idx
class_names = dict((val, key) for key, val in flower_list.items())
json_str = json.dumps(class_names, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)
print(class_names)

batch_size = 32
train_data_loader = torch.utils.data.DataLoader(
    train_data_set, batch_size=batch_size, shuffle=True)

valid_data_set = datasets.ImageFolder(
    validation_dir, transform=transform["valid"])
valid_data_loader = torch.utils.data.DataLoader(
    valid_data_set, batch_size=batch_size, shuffle=False
)
test_data_iter = iter(valid_data_loader)
test_image, test_label = test_data_iter.next()


# def imshow(img):
#     img = img / 2 + 0.5  # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()


# print(' '.join('%5s' % class_names[test_label[j].item()]
#                for j in range(len(test_label))))
# imshow(utils.make_grid(test_image))

epochs = 10
model = AlexNet(num_classes=5)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
best_acc = 0.0

for epoch in range(epochs):
    # Train model
    model.train()
    running_loss = 0.0
    t = time.perf_counter()
    for index, data in enumerate(train_data_loader):
        imgs, labels = data
        outputs = model(imgs)
        optimizer.zero_grad()
        loss = loss_function(outputs, labels)
        running_loss += loss
        loss.backward()
        optimizer.step()

        rate = index / train_data_loader.__len__()
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print(
            "\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
    print("\n time.perf_counter()-t1")

    model.eval()
    acc = 0.0
    with torch.no_grad():
        for data in valid_data_loader:
            imgs, labels = data
            outputs = model(imgs)
            acc += (torch.max(outputs, dim=1)[1] == labels).sum().item()
        acc = acc / valid_data_loader.dataset.__len__()
        if acc > best_acc:
            best_acc = acc
            print("Saving Model")
            torch.save(model.state_dict(), 'AlexNet_weights.pth')
            torch.save(model, 'AlexNet.pth')
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
              (epoch + 1, running_loss, acc))

print('Finished Training')
