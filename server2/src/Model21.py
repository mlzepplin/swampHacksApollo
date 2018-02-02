import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

transform = transforms.Compose(
    [transforms.Resize([224, 224]), transforms.ToTensor(),
     transforms.Normalize(
     mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229])])
trainpath = './Train'
testpath = './Test'
trainset = datasets.ImageFolder(root=trainpath, transform=transform)
testset = datasets.ImageFolder(root=testpath, transform=transform)
trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)
testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)

use_gpu = torch.cuda.is_available()
# use_gpu = False
classes = trainset.classes
print(classes)
print(testset.classes)


def imshow(img, title=None):
    img = img.numpy().transpose((1, 2, 0))
    img = img * 0.229 + 0.485  # unnormalize
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.show()


net = models.resnet34(pretrained=False)
num_features = net.fc.in_features
net.fc = nn.Linear(num_features, 2)

if use_gpu:
    net = net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

modelpath = './server2/src/Model21.json'
from_file = True
if from_file:
    net.load_state_dict(torch.load(modelpath))
    print('read')

# training
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get inputs and labels
        inputs, labels = data
        # wrap in variables
        if use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs = Variable(inputs)
            labels = Variable(labels)
        # zero parameter gradients
        optimizer.zero_grad()
        # forward, backward, update
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # add to running loss
        running_loss += loss.data[0]
        # print progress every minibatch
        print('[%d, %d] loss: %f' % (epoch + 1, i + 1, running_loss / (i + 1)))

# dataiter to fetch 4 of images
dataiter = iter(testloader)
for i in range(3):
    images, labels = dataiter.next()
    images_cpu = images
    if use_gpu:
        images = images.cuda()
        labels = labels.cuda()
    # predict classes of images
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    # show images
    imshow(utils.make_grid(images_cpu), title=[classes[x] for x in labels])
    print([classes[x] for x in labels])
    # display predictions
    print('Predicted: ', ' '.join('%5s' %
                                  classes[j] for j in predicted))

# testing
class_correct = list(0 for i in range(len(classes)))
class_total = list(0 for i in range(len(classes)))
for data in testloader:
    images, labels = data
    if use_gpu:
        images = images.cuda()
        labels = labels.cuda()
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted == labels).squeeze()
    for i in range(labels.size()[0]):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1

for i in range(len(classes)):
    print('Accuracy of %5s : %f %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

to_file = True
if to_file:
    torch.save(net.state_dict(), modelpath)
    print('write')
