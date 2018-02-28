################################################################################
##                                                                            ##
##      AVALANCHES                                                            ##
##     ----------------------------------------------------------             ##
##                                                                            ##
##      I want to train a network (find a good minimum) and then              ##
##      add a perturbing force (e.g.\ linear coupling with the                ##
##      weights. Under this force, I minimize again, and then I               ##
##      can study the distribution of avalanches.                             ##
##                                                                            ##
##      Q: Are there wiser choices for the perturbation?                      ##
##                                                                            ##
##      Q: I don't expect the distribution to be power-law dist.,             ##
##         can I still say something about the landscape?                     ##
##                                                                            ##
################################################################################
##                                                                            ##
##      TODO:                                                                 ##
##                                                                            ##
##      * Write the code...?                                                  ##
##                                                                            ##
################################################################################


import os
import pickle
import numpy as np
import torch
from torch import Tensor, nn, optim, cuda
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# --  Models  ---------------------------------------------------------------- #


class SimpleNet(torch.nn.Module):
    """Simple convolutional networ: 2 conv layers followed by 2 fc layers.

      -- model = SimpleNet(# input channels, # num of output classes, image_size)
      -- model(data) performs the forward computation
    """

    def __init__(self, input_features, output_classes, image_size):
        super(SimpleNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(input_features, 10, kernel_size = 5, stride = 2)
        image_size = (image_size + 2*0 - 5)//2 + 1
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size = 5, stride = 2)
        image_size = (image_size + 2*0 - 5)//2 + 1
        self.fc1 = torch.nn.Linear(20*image_size**2, 50)
        self.fc2 = torch.nn.Linear(50, output_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim = 1)
        return x


# --  Datasets  -------------------------------------------------------------- #


# Fashion-MNIST dataset: 1 channel, 10 classes, 28x28 pixels
# Normalized as MNIST -- I should probably change it
trainset = list(datasets.FashionMNIST(
    '../data/',
    train = True,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
))

#testset = list(datasets.FashionMNIST('../data/', train = False, download = True, transform = transforms.Compose([ \
#    transforms.ToTensor(),
#    transforms.Normalize((0.1307,), (0.3081,))
#])))


# --  Other definitions  ----------------------------------------------------- #


class RandomSampler:
    """RandomSampler is a sampler for torch.utils.data.DataLoader.

     -- Each batch is independent (i.e. with repetition).
     -- __iter__() instead of returning a permutation of range(n), it gives n random numbers each in range(n).
    """

    def __init__(self, length):
        self.length = length

    def __iter__(self):
        return iter(np.random.choice(self.length, size = self.length))

    def __len__(self):
        return self.length

def load_batch(loader, cuda = False, only_one_epoch = False):
    """This function loads a single batch with torch.utils.data.DataLoader and RandomSampler.

     -- By default, there is no end to the number of batches (no concept of epoch).
        This is overridden with only_one_epoch = True.
    """

    while True:
        for data, target in iter(loader):
            if  cuda: data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            yield data, target

        if only_one_epoch: break  # exit the loop if only_one_epoch == True


# --  Training function  ----------------------------------------------------- #


def minimize_step(model, optimizer, data, target):
    output = model(data)
    loss = F.nll_loss(output, target, size_average = True)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def compute_batch_loss_grad(model, optimizer, data, target):
    output = model(data)
    loss = F.nll_loss(output, target, size_average = False)
    optimizer.zero_grad()
    loss.backward()
    grad = torch.cat([ p.grad.view(-1) for p in model.parameters() ])
    return loss, grad

def minimize(model, trainset, lr, bs, delta_tolerance, grad_tolerance, time_factor):
    """Train the model on the dataset until abs(Loss(t) - Loss(t-1)) < delta_tolerance.

     -- lr, bs are the learning rate and batch size.
     -- time_factor is used to build a logarithmic time scale; this is used to
        compute the TOTAL Loss at those times: this TOTAL Loss is used to
        check for convergence (Delta TOTAL Loss < delta_tolerance).
    """

    if cuda.is_available(): model.cuda()
    model.train()  # not necessary in this simple model, but I keep it for the sake of generality
    optimizer = optim.SGD(model.parameters(), lr = lr)    # learning rate

    trainloader = DataLoader(
        trainset,                                # dataset
        batch_size = bs,                        # batch size
        pin_memory = cuda.is_available(),        # speed-up for gpu's
        sampler = RandomSampler(len(trainset))    # no epochs
    )

    prev_loss = np.inf
    curr_loss = 0
    curr_grad_norm = 0

    next_t = 1.0
    batch = 0

#    with open(file_losses, 'wb') as losses_dump:
    for data, target in load_batch(trainloader, cuda = cuda.is_available()):
        batch += 1
        minimize_step(model, optimizer, data, target)

        if batch > next_t:
            next_t *= time_factor
            avg_loss, avg_grad = 0, 0
            total_trainloader = DataLoader(
                trainset,
                batch_size = 1024,  # I don't need small batches for this
                pin_memory = cuda.is_available(),
                sampler = RandomSampler(len(trainset))
            )

            for data, target in load_batch(total_trainloader, cuda = cuda.is_available(), only_one_epoch = True):
                d_avg_loss, d_avg_grad = compute_batch_loss_grad(model, optimizer, data, target)
                avg_loss += d_avg_loss.data[0]
                avg_grad += d_avg_grad

            curr_grad_norm = float(torch.norm(avg_grad/len(trainset)).data[0])
            curr_loss = avg_loss/len(trainset)
#            print("#", curr_loss, abs(curr_loss - prev_loss), curr_grad_norm)
            if abs(curr_loss - prev_loss) < delta_tolerance and curr_grad_norm < grad_tolerance: break
            prev_loss = curr_loss

    return curr_loss, curr_grad_norm


# ==  MAIN  ================================================================== #


# input_channels, output_classes, image_size (Fashion-MNIST = 28x28 -> size = 28)
network_parameters = (1, 10, 28)
model = SimpleNet(*network_parameters)

# number of perturbation steps
num_steps = 10  # JUST A TEST ##################################################

# minimizations stop when L(t) - L(t-1) < delta_tolerance and
# |grad| < grad_tolerance:
delta_tolerance = 1e-3  # JUST A TEST ##########################################
grad_tolerance = 0.3  # JUST A TEST ############################################
# time_factor is used to define logarithmic time intervals to compute TOTAL
# loss and gradient, to check for convergence
time_factor = 1.2

# temperatures, LR and BS
lr = 0.01
bs = 64
temp = lr/bs


# --  Prepare the system  ---------------------------------------------------- #


print("Training the system for the first time...")
init_loss, init_grad_norm = minimize(model, trainset, lr, bs,
    delta_tolerance = delta_tolerance,
    grad_tolerance = grad_tolerance,
    time_factor = time_factor
)
print(0, 0, init_loss, init_grad_norm, 0)
prev_loss = init_loss

print("Perturbation steps...")  # FOR THE MOMENT: NO PERTURBATION ##############
for t in range(num_steps):
    curr_loss, curr_grad_norm = minimize(model, trainset, lr, bs,
        delta_tolerance = delta_tolerance,
        grad_tolerance = grad_tolerance,
        time_factor = time_factor
    )
    print(t*lr, curr_loss, abs(curr_loss - prev_loss), curr_grad_norm)
    prev_loss = curr_loss
