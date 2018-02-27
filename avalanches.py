########################################################################
##                                                                    ##
##      AVALANCHES                                                    ##
##     ----------------------------------------------------------     ##
##                                                                    ##
##      I want to train a network (find a good minimum) and then      ##
##      add a perturbing force (e.g.\ linear coupling with the        ##
##      weights. Under this force, I minimize again, and then I       ##
##      can study the distribution of avalanches.                     ##
##                                                                    ##
##      Q: Are there wiser choices for the perturbation?              ##
##                                                                    ##
##      Q: I don't expect the distribution to be power-law dist.,     ##
##         can I still say something about the landscape?             ##
##                                                                    ##
########################################################################
##                                                                    ##
##      TODO:                                                         ##
##                                                                    ##
##      * Write the code...?                                          ##
##                                                                    ##
########################################################################


import os
import pickle
import numpy as np
import torch
from torch import Tensor, nn, optim, cuda
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# --  Models  -------------------------------------------------------- #


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


# --  Datasets  ------------------------------------------------------ #


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


# --  Other definitions  --------------------------------------------- #


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


# --  Training function  --------------------------------------------- #


def train_and_save(model, trainset, lr, bs, minimization_time, file_state, file_losses, time_factor = None):
    """
    """

    if cuda.is_available(): model.cuda()
    model.train()  # not necessary in this simple model, but I keep it for the sake of generality
    optimizer = optim.SGD(model.parameters(), lr = lr)	# learning rate

    trainloader = DataLoader(
        trainset,								# dataset
        batch_size = bs,						# batch size
        pin_memory = cuda.is_available(),		# speed-up for gpu's
        sampler = RandomSampler(len(trainset))	# no epochs
    )

    if time_factor == None: time_factor = minimization_time**(1.0/200)
    next_t = 1.0
    batch = 0

    with open(file_losses, 'wb') as losses_dump:
        for data, target in load_batch(trainloader, cuda = cuda.is_available()):
            batch += 1
            if batch > minimization_time:
                break

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target, size_average = True)
            loss.backward()
            optimizer.step()

            if batch > next_t:
                # I want to save the average loss on the total training set
                avg_loss = 0
                total_trainloader = DataLoader(
                    trainset,
                    batch_size = 1024,  # I don't need small batches for this
                    pin_memory = cuda.is_available(),
                    sampler = RandomSampler(len(trainset))
                )

                for data, target in load_batch(total_trainloader, cuda = cuda.is_available(), only_one_epoch = True):
                    output = model(data)
                    avg_loss += F.nll_loss(output, target, size_average = False).data[0]

                pickle.dump(( batch, avg_loss/len(trainset) ), losses_dump)
                next_t *= time_factor

    state_dict = model.state_dict()  # == losses[-1]['state_dict']
    torch.save(state_dict, file_state)

    return state_dict


# ==  MAIN  ========================================================== #


# input_channels, output_classes, image_size (Fashion-MNIST = 28x28 -> size = 28)
network_parameters = (1, 10, 28)
model = SimpleNet(*network_parameters)

# number of perturbation steps
num_steps = 100  # JUST A TEST ########################################

# minimizations stop when L(t) - L(t-1) < delta_tolerance:
delta_tolerance = 1e-4  # JUST A TEST ##################################

# temperatures, LR and BS
lr = 0.01
bs = 64
temp = lr/bs


# --  Prepare the system  -------------------------------------------- #


init_loss = minimize(model, lr, bs, delta_tolerance = delta_tolerance)
print(0, 0, init_loss)
prev_loss = init_loss

for t in range(num_steps):
	curr_loss = minimize(model, lr, bs, delta_tolerance = delta_tolerance)
	print(t*lr, curr_loss, curr_loss - prev_loss)  # JUST A TEST #######
	prev_loss = curr_loss
