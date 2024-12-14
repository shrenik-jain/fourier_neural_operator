# fourier_2d_rnn.py

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from utilities3 import *

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
import scipy.io


torch.manual_seed(0)
np.random.seed(0)


################################################################
# Model Definition
################################################################

class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()
        """
        2D Fourier layer

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            modes1 (int): Number of Fourier modes to keep along the first dimension.
            modes2 (int): Number of Fourier modes to keep along the second dimension.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to retain along the 1st dimension
        self.modes2 = modes2 # Number of Fourier modes to retain along the 2nd dimension

        self.scale = (1 / (in_channels * out_channels))
        # weight matrix for the Fourier coefficients 
        self.weights1 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input, weights):
        """
        # Performs complex multiplication of the input and weights
        # input: (batch, in_channel, x, y), complex tensor
        # weights: (in_channel, out_channel, x, y), complex tensor
        # output: (batch, out_channel, x, y), complex tensor
        """
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft2(x, norm='ortho')

        # Multiply relevant Fourier modes with weight matrices
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)

        # First modes
        # This operation selectively applies weights to the lower-frequency Fourier modes
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )

        # Negative modes
        # This operation selectively applies weights to the higher-frequency Fourier modes (negative frequencies)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm='ortho')
        return x


class SimpleBlock2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(SimpleBlock2d, self).__init__()

        # mode1, mode2 -> how many Fourier modes to retain along each dimension
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        # 12 accounts for -> 10 timesteps + 2 positional encodings
        self.fc0 = nn.Linear(12, self.width)

        # Fourier layer with (modes1, modes2) modes
        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        # 1D convolution with kernel size 1
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        # Batch normalization
        self.bn0 = torch.nn.BatchNorm2d(self.width)
        self.bn1 = torch.nn.BatchNorm2d(self.width)
        self.bn2 = torch.nn.BatchNorm2d(self.width)
        self.bn3 = torch.nn.BatchNorm2d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        # 1 output channel -> vorticity (x, y, t)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)  # (batch, channels, x, y)

        ## 1st layer
        # fourier layer (upper branch)
        x1 = self.conv0(x)
        # linear layer which operate pointwise (in channels) (Conv1x1) (lower branch)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        # batch normalization
        x = self.bn0(x1 + x2)
        # nonlinearity
        x = F.relu(x)

        ## 2nd layer
        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = self.bn1(x1 + x2)
        x = F.relu(x)

        ## 3rd layer
        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = self.bn2(x1 + x2)
        x = F.relu(x)

        ## 4th layer
        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = self.bn3(x1 + x2)

        # reshape the data
        x = x.permute(0, 2, 3, 1)  # (batch, x, y, channels)
        # fully connected layer1
        x = self.fc1(x)
        # nonlinearity
        x = F.relu(x)
        # fully connected layer2
        x = self.fc2(x)
        return x


class Net2d(nn.Module):
    def __init__(self, modes, width):
        super(Net2d, self).__init__()

        self.conv1 = SimpleBlock2d(modes, modes, width)

    def forward(self, x):
        x = self.conv1(x)
        return x

    def count_params(self):
        c = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return c


################################################################
# Parameters and Settings
################################################################   

TRAIN_PATH = 'ns_data_R256_S20.mat'
TEST_PATH = 'ns_data_R256_S20.mat'

ntrain = 16
ntest = 4

modes = 12 # Number of Fourier modes to retain
width = 20

batch_size = 4

epochs = 500
learning_rate = 0.0025
scheduler_step = 100 # number of epochs after which the learning rate is reduced
scheduler_gamma = 0.5 # factor by which the learning rate is reduced

print(f"Hyperparameters For The Model \nEpochs: {epochs}\n Learning Rate: {learning_rate}\n Scheduler Step{scheduler_step}\n Scheduler Gamma{scheduler_gamma}")

path = 'ns_fourier_2d_rnn_V10000_T20_N'+str(ntrain)+'_ep' + str(epochs) + '_m' + str(modes) + '_w' + str(width)
path_model = 'model/'+path
path_train_err = 'results/'+path+'train.txt'
path_test_err = 'results/'+path+'test.txt'
path_image = 'image/'+path

runtime = np.zeros(2, )
t1 = default_timer()

sub = 4 
S = 64 # spatial resolution
T_in = 10 # input time steps
T = 10 # output time steps
step = 1 # step size between time steps


################################################################
# Load data
################################################################

reader = MatReader(TRAIN_PATH)
train_a = reader.read_field('u')[:ntrain, ::sub, ::sub, :T_in]
train_u = reader.read_field('u')[:ntrain, ::sub, ::sub, T_in:T+T_in]

reader = MatReader(TEST_PATH)
test_a = reader.read_field('u')[-ntest:, ::sub, ::sub, :T_in]
test_u = reader.read_field('u')[-ntest:, ::sub, ::sub, T_in:T+T_in]

print("Train U Shape:", train_u.shape) # 16, 64, 64, 10
print("Test U Shape:", test_u.shape) # 4, 64, 64, 10
assert (S == train_u.shape[-2])
assert (T == train_u.shape[-1])

train_a = train_a.reshape(ntrain, S, S, T_in) # 16, 64, 64, 10
test_a = test_a.reshape(ntest, S, S, T_in)  # 4, 64, 64, 10


# These tensors (grid_x, grid_y) are used to pad the input data with the corresponding 
# x and y coordinates, which can be useful for positional information. (Basically linear positional encodings)
gridx = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
gridx = gridx.reshape(1, S, 1, 1).repeat([1, 1, S, 1])
gridy = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
gridy = gridy.reshape(1, 1, S, 1).repeat([1, S, 1, 1])

# Add the positional encodings to the input data
train_a = torch.cat((gridx.repeat([ntrain, 1, 1, 1]), gridy.repeat([ntrain, 1, 1, 1]), train_a), dim=-1)
test_a = torch.cat((gridx.repeat([ntest, 1, 1, 1]), gridy.repeat([ntest, 1, 1, 1]), test_a), dim=-1)

# Convert to dataloader object
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False
)

t2 = default_timer()

print('preprocessing finished, time used:', t2 - t1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


################################################################
# Training and evaluation
################################################################

model = Net2d(modes, width).to(device)
# model = torch.load('model/ns_fourier_V100_N1000_ep100_m8_w20')

print("Number of model parameters:", model.count_params()) 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

myloss = LpLoss(size_average=False)
gridx = gridx.to(device)
gridy = gridy.to(device)

# Initialize lists to store losses
train_losses = []
test_losses = []

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2_step = 0 # this will store the loss for each time step
    train_l2_full = 0 # this will store the loss for the entire time sequence
    for xx, yy in train_loader:
        loss = 0 # this will store the loss for each batch
        xx = xx.to(device)
        yy = yy.to(device)
        current_batch_size = xx.shape[0]

        for t in range(0, T, step):
            y = yy[..., t:t + step]
            im = model(xx) 
            loss += myloss(im.reshape(current_batch_size, -1), y.reshape(current_batch_size, -1))

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)

            xx = torch.cat(
                (
                    xx[..., step:-2],
                    im,
                    gridx.repeat([current_batch_size, 1, 1, 1]),
                    gridy.repeat([current_batch_size, 1, 1, 1])
                ),
                dim=-1
            )

        train_l2_step += loss.item()
        l2_full = myloss(pred.reshape(current_batch_size, -1), yy.reshape(current_batch_size, -1))
        train_l2_full += l2_full.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Record the average training loss for this epoch
    avg_train_loss = train_l2_full / ntrain
    train_losses.append(avg_train_loss)

    test_l2_step = 0 # this will store the loss for each time step
    test_l2_full = 0 # this will store the loss for the entire time sequence
    with torch.no_grad():
        model.eval()
        for xx, yy in test_loader:
            loss = 0 # this will store the loss for each batch
            xx = xx.to(device)
            yy = yy.to(device)
            current_batch_size = xx.shape[0]

            for t in range(0, T, step):
                y = yy[..., t:t + step]
                im = model(xx)
                loss += myloss(im.reshape(current_batch_size, -1), y.reshape(current_batch_size, -1))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)

                xx = torch.cat(
                    (
                        xx[..., step:-2],
                        im,
                        gridx.repeat([current_batch_size, 1, 1, 1]),
                        gridy.repeat([current_batch_size, 1, 1, 1])
                    ),
                    dim=-1
                )

            test_l2_step += loss.item()
            test_l2_full += myloss(pred.reshape(current_batch_size, -1), yy.reshape(current_batch_size, -1)).item()

    # Record the average testing loss for this epoch
    avg_test_loss = test_l2_full / ntest
    test_losses.append(avg_test_loss)

    t2 = default_timer()
    scheduler.step()
    print(
        f"Epoch {ep}, Time: {t2 - t1:.2f}s, "
        f"Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}"
    )


# Saving the trained model
torch.save(model.state_dict(), path_model + '.pth')


# ################################################################
# # Plotting Losses
# ################################################################

plt.figure(figsize=(10, 5))
plt.plot(range(epochs), train_losses, label='FNO-2D', color='orange')
# plt.plot(range(epochs), test_losses, label='Testing Loss')
plt.xlabel('Epochs')
plt.ylabel('Relative error')

# Set y-axis to logarithmic scale
plt.yscale('log')

# Optional: Set custom y-axis ticks (optional)
plt.yticks([10**i for i in range(-2, 1)])

# plt.title('Error vs. Epochs')
plt.legend()
# plt.grid(True)
plt.show()

# Save the plot
plt.savefig('fno_2d_rnn.png')


# ################################################################
# # Generating and Saving Predictions
# ################################################################

predictions = []
test_l2 = 0.0
with torch.no_grad():
    model.eval()
    for xx, yy in test_loader:
        xx = xx.to(device)
        yy = yy.to(device)

        print(f"Input shape: {xx.shape}")
        print(f"Target shape: {yy.shape}")
        
        pred = model(xx)
        print(f"Raw prediction shape: {pred.shape}")
        
        # Repeat prediction along last dimension to match target
        pred = pred.expand(-1, -1, -1, yy.shape[-1])
        print(f"Expanded prediction shape: {pred.shape}")

        predictions.append(pred.cpu())
        test_l2 += myloss(pred.reshape(pred.size(0), -1), yy.to(device).reshape(pred.size(0), -1)).item()

# Concatenate all predictions
predictions = torch.cat(predictions, dim=0)  # Shape: (ntest, S, S, 1)

# Save predictions as a PyTorch tensor
torch.save({'predictions': predictions}, 'predictions.pt')

# Optionally, save predictions in .mat format if needed
scipy.io.savemat('pred/' + path + '.mat', {'pred': predictions.numpy()})


# ################################################################
# # Visualizing Predictions vs Ground Truth
# ################################################################

# Load the saved predictions
predictions_data = torch.load('predictions.pt')
predictions = predictions_data['predictions']

# Load the ground truth data
reader = MatReader(TEST_PATH)
ground_truth = reader.read_field('u')[-ntest:, ::sub, ::sub, T_in:T+T_in]

def visualize_predictions(predictions, ground_truth, num_samples=2, timesteps=[0, 5, 9]):
    """
    Visualize predictions vs ground truth at specific timesteps.
    
    :param predictions: model predictions [batch, height, width, time]
    :param ground_truth: ground truth data [batch, height, width, time]
    :param num_samples: number of samples to visualize
    :param timesteps: list of timesteps to visualize
    """
    num_timesteps = len(timesteps)
    fig, axes = plt.subplots(2 * num_samples, num_timesteps, figsize=(4*num_timesteps, 4*num_samples))
    
    for i in range(num_samples):
        for j, t in enumerate(timesteps):
            # Ground truth
            axes[2*i, j].imshow(ground_truth[i, :, :, t].cpu().numpy(), cmap='viridis')
            axes[2*i, j].set_title(f'Ground Truth t={t}')
            axes[2*i, j].axis('off')
            
            # Prediction
            axes[2*i+1, j].imshow(predictions[i, :, :, t].cpu().numpy(), cmap='viridis')
            axes[2*i+1, j].set_title(f'Prediction t={t}')
            axes[2*i+1, j].axis('off')
    
    plt.tight_layout()
    plt.show()


visualize_predictions(predictions, ground_truth)