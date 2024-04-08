#%% Imports
# Import general modules and packages needed for the script to run
import os.path  
import h5py 
import time
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from tqdm import tqdm 

import torch 
from torch import nn 
import torch.nn as nn 
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torchsummary import summary 
from torch.nn.utils import prune 

import brevitas.nn as qnn 
from brevitas.quant import IntBias
from brevitas.inject.enum import ScalingImplType
from brevitas.inject.defaults import Int8ActPerTensorFloatMinMaxInit

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

if torch.cuda.is_available():
    print("CUDA is available!")
else:
    print("CUDA is not available.")

#%% Data loading 
# Data loading 
#Check if the dataset is available locally 
dataset_path = "/home/student/Downloads/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5"
os.path.isfile(dataset_path) 

#Load data from HDF5 file into a PyTorch tensor 
class Radioml_18(Dataset):
    def __init__(self, dataset_path): 
        super(Radioml_18, self).__init__()
        h5py_file = h5py.File(dataset_path, 'r')
        self.data = h5py_file['X']
        self.modulations  = np.argmax(h5py_file['Y'], axis=1) 
        self.snr = h5py_file['Z'][:, 0]
        self.len = self.data.shape[0] 

        self.mod_classes = ['OOK','4ASK','8ASK','BPSK','QPSK','8PSK','16PSK','32PSK',
        '16APSK','32APSK','64APSK','128APSK','16QAM','32QAM','64QAM','128QAM','256QAM',
        'AM-SSB-WC','AM-SSB-SC','AM-DSB-WC','AM-DSB-SC','FM','GMSK','OQPSK']

        self.snr_classes = np.arange(-20.,31.,2) 

        np.random.seed(0) #For reproducibility 
        train_indices = []
        validation_indices = []
        test_indices = []
        for mod in range(0, 24): # All  24 modulationa
            for snr_idx in range(0, 26): # All signal to noise ratios from (-20, 30) Db
                start_index = 26*4096*mod + 4069*snr_idx 
                # Because X holds frames srticktly ordered by modulation and snr  
                indices_subclass = list(range(start_index, start_index+4096))

                # Splitting the data into 80% training and 20% testing 
                split = int(np.ceil(0.1*4096))
                np.random.shuffle(indices_subclass) 
                train_indicies_sublcass = indices_subclass[:int(0.7*len(indices_subclass))]
                validation_indices_subclass = indices_subclass[int(0.7*len(indices_subclass)):int(0.8*len(indices_subclass))]
                test_indicies_subclass = indices_subclass[int(0.8*len(indices_subclass)):] 
                
                # to choose a specific SNR valaue or range is here 
                if snr_idx >= 0: 
                    train_indices.extend(train_indicies_sublcass)
                    validation_indices.extend(validation_indices_subclass)
                    test_indices.extend(test_indicies_subclass)

        self.train_sampler = SubsetRandomSampler(train_indices)
        self.validation_sampler = SubsetRandomSampler(validation_indices)
        self.test_sampler = SubsetRandomSampler(test_indices)

        print('Dataset shape:', self.data.shape)
        print("Train indices:", len(train_indices))
        print("Validation indices:", len(validation_indices))
        print("Test indices:", len(test_indices)) 

        # Print input length
        input_length = self.data.shape[1]
        print("Input length:", input_length)

    def __getitem__(self, index):
        # Transform frame into pytorch channels-first format 
        return self.data[index].transpose(), self.modulations[index], self.snr[index]
    
    def __len__(self): 
        return self.len 
    


dataset = Radioml_18(dataset_path)

batchsize = 1024 

train_loader = DataLoader(dataset, batch_size= batchsize, sampler= dataset.train_sampler) 
validation_loader = DataLoader(dataset, batch_size= batchsize, sampler= dataset.validation_sampler)
test_loader = DataLoader(dataset, batch_size= batchsize, sampler= dataset.test_sampler)


# %% The CNN quantizied model 
# Defining the quantization parameters 
in_bits = 8 
activation_bits = 8 
w_bits = 8 
filter_conv = 64 
filter_dense = 128

torch.manual_seed(0)
np.random.seed(0) 

class InputQuantizer(Int8ActPerTensorFloatMinMaxInit): 
    bit_width = in_bits
    min_val = -2.0 
    max_val = 2.0 
    scaling_impl_type = ScalingImplType.CONST


model = nn.Sequential( 
    qnn.QuantHardTanh(act_quant= InputQuantizer), 

    qnn.QuantConv1d(in_channels=2, out_channels=64, kernel_size=3, padding=1, weight_bit_width=w_bits, bias=False), 
    nn.BatchNorm1d(filter_conv), 
    qnn.QuantReLU(bit_width = activation_bits), 
    nn.MaxPool1d(2), 

    qnn.QuantConv1d(filter_conv, filter_conv, 3, padding=1, weight_bit_width=w_bits, bias=False), 
    nn.BatchNorm1d(filter_conv), 
    qnn.QuantReLU(bit_width = activation_bits), 
    nn.MaxPool1d(2), 

    qnn.QuantConv1d(filter_conv, filter_conv, 3, padding=1, weight_bit_width=w_bits, bias=False), 
    nn.BatchNorm1d(filter_conv), 
    qnn.QuantReLU(bit_width = activation_bits), 
    nn.MaxPool1d(2), 

    qnn.QuantConv1d(filter_conv, filter_conv, 3, padding=1, weight_bit_width=w_bits, bias=False), 
    nn.BatchNorm1d(filter_conv), 
    qnn.QuantReLU(bit_width = activation_bits), 
    nn.MaxPool1d(2), 

    qnn.QuantConv1d(filter_conv, filter_conv, 3, padding=1, weight_bit_width=w_bits, bias=False), 
    nn.BatchNorm1d(filter_conv), 
    qnn.QuantReLU(bit_width = activation_bits), 
    nn.MaxPool1d(2), 

    qnn.QuantConv1d(filter_conv, filter_conv, 3, padding=1, weight_bit_width=w_bits, bias=False), 
    nn.BatchNorm1d(filter_conv), 
    qnn.QuantReLU(bit_width = activation_bits), 
    nn.MaxPool1d(2), 

    qnn.QuantConv1d(filter_conv, filter_conv, 3, padding=1, weight_bit_width=w_bits, bias=False), 
    nn.BatchNorm1d(filter_conv), 
    qnn.QuantReLU(bit_width = activation_bits), 
    nn.MaxPool1d(2), 

    nn.Flatten(), 

    qnn.QuantLinear(filter_conv*8, filter_dense, weight_bit_width= w_bits, bias=False), 
    nn.BatchNorm1d(filter_dense), 
    qnn.QuantReLU(bit_width = activation_bits),

    qnn.QuantLinear(filter_dense, filter_dense, weight_bit_width= w_bits, bias=False), 
    nn.BatchNorm1d(filter_dense), 
    qnn.QuantReLU(bit_width = activation_bits, return_quant_tensor=True), 

    qnn.QuantLinear(filter_dense, 24, weight_bit_width = w_bits, bias=True, bias_quant=IntBias)
)

#device = torch.device('cuda')
#model.to(device)

# Upload a saved model and then using the pruning function below to prune the model 
state_dict = torch.load('/home/student/Desktop/saved_quantized_model_all_snrs_50_epochs.pth')
model.load_state_dict(state_dict)

# %% Loops for training and testing
# Training and testing loops  

def train_loop(model, train_loader, optimizer, criterion):
    losses = []
    model.train()

    for (inputs, labels, snr) in tqdm(train_loader):
        #inputs, labels = inputs.cuda(), labels.cuda()  # Move data to GPU
        # Forward pass
        output = model(inputs)
        loss = criterion(output, labels)

        # Backward pass and optimizer
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        losses.append(loss.detach().numpy())  # Use .item() to get the scalar value of the loss

    return losses
    
def test_loop(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for (inputs, labels, snr) in tqdm(test_loader):
            #inputs, labels = inputs.cuda(), labels.cuda()  # Move data to GPU
            outputs = model(inputs)
            pred = outputs.argmax(dim=1, keepdim=True)
            
            y_true.append(labels.numpy())  # Move labels back to CPU for concatenation
            y_pred.append(pred.reshape(-1).numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    print(y_true)
    print(y_pred)
    # Calculate and print F1 score
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f'F1 Score: {f1}')

    return accuracy_score(y_true, y_pred) 

def display_loss(losses, title = 'Training loss', xlabel= 'Iterations', ylabel= 'Loss'):
    x_axis = [i for i in range(len(losses))] 
    plt.plot(x_axis, losses)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


# %% Training the model 

num_epochs = 20
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1)  # Example scheduler

train_data = train_loader
validation_data = validation_loader

running_loss = []
accuracy = []

for epoch in range(num_epochs):

    loss_epoch = train_loop(model, train_data, optimizer, criterion)
    test_acc = test_loop(model, validation_data)

    print("Epoch %d: Training loss = %f, validation accuracy = %f" % (epoch, np.mean(loss_epoch), test_acc))
    running_loss.append(loss_epoch)
    accuracy.append(test_acc)
    
    # Step the scheduler
    scheduler.step()


# plot the running loss and accuracy
display_loss(running_loss)

 # %% Plotting the loss and the accuracy of the model 
    
# Plot training loss over epochs
loss_per_epoch = [np.mean(loss_per_epoch) for loss_per_epoch in running_loss]
display_loss(loss_per_epoch)

# Plot validation accuracy over epochs
acc_per_epoch = [np.mean(acc_per_epoch) for acc_per_epoch in accuracy]
display_loss(acc_per_epoch, title="Validation accuracy", ylabel="Accuracy [%]")

# %% Evaluating the accuracy of the model 
#Evaluating the accuracy 
test_data = test_loader
y_exp = np.empty((0))
y_snr = np.empty((0))
y_pred = np.empty((0, len(dataset.mod_classes)))

model.eval()
total_time = 0
num_samples = 0 
with torch.no_grad():
    for data in tqdm(test_data, desc="Batches"):
        inputs, target, snr = data
        #inputs, target = inputs.cuda(), target.cuda()
        start_time = time.time()
        output = model(inputs)
        end_time = time.time() 
        total_time += end_time - start_time 

        num_samples += inputs.size(0) 

        y_pred = np.concatenate((y_pred, output.numpy()))
        y_exp = np.concatenate((y_exp, target.numpy()))  
        y_snr = np.concatenate((y_snr, snr))
    
    avg_time = total_time / num_samples
    print("Average time per output: ", avg_time)



#%% The confusion matrix 
#Print the condustion matrix 
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

conf = np.zeros([len(dataset.mod_classes),len(dataset.mod_classes)])
confnorm = np.zeros([len(dataset.mod_classes),len(dataset.mod_classes)])
for i in range(len(y_exp)):
    j = int(y_exp[i])
    k = int(np.argmax(y_pred[i,:]))
    conf[j,k] = conf[j,k] + 1
for i in range(0,len(dataset.mod_classes)):
    confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])

plt.figure(figsize=(12,8))
plot_confusion_matrix(confnorm, labels=dataset.mod_classes)

cor = np.sum(np.diag(conf))
ncor = np.sum(conf) - cor
print("Overall Accuracy - all SNRs: %f"%(cor / (cor+ncor)))


# %% Saving the model parameters 

torch.save(model.state_dict(), '/home/student/Desktop/saved_quantized_model_10%pruned_20epochs.pth') 

#%% Confustion matricies for multiple SNRs

snr_to_plot = [-20, 0, +20, +30]
acc = []

plt.figure(figsize=(16,10))

for snr in dataset.snr_classes: 
    indices_snr = (y_snr == snr).nonzero() 
    y_exp_i = y_exp[indices_snr] 
    y_pred_i = y_pred[indices_snr] 

    conf = np.zeros([len(dataset.mod_classes), len(dataset.mod_classes)])
    confnorm = np.zeros([len(dataset.mod_classes), len(dataset.mod_classes)])

    for i in range(len(y_exp_i)):
        j = int(y_exp_i[i])
        k = int(np.argmax(y_pred_i[i, :]))
        conf[j, k] = conf[j,k] + 1

    for i in range(0, len(dataset.mod_classes)): 
        confnorm[i, :] = conf[i, :] / np.sum(conf[i, :]) 

    if snr in snr_to_plot:
        plot, = np.where(snr_to_plot ==snr) [0] 
        plt.subplot(221+plot) 
        plot_confusion_matrix(confnorm, labels = dataset.mod_classes, title = 'Confusion MAtrix @ %d dB'%(snr))

    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) -cor 
    
    acc.append(cor / (cor+ncor))

# %% Accuracy plots
# plotting the accuracy over SNR 

plt.figure(figsize=(10,6)) 
plt.plot(dataset.snr_classes, acc, marker='o')
plt.xlabel=(r'SNR $dB$')
plt.ylabel('Classification Accuracy')
plt.title('Classification Accuracy vs Signal-to-Noise Ratio')
plt.xlim([-20, 30]) 
plt.yticks(np.arange(0 , 1.1 , 0.1))
plt.grid(True) 

print('Accuracy at highest SNR(30dB): %f' %(acc[-1]))
print('Accuracy Overall: %f' %(np.mean(acc))) 


