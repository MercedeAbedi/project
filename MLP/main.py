import torch
import torch.nn as nn
from customDataset import bostonDataset
from customModel import MLP
from torch.utils.data import random_split, DataLoader
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F

#hyperparametrs
batch_size = 10
lr = 0.01
num_features = 11
num_epoch = 500

#Model
mymodel = MLP(num_features= num_features)


#load dataset
mydataset = bostonDataset('./dataset/Boston.csv')
num_train = (0.7 * len(mydataset)).__int__()
num_test = (0.2 * len(mydataset)).__int__()
num_valid = (0.1 * len(mydataset)).__int__()

train_data, valid_data, test_data = random_split(mydataset, [num_train, num_valid, num_test])

#scaler = StandardScaler()
#scaler.fit(train_data)
#x_train_std = scaler.transform(train_data)
#x_valid = scaler.transform(valid_data)
#x_test = scaler.transform(test_data)

dataloader = DataLoader(train_data, batch_size= batch_size, shuffle= True)

#loss and optimizer
Loss = nn.MSELoss()
optimizer = torch.optim.Adam(mymodel.parameters(), lr= lr)

#Train the Model
for epoch in range(num_epoch):
    #enumerate mini batches
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        out = mymodel(inputs)
        targets = targets.reshape((targets.shape[0], 1))
        loss_value = Loss(out, targets)
        loss_value.backward()
        optimizer.step()

        if epoch % 50 == 0:
            out = mymodel(mydataset.x[valid_data.indices, :num_features])
            _, predicted = torch.max(out.data, 1)
            label_v = mydataset.y[valid_data.indices]
            loss_v = Loss(out, label_v)
            print('Epoch [%d/%d] Train loss :%.4f' % (epoch + 1, num_epoch, loss_value.item()))
            print('Epoch [%d/%d] valid loss: %.4f' % (epoch + 1, num_epoch, loss_v.item()))
            acc = (100 * torch.sum(label_v == predicted) / num_valid)
            print('Accuracy of the network in validation %.4f %%' % acc)


