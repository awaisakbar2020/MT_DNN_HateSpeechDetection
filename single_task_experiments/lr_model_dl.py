import torch
import torch.nn as nn
import numpy as np
from get_embeddings_awais import obtain_embeddings
from preprocess_awais import yield_data
from torch import optim
from matplotlib import pyplot as plt


#calling the method to get the splitted data
train, dev, test = yield_data(test_program=False)

#training data 
X_train = obtain_embeddings(train['sentence'].values)
y_train = train['class'].values
X_train=torch.from_numpy(X_train.astype(np.float32))
y_train=torch.from_numpy(y_train.astype(np.float32))
train_data = torch.utils.data.TensorDataset(X_train,y_train) 
train_loader = torch.utils.data.DataLoader(train_data, batch_size=1)

#validation data
X_val = obtain_embeddings(dev['sentence'].values)
y_val = dev['class'].values
X_val=torch.from_numpy(X_val.astype(np.float32))
y_val=torch.from_numpy(y_val.astype(np.float32))
validation_data = torch.utils.data.TensorDataset(X_val,y_val) 
validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=1)

#test data
X_test = obtain_embeddings(test['sentence'].values)
y_test = test['class'].values
X_test=torch.from_numpy(X_test.astype(np.float32))
y_test=torch.from_numpy(y_test.astype(np.float32))
test_data = torch.utils.data.TensorDataset(X_test,y_test) 
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1)

# 1) Model
# Linear model z = wx + b , sigmoid at the end
class Model(nn.Module):
  # def __init__(self, n_input_features):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(512, 1) 

    def forward(self, x):
        #should this be torch.nn.sigmoid?
        y_pred = torch.sigmoid(self.linear(x)) 
        return y_pred

model = Model()

training_loss_list=[]
validation_loss_list=[]
testing_loss_list=[]
epoch_list=[]


# 2) Loss and optimizer
num_epochs = 100
learning_rate = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) ---- Training loop ------
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for data, target in train_loader:
        # Forward pass and loss
        print("shape",data.shape,target.shape)
        y_pred = model(data) 
        #y_pred = model(X_train)
        target = target.unsqueeze(1)
        loss = criterion(y_pred, target)

        #Backward pass and update
        loss.backward()
        optimizer.step()

        # zero grad before new step
        optimizer.zero_grad()
    
        train_loss += loss.item()
        if (epoch+1) % 10 == 0:
            print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')
    

# 4) ---- Validation loop ------
    # set the model to eval mode
    model.eval()
    valid_loss = 0
    # turn off gradients for validation
    with torch.no_grad():
        for data, target in validation_loader:
            # forward pass
            y_pred_val = model(data)
            # validation batch loss
            target = target.unsqueeze(1)
            loss = criterion(y_pred_val, target) 
            # accumulate the valid_loss
            valid_loss += loss.item()
            
    train_loss /= len(X_train)
    valid_loss /= len(X_val)
    print(f'Epoch: {epoch+1}/{num_epochs}.. Training loss: {train_loss}.. Validation Loss: {valid_loss}')
    training_loss_list.append(train_loss)
    validation_loss_list.append(valid_loss)
    epoch_list.append(epoch+1)
    #----early stopping---
    if (validation_loss_list[epoch]>validation_loss_list[epoch-1]):
        break


# 5) ----------- testing ---------
model.eval()
test_loss = 0
correct = 0
# turn off gradients for testing
with torch.no_grad():
    for data, target in test_loader:
        y_predicted = model(data)
        target = target.unsqueeze(1)
        loss = criterion(y_predicted, target) 
        # accumulate the valid_loss
        test_loss += loss.item()
        
y_predicted_cls = y_predicted.round()
acc = y_predicted_cls.eq(target).sum() / float(target.shape[0])
print(f'accuracy: {acc.item():.4f}')  
test_loss /= len(X_test)
print(f'Test loss: {test_loss}')   

plt.figure(figsize=(10,6))

plt.style.use('fivethirtyeight')

plt.plot(epoch_list, training_loss_list, label='Training Loss')
plt.plot(epoch_list, validation_loss_list, color='#444444', linestyle='--', label='Validation Loss')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Logistic Regression: Training and Validation Loss')


plt.legend()

plt.tight_layout()

plt.savefig('logistic_regression.png')

plt.show()
