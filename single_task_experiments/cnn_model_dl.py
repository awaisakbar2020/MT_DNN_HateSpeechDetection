import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from cnn_preprocess_w import yield_data
from torch import optim
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import fasttext 
import fasttext.util
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer

mlb=MultiLabelBinarizer

ft = fasttext.load_model('fastText/wiki.en.bin')
ft.get_dimension()

if torch.cuda.is_available():
    device=torch.device("cuda:0")
else:
    device=torch.device("cpu")


#calling the method to get the splitted data
train, dev, test = yield_data(test_program=True)

def get_fastText_embeddings(sentence_list):
    embeddings=np.zeros((len(sentence_list),200,300),dtype='float32')
    for index,sentence in enumerate(sentence_list):
        if sentence != sentence:
            sentence = " "
        sentence=fasttext.tokenize(sentence)
        sentence_embeddings=np.zeros((200,300), dtype='float32')
        for token_no,word in enumerate(sentence[:200]):
            sentence_embeddings[token_no,:]=ft.get_word_vector(word)
        embeddings[index,:,:]=sentence_embeddings

    return embeddings;

#training data 
X_train = get_fastText_embeddings(train['sentence'].values)
y_train = train['class'].values
X_train=torch.from_numpy(X_train.astype(np.float32))
X_train=X_train.to(device)
y_train=torch.from_numpy(y_train.astype(np.float32))
train_data = torch.utils.data.TensorDataset(X_train,y_train) 
train_loader = torch.utils.data.DataLoader(train_data, batch_size=250)

#validation data
X_val = get_fastText_embeddings(dev['sentence'].values)
y_val = dev['class'].values
X_val=torch.from_numpy(X_val.astype(np.float32))
X_val=X_val.to(device)
y_val=torch.from_numpy(y_val.astype(np.float32))
validation_data = torch.utils.data.TensorDataset(X_val,y_val) 
validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=250)

#test data
X_test = get_fastText_embeddings(test['sentence'].values)
y_test = test['class'].values
X_test=torch.from_numpy(X_test.astype(np.float32))
X_test=X_test.to(device)
y_test=torch.from_numpy(y_test.astype(np.float32))
test_data = torch.utils.data.TensorDataset(X_test,y_test) 
test_loader = torch.utils.data.DataLoader(test_data, batch_size=250)

#----------------------------------------

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=300,out_channels=64,kernel_size=4,stride=4)
        self.conv2 = nn.Conv1d(in_channels=64,out_channels=64,kernel_size=3,stride=3)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 3)
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self,x):
        #first conv layer
        x=x.permute(0,2,1)
        x=self.conv1(x)
        x=F.relu(x)
        x=F.max_pool1d(x,2)
        
        #second conv layer
        x=self.conv2(x)
        x=F.relu(x)
        x=F.max_pool1d(x,2)
        
        #fully connected layer
        x = torch.flatten(x, 1)
        x = self.fc1(self.dropout(x))
        x = F.relu(x)
        x = self.fc2(self.dropout(x))
        #print("after fc2: ",x.shape)
        #x = torch.sigmoid(x)
        output = x
        return output

#---------------------------------------------

model = CNN()
model.to(device)

training_loss_list=[]
validation_loss_list=[]
testing_loss_list=[]
epoch_list=[]


# 2) Loss and optimizer
num_epochs = 10 # ashwin: reduce this for CNN
n_epochs_stop=5
epochs_no_improve=0
early_stop=False
learning_rate = 0.001 #ashwin: try with 0.001, which is the default one for Adam; the current one might be the probable reason for the model to overfit in the 1st epoch itself
criterion = nn.CrossEntropyLoss()
#criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 3) ---- Training loop ------
for epoch in range(num_epochs):
    model.train()
    train_loss =0
    for data, target in train_loader:
        # Forward pass and loss
        y_pred = model(data) 
        target=target.long()
        #target=target.unsqueeze(1)
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
            target=target.long()
            #target=target.unsqueeze(1)
            loss = criterion(y_pred_val, target) 
            # accumulate the valid_loss
            valid_loss += loss.item()
            
    train_loss /= len(X_train)
    valid_loss /= len(X_val)
    print(f'Epoch: {epoch+1}/{num_epochs}.. Training loss: {train_loss}.. Validation Loss: {valid_loss}')
    training_loss_list.append(train_loss)
    validation_loss_list.append(valid_loss)
    epoch_list.append(epoch+1)


    #------- early stopping & model saving ----------
    
    if(epoch==0):
        min_val_loss=valid_loss
        torch.save(model.state_dict(),'./cnn_model_inf.pt')
    else:
        if(valid_loss<min_val_loss):
            epochs_no_improve=0
            min_val_loss=valid_loss
            torch.save(model.state_dict(),'./cnn_model_inf.pt')
        else:
            epochs_no_improve+=1
            if(epochs_no_improve==n_epochs_stop):
                early_stop=True
                break
    if(early_stop):
        break
#new
# 5) ----------- testing ---------
model.load_state_dict(torch.load('./cnn_model_inf.pt'))
model.eval()
test_loss = 0
correct = 0
#y_pred_list= None
#target_list= None
pred_list=[]
targ_list=[]
acc_list=[]
# turn off gradients for testing
with torch.no_grad():
    for data, target in test_loader:
        y_predicted=model(data)   
        target=target.long()
        loss = criterion(y_predicted, target) 
        # accumulate the valid_loss
        test_loss += loss.item()
        pred_acc=torch.argmax(y_predicted,dim=1)
        pred_list.extend(pred_acc)
        targ_list.extend(target)

        #y_pred_list = torch.cat((y_pred_list,pred_acc)) if (y_pred_list not None) else pred_acc ## ashwin: modified line
        #target_list = torch.cat((target_list,target)) if (target_list not None) else target ## ashwin: modified line
        
        #acc=pred_acc.eq(target).sum()/float(target.shape[0])
        #acc_list.append(acc)
      
#y_pred_list=[a.squeeze().tolist() for a in y_pred_list]
#target_list=[b.squeeze().tolist() for b in target_list]

target_names=['Normal','Abusive', 'Hateful']
pred_tensor=torch.Tensor(pred_list)
targ_tensor=torch.Tensor(targ_list)
print(classification_report(targ_tensor,pred_tensor,target_names=target_names))
#print(classification_report(target_list.tolist(),y_pred_list.tolist(),target_names=target_names)) ## ashwin: modified line


accuracy=sum(acc_list)/len(acc_list)
print("accuracy is: ", accuracy.item())

test_loss /= len(X_test)
print(f'Test loss: {test_loss}')   

plt.figure(figsize=(10,6))

plt.style.use('fivethirtyeight')
#plt.set_size_inches(18.5, 10.5)

plt.plot(epoch_list, training_loss_list, label='Training Loss')
plt.plot(epoch_list, validation_loss_list, color='#444444', linestyle='--', label='Validation Loss')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('CNN-fastText: Training and Validation Loss')


plt.legend()

plt.tight_layout()

plt.savefig('cnn_fastText_w.png')

plt.show()
