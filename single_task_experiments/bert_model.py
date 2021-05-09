import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from bert_preprocess import yield_data
from torch import optim
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import fasttext 
import fasttext.util
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer

#calling the method to get the splitted data
X_train, X_val, X_test, y_train, y_val, y_test = yield_data()

if torch.cuda.is_available():    
    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
    
    
from transformers import BertTokenizer

print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

attention_masks = []

from keras.preprocessing.sequence import pad_sequences

def tokenize_encode_mask(sentences):
   
    input_ids=[]
    attention_masks=[]
    MAX_LEN=84
    for sent in sentences:
        encoded_sent = tokenizer.encode_plus(sent, return_tensors='pt', padding="max_length", truncation=True,max_length=MAX_LEN, add_special_tokens = True)
        input_i = encoded_sent['input_ids']
        attention_m = encoded_sent['attention_mask']
        input_ids.append(input_i)
        attention_masks.append(attention_m)
    
    return torch.cat(input_ids,dim=0), torch.cat(attention_masks,dim=0)

    
#training data 
X_train, train_masks = tokenize_encode_mask(X_train)
train_masks=train_masks.long()
X_train=X_train.long()
X_train=X_train.to(device)
y_train=pd.DataFrame(y_train).to_numpy()
y_train=torch.from_numpy(y_train.astype(np.float32))
y_train=y_train.to(device).long()


#validation data
X_val, val_masks = tokenize_encode_mask(X_val)
val_masks=val_masks.long()
X_val=X_val.long()
X_val=X_val.to(device)
y_val=pd.DataFrame(y_val).to_numpy()
y_val=torch.from_numpy(y_val.astype(np.float32))
y_val=y_val.to(device).long()

#test data
X_test, test_masks = tokenize_encode_mask(X_test)
test_masks=test_masks.long()
X_test=X_test.long()
X_test=X_test.to(device)
y_test=pd.DataFrame(y_test).to_numpy()
y_test=torch.from_numpy(y_test.astype(np.float32))
y_test=y_test.to(device).long()

batch_size = 32

train_data = torch.utils.data.TensorDataset(X_train,train_masks, y_train) 
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)

validation_data = torch.utils.data.TensorDataset(X_val,val_masks, y_val) 
validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size)

test_data = torch.utils.data.TensorDataset(X_test,test_masks, y_test) 
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)    
    

from transformers import BertForSequenceClassification, AdamW, BertConfig

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',num_labels = 3, output_attentions = False, output_hidden_states = False, 
    ) 

model.to(device)

optimizer = AdamW(model.parameters(),
                  lr = 2e-5, 
                  eps = 1e-8 
                )
               
from transformers import get_linear_schedule_with_warmup

num_epochs = 3

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_loader) * num_epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)



early_stop=False
training_loss_list=[]
validation_loss_list=[]
epoch_list=[]

# 3) ---- Training loop ------
for epoch in range(num_epochs):
    model.train()
    train_loss =0
    for data, mask, target in train_loader:
        #print("debug: ", data, mask, target)
        outputs = model(input_ids=data,attention_mask=mask,labels=target)
        
        #print("debug: outputs: ", outputs)

        
        loss=outputs[0]

        #print("debug: loss ", loss)
        logits=outputs[1] 
        #loss=outputs.loss
        #logits=outputs.logits

        #Backward pass and update
        loss.backward()
        optimizer.step()
        
        # Update the learning rate.
        scheduler.step()

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
        for data, mask, target in validation_loader:
            # forward pass
            outputs = model(input_ids=data,attention_mask=mask,labels=target)
            # validation batch loss
            loss=outputs[0]
            logits=outputs[1]
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
        torch.save(model.state_dict(),'./bert_model_inf.pt')
    else:
        if(valid_loss<min_val_loss):
            epochs_no_improve=0
            min_val_loss=valid_loss
            torch.save(model.state_dict(),'./bert_model_inf.pt')
        else:
            epochs_no_improve+=1
            if(epochs_no_improve==n_epochs_stop):
                early_stop=True
                break
    if(early_stop):
        break
        
        
# 5) ----------- testing ---------
model.load_state_dict(torch.load('./bert_model_inf.pt'))
model.eval()
test_loss = 0
correct = 0

pred_list=[]
targ_list=[]
acc_list=[]
# turn off gradients for testing
with torch.no_grad():
    for data, mask, target in test_loader:
        outputs = model(input_ids=data,attention_mask=mask,labels=target)
        loss=outputs[0]
        logits=outputs[1]
        #loss = criterion(y_predicted, target) 
        # accumulate the valid_loss
        test_loss += loss.item()
        pred_acc=torch.argmax(logits,dim=1)
        pred_list.extend(pred_acc)
        targ_list.extend(target)

target_names=['Normal','Abusive', 'Hateful']
pred_tensor=torch.Tensor(pred_list)
targ_tensor=torch.Tensor(targ_list)
print(classification_report(targ_tensor,pred_tensor,target_names=target_names))

test_loss /= len(X_test)
print(f'Test loss: {test_loss}')   

plt.figure(figsize=(10,6))
plt.style.use('fivethirtyeight')
plt.plot(epoch_list, training_loss_list, label='Training Loss')
plt.plot(epoch_list, validation_loss_list, color='#444444', linestyle='--', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('BERT: Training and Validation Loss')
plt.legend()
plt.tight_layout()
plt.savefig('bert_model.png')
plt.show()
