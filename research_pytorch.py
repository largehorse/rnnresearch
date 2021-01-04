# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 17:17:58 2020

@author: Andrew
"""
import numpy as np
import torch
import torch.utils.data as data_utils
import torch.nn as nn

import tensorflow as tf
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
text = open("shakespeare.txt", 'rb').read().decode(encoding='utf-8')

ztarr = []
carr = []
fcarr = []
zmfcarr = []
epoch_idx = [0]


'''
Creates input and target text for use in training.
'''
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

'''
Takes file and converts it into batches of text, where the words are replaced by scalars. 
'''
def get_txt_data(text, batch_size, seq_size):
    text =text.replace('\r', '')
    vocab = sorted(set(text))

    idx2char = np.array(vocab)
    
    '''
    char_onehot = []
    for i in range(len(vocab)):
        temp = np.zeros(len(vocab))
        temp[i] = 1
        char_onehot.append(temp)
    
    #char_onehot = torch.Tensor(char_onehot)
    
    onehot_dict = {u:char_onehot[i] for i, u in enumerate(vocab)}
    text_as_onehot = np.array([onehot_dict[c] for c in text])
    '''
    
    char2idx = {u:i for i, u in enumerate(vocab)}
    txt_as_idx = [char2idx[c] for c in text]
    num_batches = int((len(txt_as_idx)-1)/(batch_size*seq_size))
    in_txt = txt_as_idx[:num_batches*batch_size*seq_size]
    out_txt = txt_as_idx[1:num_batches*batch_size*seq_size+1]

    in_txt = np.reshape(in_txt, (batch_size,-1))
    out_txt = np.reshape(out_txt, (batch_size,-1))
    
    return len(vocab), idx2char, char2idx, in_txt, out_txt

'''
Returns input and target batches.
'''
def get_batches(in_txt, out_txt, batch_size, seq_size):
    num_batches = np.prod(in_txt.shape) // (seq_size * batch_size)
    for i in range(0, num_batches*seq_size, seq_size):
        yield in_txt[:, i:i+seq_size], out_txt[:, i:i+seq_size]
        

def loss_optim(model, lr=0.001):
    crit = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    return crit, optim

'''
GRU implementation using PyTorch GRUcell. Does not include an embedding layer.
'''
class boxGRU(nn.Module):
    def __init__(self, batch_size, seq_size, input_size, embed_size, hidden_size):
        super(boxGRU, self).__init__()
        
        self.batch_size = batch_size
        self.seq_size = seq_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        
        #self.embedding = nn.Embedding(input_size, embed_size)
        self.gruCell = nn.GRUCell(input_size, hidden_size)
        self.dense = nn.Linear(hidden_size, input_size)
        
    def forward(self, x, hprev):
        
        output_seq = torch.empty((self.seq_size, self.batch_size, self.input_size)).to(device)
        h = hprev

        for i in range(len(x)):
            #xt = self.embedding(x[i])       
            
            xt = x[i].float().to(device)
            output = self.gruCell(xt, h)
            logit = self.dense(output)
            
            output_seq[i] = logit
            
        return output_seq.view((self.seq_size * self.batch_size, -1)), output
        
    def initHidden(self):
        return torch.zeros(self.batch_size, self.hidden_size)
    
    def initHiddenPredict(self):
        return torch.zeros(1, self.hidden_size)
    
    def predict(self, chars, seq_len, idx2char, char2idx, top_k=5):
        self.eval()
        seq = []
        chars = 'a'
        
        h = self.initHiddenPredict().to(device)
    
        seq.append(chars)
        
        #xt = np.array(char2idx[chars])
        xt = tf.keras.utils.to_categorical(char2idx[chars], num_classes=self.input_size)
        xt = torch.from_numpy(xt).unsqueeze(0)
        xt = xt.to(device)
        #print([xt].shape)
        
        output, h = self([xt], h)
    
        _, top = torch.topk(output, k=top_k)
        choices = top.tolist()
        choice = np.random.choice(choices[0])
        seq.append(idx2char[choice])
        
        for i in range(seq_len):
            xt = tf.keras.utils.to_categorical(choice, num_classes=self.input_size)
            xt = torch.from_numpy(xt).unsqueeze(0)
            xt = xt.to(device)
            
            output, h = self([xt], h)
            
            _, top = torch.topk(output, k=top_k)
            choices = top.tolist()
            choice = np.random.choice(choices[0])
            seq.append(idx2char[choice])
            
        print(''.join(seq))
        return seq

'''
GRU implementation using GRUcell with an embedding layer.
'''  
class myGRU(nn.Module):
    def __init__(self, batch_size, seq_size, input_size, embed_size, hidden_size):
        super(myGRU, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embed_size)
        
        self.batch_size = batch_size
        self.seq_size = seq_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.rx = nn.Linear(embed_size, hidden_size)
        self.rh = nn.Linear(hidden_size, hidden_size)
        self.zx = nn.Linear(embed_size, hidden_size)
        self.zh = nn.Linear(hidden_size, hidden_size)
        self.nx = nn.Linear(embed_size, hidden_size)
        self.nh = nn.Linear(hidden_size, hidden_size)
        
        self.dense = nn.Linear(hidden_size, input_size)
        
    def GRU_cell(self, inputx, hidden_in):

        
        rt = torch.sigmoid(self.rx(inputx) + self.rh(hidden_in))
        zt = torch.sigmoid(self.zx(inputx) + self.zh(hidden_in))
        #print(torch.mean(zt))
        ztarr.append(torch.mean(zt).item())
        nt = torch.tanh(self.nx(inputx) + torch.mul(rt, self.nh(hidden_in)))
        ht = (1-zt)*hidden_in + torch.mul(zt, nt)
        
        if epoch_idx[0] == 1:
            ztarr.append(torch.mean(zt).item())


        return ht
    
    def forward(self, x, hprev):

        output_seq = torch.empty((self.seq_size, self.batch_size, self.input_size)).to(device)
        h = hprev

        for i in range(len(x)):
            xt = x[i]
            xt = self.embedding(xt)      
            
            #xt = x[i].float().to(device)
            output = self.GRU_cell(xt, h)
            logit = self.dense(output)
            
            output_seq[i] = logit
            
        return output_seq.view((self.seq_size * self.batch_size, -1)), output
    
    def initHidden(self):
        return torch.zeros(self.batch_size, self.hidden_size)
    
    def initHiddenPredict(self):
        return torch.zeros(1, self.hidden_size)
    
    def predict(self, chars, seq_len, idx2char, char2idx, top_k=5):
        epoch_idx[0] = 2
        self.eval()
        seq = []
        chars = 'a'
        
        h = self.initHiddenPredict().to(device)
    
        seq.append(chars)
        
        #xt = np.array(char2idx[chars])
        #xt = tf.keras.utils.to_categorical(char2idx[chars], num_classes=self.input_size)
        xt = np.array(char2idx[chars])
        xt = torch.from_numpy(xt).type(torch.LongTensor).unsqueeze(0)
        xt = xt.to(device)
        #print([xt].shape)
        
        output, h = self([xt], h)
    
        _, top = torch.topk(output, k=top_k)
        choices = top.tolist()
        choice = np.random.choice(choices[0])
        seq.append(idx2char[choice])
        
        for i in range(seq_len):
            #xt = tf.keras.utils.to_categorical(choice, num_classes=self.input_size)
            xt = np.array(choice)
            xt = torch.from_numpy(xt).type(torch.LongTensor).unsqueeze(0)
            xt = xt.to(device)
            
            output, h = self([xt], h)
            
            _, top = torch.topk(output, k=top_k)
            choices = top.tolist()
            choice = np.random.choice(choices[0])
            seq.append(idx2char[choice])
            
        print(''.join(seq))
        return seq

'''
Custom GRU using ct_t. Includes an embedding layer.
'''        
class customGRU(nn.Module):
    def __init__(self, batch_size, seq_size, input_size, embed_size, hidden_size):
        super(customGRU, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embed_size)
        
        self.batch_size = batch_size
        self.seq_size = seq_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.rx = nn.Linear(embed_size, hidden_size)
        self.rh = nn.Linear(hidden_size, hidden_size)
        self.zx = nn.Linear(embed_size, hidden_size)
        self.zh = nn.Linear(hidden_size, hidden_size)
        self.nx = nn.Linear(embed_size, hidden_size)
        self.nh = nn.Linear(hidden_size, hidden_size)
        
        self.dense = nn.Linear(hidden_size, input_size)
        
    def GRU_cell(self, inputx, hidden_in):
    
        ct = 1 - ((self.c)/(self.c+1))**(2)
        #ct = self.c * (1 - self.prevz) + 1
        fc = torch.log(ct/(1-ct))
        #print(torch.mean(self.c))
        #print(torch.mean(fc))
        #print(ct)
        #print(fc)

        
        rt = torch.sigmoid(self.rx(inputx) + self.rh(hidden_in))
        zt = torch.sigmoid(self.zx(inputx) + self.zh(hidden_in) + fc)
        #print(zt)
        #print(torch.mean(zt).item())
        #print(torch.mean(torch.sigmoid(self.zx(inputx) + self.zh(hidden_in))))
        #print("")
        nt = torch.tanh(self.nx(inputx) + torch.mul(rt, self.nh(hidden_in)))
        ht = (1-zt)*hidden_in + torch.mul(zt, nt)
        
        if epoch_idx[0] == 1:
            ztarr.append(torch.mean(zt).item())
            carr.append(torch.mean(self.c).item())
            fcarr.append(torch.mean(fc).item())
            zmfcarr.append(torch.mean(torch.sigmoid(self.zx(inputx) + self.zh(hidden_in))).item())

        
        
        #temp = self.c * (1- zt) + 1
        #tp = temp.clone().detach()
        #self.c = tp
        tp = self.c.clone().detach()
        self.c = tp + 1
        
        #self.c += 0.1
        
        #print(ht.shape)

        return ht
    
    def forward(self, x, hprev):

        output_seq = torch.empty((self.seq_size, self.batch_size, self.input_size)).to(device)
        h = hprev

        for i in range(len(x)):
            xt = x[i]
            xt = self.embedding(xt)      
            
            #xt = x[i].float().to(device)
            output = self.GRU_cell(xt, h)
            logit = self.dense(output)
            
            output_seq[i] = logit
            
        return output_seq.view((self.seq_size * self.batch_size, -1)), output
    
    def initHidden(self):
        return torch.zeros(self.batch_size, self.hidden_size)
    
    def initHiddenPredict(self):
        return torch.zeros(1, self.hidden_size)
    
    def initC(self):
        self.c = torch.ones(self.batch_size, self.hidden_size).fill_(1.).to(device)
        self.c = self.c * 1
        
    
    def predict(self, chars, seq_len, idx2char, char2idx, top_k=5):
        self.eval()
        seq = []
        chars = 'a'
        
        h = self.initHiddenPredict().to(device)
        self.initC()
    
        seq.append(chars)
        
        #xt = np.array(char2idx[chars])
        #xt = tf.keras.utils.to_categorical(char2idx[chars], num_classes=self.input_size)
        xt = np.array(char2idx[chars])
        xt = torch.from_numpy(xt).type(torch.LongTensor).unsqueeze(0)
        xt = xt.to(device)
        #print([xt].shape)
        
        output, h = self([xt], h)
    
        _, top = torch.topk(output, k=top_k)
        choices = top.tolist()
        choice = np.random.choice(choices[0])
        seq.append(idx2char[choice])
        
        for i in range(seq_len):
            #xt = tf.keras.utils.to_categorical(choice, num_classes=self.input_size)
            xt = np.array(choice)
            xt = torch.from_numpy(xt).type(torch.LongTensor).unsqueeze(0)
            xt = xt.to(device)
            
            output, h = self([xt], h)
            
            _, top = torch.topk(output, k=top_k)
            choices = top.tolist()
            choice = np.random.choice(choices[0])
            seq.append(idx2char[choice])
            
        print(''.join(seq))
        return seq    
    

def predict1(device, model, start_chars_orig, input_size, char2idx, idx2char, top_k=3, pred=400):
    model.eval()
    h_state = model.initHidden(16)
    h_state = h_state.to(device)
    
    start_chars = start_chars_orig.copy()
    
    for c in start_chars:
        ix = torch.tensor([char2idx[c]]).long().to(device).reshape(1,)
        print(ix.shape)
        output, h_state = model(ix, h_state)

    _, top = torch.topk(output[0], k=top_k)
    choices = top.tolist()
    choice = np.random.choice(choices[0])
    start_chars.append(idx2char[choice])
    
    for i in range(pred):
        ix = torch.tensor([[choice]]).long().to(device)
        output, h_state = model(ix, h_state)
        _, top = torch.topk(output[0], k=top_k)
        choices = top.tolist()
        choice = np.random.choice(choices[0])
        start_chars.append(idx2char[choice])
    

    print(''.join(start_chars))
    


'''
Training loop. Also includes text generation after each epoch.
'''

def train_loop():
    BATCH_SIZE = 256
    SEQ_SIZE = 1
    GRU_HIDDEN = 128
    EMBEDDING_SIZE = 128
    EPOCHS = 1
    PREDICT_LEN = 400
    start_chars = ['R', 'O', 'M', 'E', 'O', ':']
    
    input_size, idx2char, char2idx, in_txt, out_txt = get_txt_data(text, BATCH_SIZE, SEQ_SIZE)
    model = customGRU(BATCH_SIZE, SEQ_SIZE, input_size, EMBEDDING_SIZE, GRU_HIDDEN)
    model = model.to(device)
    crit, optim = loss_optim(model, 0.001)
    
    for epoch in range(EPOCHS):
        epoch_idx[0] += 1
        h = model.initHidden()
        h = h.to(device)
        model.initC()

        batches = get_batches(in_txt, out_txt, BATCH_SIZE, SEQ_SIZE)
        print(epoch)
        
        for i, (x, y) in enumerate(batches):
            valx = torch.from_numpy(x).transpose(1, 0).type(torch.LongTensor).to(device)
            valy = torch.from_numpy(y.T).type(torch.LongTensor).to(device)

            optim.zero_grad()
            
            output, h = model(valx, h)
            loss = crit(output, valy.contiguous().view(BATCH_SIZE * SEQ_SIZE))
    
            loss.backward()
            h = h.detach()
            optim.step()
            
        print("loss: {}".format(loss.item()))

        epoch_idx[0] += 1
        model.predict(start_chars, PREDICT_LEN, idx2char, char2idx, top_k=3)
        
    
    
    
    
    
train_loop()

import matplotlib.pyplot as plt
plt.plot(ztarr)
plt.title("Z_t of custom GRU on first epoch")
plt.xlabel("iteration")
plt.ylabel("z_t value")
plt.show()

plt.plot(carr)
plt.title("c_t of custom GRU on first epoch")
plt.xlabel("iteration")
plt.ylabel("c_t value")
plt.show()

plt.plot(fcarr)
plt.title("f(c_t) of custom GRU on first epoch")
plt.xlabel("iteration")
plt.ylabel("z_t value")
plt.show()

plt.plot(zmfcarr)
plt.title("Z_t no f(c) of custom GRU on first epoch")
plt.xlabel("iteration")
plt.ylabel("z_t value")
plt.show()
    
    
    
    
    
    
'''
input_size, idx2char, char2idx, in_txt, out_txt = get_txt_data(text, BATCH_SIZE, SEQ_SIZE)
model = boxGRU(input_size, EMBEDDING_SIZE, GRU_HIDDEN)
model = model.to(device)

criterion, optimizer = loss_optim(model, 0.001)
iteration = 0

for e in range(EPOCHS):
    batches = get_batches(in_txt, out_txt, BATCH_SIZE, SEQ_SIZE)
    h_state = model.initHidden(BATCH_SIZE)
    h_state = h_state.to(device)
    
    for x,y in batches:
        iteration += 1
        print(iteration)
        model.train()
        optimizer.zero_grad()

        for i in range(int(PREDICT_LEN/40)):
            xt = x[:, i]
            yt = y[:, i]
            #print(xt)
            #print(yt)
            xt = torch.tensor(xt).long().to(device)
            yt = torch.tensor(yt).long().to(device)

            logits, h_state = model(xt, h_state)
            
            #print(logits.shape)
            #print(yt.shape)
        loss = criterion(logits, yt)
        h_state = h_state.detach()
        
        loss_val = loss.item()
        loss.backward()
        optimizer.step()
        
        
    print('')
    print('Epoch: {}/{}'.format(e, EPOCHS),
          'Iteration: {}'.format(iteration),
          'Loss: {}'.format(loss_val))
            
        
    predict(device, model, start_chars, input_size,
            char2idx, idx2char, top_k=3, pred=PREDICT_LEN)
    torch.save(model.state_dict(),
               'checkpoint_pt/model-{}.pth'.format(iteration))

'''

"""
data1 = dataiter.next()
txt = ""
for i in range(len(data1)):
    c = np.argmax(data1[i])
    txt = txt + idx2char[c]
print(txt)
"""

