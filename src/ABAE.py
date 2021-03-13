# ABAE.py
#Attention-based Aspect Extraction

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import tqdm.notebook as tqdm
import numpy as np

import datasets


class ABAENet(nn.Module):
    """
    Neural Net associated to Attention Based Aspect Extraction
    """
    def __init__(self, K, max_length, dim_emb = 300, reg = 1., **kwargs):
        super(ABAENet, self).__init__()
        
        self.max_length = max_length #maximal number of token per sentence
        self.dim_emb = dim_emb #Dim of the embedding
        self.K = K #number of aspects we want to extract
        self.reg = reg
        
        self.linM = nn.Linear(dim_emb, dim_emb, bias = False)
        self.linW = nn.Linear(dim_emb, K)
        self.linT = nn.Linear(K, dim_emb, bias = False)
          
        
    def forward(self, e_w):
        """
            The net take as input word embedding (for now)
            e_w = [batch_size, max_length, dim]
        """
        
        #Attention Weights
        y_s = torch.mean(e_w, axis = 1)
        
        di = torch.bmm(e_w, self.linM(y_s).view(-1, self.dim_emb, 1))
        
        a = torch.softmax(di.squeeze(), axis = 1)
        
        #Sentence Embedding
        z_s = torch.bmm(a.unsqueeze(dim = 1), e_w).view(-1, self.dim_emb) # [batch_size, dim]
        
        #Sentence Reconstruction
        p_t = torch.softmax(self.linW(z_s), axis = 1) #[batch_size, K]
        
        r_s = self.linT(p_t) #[batch_size, dim]
        
        return z_s, r_s
        
    def get_T(self):
        return self.linT.weight
        
        
    def MaxMargin(self, e_w, e_wn):
        """
            e_w = [batch_size, max_length, dim] 
            e_wn = [m, max_length, dim] negative samples
        """
        
        z_s, r_s = self.forward(e_w) # [batch_size, dim], [batch_size, dim]
        
        z_n = torch.mean(e_wn, axis = 1) # [m, dim]
        
        #/!\ Potential renormalization (cf. l. 171, my_layers.py) /!\
        
        pos = torch.sum(r_s*z_s, axis=1)
        neg = torch.sum(r_s[:,None,:]*z_n[None,:,:], axis= 2)
        
        loss = torch.max(torch.zeros_like(neg), torch.ones_like(neg) - pos[:, None] + neg)

        return torch.sum(loss)
        
    def RegLoss(self):
        
        T = self.get_T()
        T_n = T/T.sum(axis=0)[None, :] #rk : T is, compare to the article, T^t
        
        U = torch.norm(torch.mm(torch.transpose(T_n, 0, 1), T_n) - torch.eye(self.K).cuda()) #Not clean but you know ...
        
        return U
        
    def TotalLoss(self, e_w, e_wn):
        return self.MaxMargin(e_w, e_wn) + self.reg*self.RegLoss()


class TextData(Dataset):
    def __init__(self, dataset, transform=None, max_length = 512, train = True):
        self.transform = transform
        self.max_length = max_length
        
        self.data = dataset
            
    def __padding__(self, np_arr):
        """
            Add value (0,...,0) to equalize all lengths
            inputs:
                np_arr [n_words, dim]
            returns:
                pad_arr [max_length, dim]
        """
        
        n_words, d = np_arr.shape
        
        n2add = max(0,self.max_length - n_words)
        
        return np.concatenate((np_arr, np.zeros((n2add, d))), axis = 0)[:self.max_length]
        
    def __getitem__(self, index):

        sentence = self.data[index]
        emb_sentence = self.transform(sentence)
        pad_sentence = self.__padding__(emb_sentence)
        
        return torch.from_numpy(pad_sentence).float()

    def __len__(self):
        return len(self.data)



class ABAE:
    """
        Detect aspect by applying ABAE.
    """
    def __init__(self, emb, dataset = None, k = 5, emb_name = 'w2v', language="english", dist="L2", **kwargs):
        
        self.emb_name = emb_name
        self.k = k
        self.language = language
        self.dist = dist
        
        if self.emb_name == 'w2v':
            self.w2v = emb
            self.stop_words = set(stopwords.words(language))
            valid_words = [word for word in self.w2v.index if word not in self.stop_words and word.isalnum()]
            self.w2v = self.w2v.loc[valid_words]
            self.w2v_words = set(valid_words)
        else:
            raise Exception("Not Implemented Yet.")
        
        self.kwargs = kwargs
        
        self.dataset = dataset
        
        self.net = ABAENet(K = k, max_length = kwargs['max_length'], dim_emb = kwargs['dim_emb'], reg = kwargs['reg']).cuda() # /!\ to change ... /!\
        
        
    def sentence_emb_w2v(self, sentence):
        word_tokens = word_tokenize(sentence)  
        filtered_sentence = [w for w in word_tokens if not w in self.stop_words]  
        filtered_sentence = [word for word in filtered_sentence if word.isalnum()]
        filtered_sentence = [word for word in filtered_sentence if word in self.w2v_words]
        vectors = self.w2v.loc[filtered_sentence] #panda dataframe of word/w2v embedding
        arr = vectors.to_numpy() # [len_sentence, dim_emb] 
        return arr
        
    def _create_dataset(self):
        
        if self.emb_name == 'w2v':
            text_dataset = TextData(self.dataset, transform = self.sentence_emb_w2v, max_length = self.kwargs['max_length'])
        else:
            raise Exception("Not Implemented Yet.")
        
        return text_dataset
        
    def _train_one_epoch(self, train_loader, neg_loader, optimizer, silent = True):
        
        data_size = len(train_loader)
        
        loss_ep = 0.
        
        for batch_idx, e_w in tqdm.tqdm(enumerate(train_loader), leave = False, total = data_size, disable = silent):
            #sample negative samples
            e_wn = iter(neg_loader).next()
            #Device
            e_w = e_w.cuda()
            e_wn = e_wn.cuda()
            
            #Loss & Optim
            optimizer.zero_grad()
            
            loss = self.net.TotalLoss(e_w, e_wn) 

            loss.backward()
            optimizer.step()
            
            loss_ep += loss.cpu().float().item()
        
        return loss_ep/data_size
        
        
    def train(self, silent = True, path = 'models/ABAE.pt'):
        print("Initializing Training ...")
        
        num_epochs = self.kwargs['epochs']
        
        train_loader = torch.utils.data.DataLoader(self._create_dataset(), batch_size=self.kwargs['batch_size'], shuffle=True, num_workers=1)
        
        neg_loader = torch.utils.data.DataLoader(self._create_dataset(), batch_size=self.kwargs['neg_m'], shuffle=True, num_workers=1)
        
        optimizer = optim.Adam(self.net.parameters(), lr=self.kwargs['lr'], betas=(0.9, 0.999))
        
        self.net.train()
        
        print("Training Started !")
        print('')
        print("####################################################")
        print('')
        
        for ep in tqdm.tqdm(range(num_epochs), disable = silent):
            loss_ep = self._train_one_epoch(train_loader, neg_loader, optimizer, silent = True)
            print('Epoch: {}/{}, Loss: {:.4f}'.format(ep, num_epochs, loss_ep))
        
        print('')
        print("####################################################")
        print('')
        print("Saving Model Weights ...")
        torch.save(self.net.state_dict(), path)
        print("Done ! ")
        
        
    def _retrieve_word(self, aspects_vector):
        """
        Retrieve word closest to a vector
        """
        if self.dist == "cosin":
            scores = (self.w2v @ aspects_vector)
        else:
            assert self.dist == "L2", f"Unknown distance {self.dist}"
            diff = self.w2v - aspects_vector
            scores = -(diff*diff).sum(axis=1)
            
        return scores.index[scores.argmax()]
        
        
    def predict_sentence(self, sentence, model_path = 'models/ABAE.pt'):
        #Load Model
        model = self.net
        model.load_state_dict(torch.load(model_path))
        
        T = model.get_T().detach().cpu().numpy() 
        aspects_vectors = T.transpose(1,0)
        aspects = []
        for i in range(self.k):
            aspect = self._retrieve_word(aspects_vectors[i])
            aspects.append(aspect)
        
        return aspects 
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        