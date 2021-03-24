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

import tqdm as tqdm
import numpy as np

import datasets


def get_mask(embedings):
    """
    Compute corresponding to the embeded sentence.
    embedings is [batch_size, max_length, dim]
    Result is [batch_size, max_length, 1]
    """
    max_value = torch.max(torch.abs(embedings),dim=-1, keepdim=True)[0]
    return (max_value >= 1e-6)
        
def masked_soft_mask(x, mask, dim):
    """
    A mask soft max, with overflow protection.
    """
    x_max = torch.max(x,dim=dim,keepdim=True)[0]
    x_exp = torch.exp(x-x_max) * mask
    x_softmax = x_exp / torch.sum(x_exp,dim=dim,keepdim=True)
    return x_softmax

def masked_mean(x, mask, dim):
    x_sum = torch.sum(x, dim=dim)
    return x_sum / torch.sum(mask, dim=dim)

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
        self.register_parameter(name='T', param=torch.nn.Parameter(torch.rand(dim_emb, K)))


    def forward(self, e_w):
        """
            The net take as input word embeddings (for now)
            e_w = [batch_size, max_length, dim]
            Assumed that masked tokens are full of 0
        """
        mask = get_mask(e_w)
        #Attention Weights
        y_s = masked_mean(e_w, mask, dim = 1) # shape: (batch_size, dim)
        
        # null words stay null because of matrix multiplication
        di = torch.bmm(e_w, self.linM(y_s).view(-1, self.dim_emb, 1)) # shape: (batch_size, max_length, 1)
                
        a = masked_soft_mask(di.squeeze(), mask.squeeze(), dim = 1)# shape: (batch_size, max_length)
        
        #Sentence Embedding
        z_s = torch.bmm(a.unsqueeze(dim = 1), e_w).view(-1, self.dim_emb) # [batch_size, dim]
        
        #Sentence Reconstruction
        p_t = torch.softmax(self.linW(z_s), axis = 1) #[batch_size, K]
        
        normed_t = self.T / torch.norm(self.T, keepdim=True, dim=0)
        r_s = p_t @ torch.transpose(normed_t,0,1)
        
        return z_s, r_s
 

    def get_T(self):
        #return self.linT.weight
        return self.T
        
        
    def MaxMargin(self, e_w, e_wn):
        """
            e_w = [batch_size, max_length, dim] 
            e_wn = [m, max_length, dim] negative samples
        """
        
        mask_w = get_mask(e_w)
        mask_wn = get_mask(e_wn)
        z_s, r_s = self.forward(e_w) # [batch_size, dim], [batch_size, dim]
        
        z_n = masked_mean(e_wn, mask_wn, dim = 1) # [m, dim]
        
        #/!\ Potential renormalization (cf. l. 171, my_layers.py) /!\
        
        pos = torch.sum(r_s*z_s, axis=1)
        neg = torch.sum(r_s[:,None,:]*z_n[None,:,:], axis= 2)
        
        loss = torch.max(torch.zeros_like(neg), torch.ones_like(neg) - pos[:, None] + neg)

        return torch.sum(loss)
        
    def RegLoss(self):
        
        T = self.get_T()
        T_norm = torch.sqrt(torch.sum(T*T, axis=0) + 1e-3)
        #T_n = T/T.sum(axis=0)[None, :] #rk : T is, compare to the article, T^t
        T_n = T/T_norm[None, :] #rk : T is, compare to the article, T^t
        
        U = torch.norm(torch.mm(torch.transpose(T_n, 0, 1), T_n) - torch.eye(self.K).cuda()) #Not clean but you know ...
        
        return U
        
    def TotalLoss(self, e_w, e_wn):
        return self.MaxMargin(e_w, e_wn) + self.reg*self.RegLoss()

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
        res = np.zeros(self.kwargs['max_length'], dtype=np.int64)
        res[:len(filtered_sentence)] = [self.word_to_idx[token] for token in filtered_sentence[:self.kwargs['max_length']]]        
        return res
        
    def _create_dataset(self, silent=True):
        if self.emb_name == 'w2v':
            self.word_to_idx = {w:i+1 for i,w in enumerate(self.w2v.index)}
            text_dataset = [self.sentence_emb_w2v(sentence) for sentence in tqdm.tqdm(self.dataset, disable=silent)]
            word_matrix = np.zeros((self.w2v.shape[0] + 1, self.w2v.shape[1]), dtype=np.float32)
            word_matrix[1:] = self.w2v.values
        else:
            raise Exception("Not Implemented Yet.")
        
        return text_dataset, word_matrix
        
    def _train_one_epoch(self, train_loader, neg_loader, word_matrix, optimizer, silent = True):
        
        data_size = len(train_loader)
        
        loss_ep = 0.
        
        for batch_idx, e_w in tqdm.tqdm(enumerate(train_loader), leave = False, total = data_size, disable = silent):
            #sample negative samples
            e_wn = iter(neg_loader).next()
            e_w = e_w.cuda()
            e_wn = e_wn.cuda()
            e_w = word_matrix[e_w]
            e_wn = word_matrix[e_wn]
            
            #Loss & Optim
            optimizer.zero_grad()
            
            loss = self.net.TotalLoss(e_w, e_wn) 

            loss.backward()
            optimizer.step()
            
            loss_ep += loss.cpu().float().item()
        
        return loss_ep/data_size
        
        
    def train(self, silent = False, path = 'models/ABAE.pt'):
        print("Initializing Training ...")
        
        num_epochs = self.kwargs['epochs']
        
        dataset, word_matrix = self._create_dataset(silent=silent)
        word_matrix = torch.Tensor(word_matrix).cuda()
        
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.kwargs['batch_size'], shuffle=True, num_workers=1)
        
        neg_loader = torch.utils.data.DataLoader(dataset, batch_size=self.kwargs['neg_m'], shuffle=True, num_workers=1)
        
        optimizer = optim.Adam(self.net.parameters(), lr=self.kwargs['lr'], betas=(0.9, 0.999))
        
        self.net.train()
        
        print("Training Started !")
        print('')
        print("####################################################")
        print('')
        
        for ep in tqdm.tqdm(range(num_epochs), disable = silent):
            T = self.net.get_T().detach().cpu().numpy() 
            aspects_vectors = T.transpose(1,0)
            aspects = []
            for i in range(self.k):
                aspect = self._retrieve_word(aspects_vectors[i])
                aspects.append(aspect)
            print(aspects)
            
            loss_ep = self._train_one_epoch(train_loader, neg_loader, word_matrix, optimizer, silent = True)
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
            norms_w2v = np.linalg.norm(self.w2v.values, axis=1)
            norms_aspect = np.linalg.norm(aspects_vector)
            scores = (self.w2v @ (aspects_vector / norms_aspect)) / norms_w2v
        else:
            assert self.dist == "L2", f"Unknown distance {self.dist}"
            diff = self.w2v - aspects_vector
            scores = -(diff*diff).sum(axis=1)
            
        return scores.index[scores.argmax()]
        
        
    def predict_aspect(self, model_path = 'models/ABAE.pt'):
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
        
        
  
        
        
        
        
        
   
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        