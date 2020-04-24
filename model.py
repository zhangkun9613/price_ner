import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from vocab import Vocab

class BiLSTM(nn.Module):
    def __init__(self, vocab, slot_vocab, embedding_dim=100, hidden_dim=256,device = -1,loss_func = ''):
        super(BiLSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = len(vocab.word2id)
        self.slot_vocab = slot_vocab
        self.tagset_size = len(slot_vocab.word2id)
        self.vocab = vocab
        self.device = device

        self.word_embeds = nn.Embedding(self.vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)
        
        #focal loss parm
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        if loss_func == 'focal':
            self.loss_func = self._focal_loss
            self.gamma = 1
            self.focal_weight = torch.ones(1,self.tagset_size)
        elif loss_func == 'f1':
            self.loss_func = self._f1_loss
        else:
            self.loss_func = nn.CrossEntropyLoss(ignore_index=self.slot_vocab['<pad>'])

    def _get_lstm_features(self, sentences):
#         self.hidden = self.init_hidden()
        seq_lens = [len(s) for s in sentences]
        sents_tensor = self.vocab.to_input_tensor(sentences)
        if self.device >= 0:
            sents_tensor = sents_tensor.to(self.device)
        #sents_tensor [len,b,embed]
        embeds = self.word_embeds(sents_tensor)
        embeds = pack_padded_sequence(embeds,seq_lens)
        lstm_out, _ = self.lstm(embeds)
        lstm_out, _ = pad_packed_sequence(lstm_out,batch_first=True)
        
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats
    
#     def _generate_sent_masks(self, enc_hiddens,source_lengths):
#         enc_masks = torch.zeros(enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float)
#         for e_id, src_len in enumerate(source_lengths):
#             enc_masks[e_id, src_len:] = 1
#         return enc_masks.to(self.device)
    
    
    def _focal_loss(self,feats,labels):
        # feats [nums,tag_size] labels [nums] nums = b * max_len
        logit = nn.functional.softmax(feats,dim=-1) * self.focal_weight
        probs = torch.gather(logit,dim=-1,index = labels.unsqueeze(-1)).squeeze(-1)
        loss = -torch.pow(1-probs,self.gamma) * torch.log(probs)
        loss = torch.sum(loss)/labels.shape[0]
        return loss
       
    def _f1_loss(self,feats,labels):
        # 有两个需要考虑的地方，一个是是否clamp，另一个是用sigmoid还是softmax
        # feats [nums,tag_size] labels [nums] nums = b * max_len
        ids = list(self.slot_vocab.word2id.values())
        # get valid_id
        ids.remove(self.slot_vocab['<pad>']);ids.remove(self.slot_vocab['<unk>']) 
        
        probs = torch.sigmoid(feats)        
#         probs = nn.functional.softmax(feats,dim=-1)  # 使用softmax会提高召回率，降低准确率 
#         probs[:,[self.slot_vocab['<pad>'], self.slot_vocab['unk']]] = -np.inf
        targets = torch.zeros_like(probs).scatter_(1,labels.unsqueeze(-1),1) # num,tag_size
        
        #remove pad,unk
        probs = probs[:,ids]; targets = targets[:,ids] 
        
#         probs = torch.clamp(probs * (1-targets),min = 0.01) + probs * targets
        epsilon = 1e-8
        tp = torch.sum(probs * targets,dim = 0)
        precision = tp / (probs.sum(dim=0) + epsilon)
        recall = tp / (targets.sum(dim=0) + epsilon)
        f1 = 2 * (precision * recall / (precision + recall + epsilon))
        return 1 - f1.mean()
           
    
    def _calcu_loss(self,feats,labels,mask=True):
        if mask:
            slot_mask = labels.view(-1)!=self.slot_vocab['<pad>']
#             feats[:,:,[self.slot_vocab['<pad>'], self.slot_vocab['unk']]] = -np.inf
            active_feats = feats.view(-1,self.tagset_size)[slot_mask]
            active_labels = labels.contiguous().view(-1)[slot_mask]
#             print(active_feats[0])
            loss = self.loss_func(active_feats, active_labels)
        else:
            loss = self.loss_func(feats.view(-1,self.tagset_size),labels.contiguous().view(-1))
        return loss
    
    def predict_slots(self,feats):
        #feats [b,len,tag_size]
        feats[:,:,[self.slot_vocab['<pad>'], self.slot_vocab['unk']]] = -np.inf
        _, preds = torch.max(feats,dim=-1)
        return preds
    
    def predict(self,sentence):
        lstm_feats = self._get_lstm_features(sentence)
        preds = self.predict_slots(lstm_feats) #[b,len]
        return preds
    
    def forward(self, sentence, labels):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        labels = self.slot_vocab.to_input_tensor(labels) # labels [len,b]
        if self.device >= 0:
            labels = labels.to(self.device)
        labels = labels.t()
        lstm_feats = self._get_lstm_features(sentence)
        loss = self._calcu_loss(lstm_feats,labels)
        preds = self.predict_slots(lstm_feats) #[b,len]
        
        slot_mask = labels.view(-1)!=self.slot_vocab['<pad>']
        acc = (labels.view(-1)[slot_mask] == preds.view(-1)[slot_mask])
        acc = acc.sum().float()/slot_mask.sum()
        return loss,acc