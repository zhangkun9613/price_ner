import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from vocab import Vocab

class BiLSTM(nn.Module):
    def __init__(self, vocab, slot_vocab, embedding_dim=100, hidden_dim=256,device = 0):
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
                            num_layers=2, dropout = 0.8, bidirectional=True)
        
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        self.loss_func = nn.CrossEntropyLoss(ignore_index=self.slot_vocab['<pad>'])

    def _get_lstm_features(self, sentences):
#         self.hidden = self.init_hidden()
        seq_lens = [len(s) for s in sentences]
        sents_tensor = self.vocab.to_input_tensor(sentences)
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
    
    def _calcu_loss(self,feats,labels,mask=True):
        if mask:
            slot_mask = labels.view(-1)!=self.slot_vocab['<pad>']
            active_feats = feats.view(-1,self.tagset_size)[slot_mask]
            active_labels = labels.contiguous().view(-1)[slot_mask]
            loss = self.loss_func(active_feats, active_labels)
        else:
            loss = self.loss_func(feats.view(-1,self.tagset_size),labels.contiguous().view(-1))
        return loss
    
    def predict_slot(self,feats):
        #feats [b,len,tag_size]
        _, preds = torch.max(feats,dim=-1)
        return preds
    
    def predict(self,sentence):
        lstm_feats = self._get_lstm_features(sentence)
        preds = self.predict_slot(lstm_feats) #[b,len]
        return preds
    
    def forward(self, sentence, labels):  
        # Get the emission scores from the BiLSTM
        labels = self.slot_vocab.to_input_tensor(labels) # labels [len,b]
        labels = labels.t()
        labels = labels.to(self.device)
        lstm_feats = self._get_lstm_features(sentence)
        loss = self._calcu_loss(lstm_feats,labels)
        preds = self.predict_slot(lstm_feats) #[b,len]
        
        slot_mask = labels.view(-1)!=self.slot_vocab['<pad>']
        acc = (labels.view(-1)[slot_mask] == preds.view(-1)[slot_mask])
        acc = acc.sum().float()/slot_mask.sum()
        return loss,acc
