#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn


# In[8]:


if torch.cuda.is_available():
    device='cuda'
else:
    device='cpu'
print(device)
#thanks to aladdin persson


# Self Attention

# In[37]:


class SelfAttention(nn.Module):
    def __init__(self,embed_size,num_heads):#num heads is number of parts in which embedding is splitted
        super(SelfAttention,self).__init__()
        self.embed_size=embed_size
        self.num_heads=num_heads
        self.head_dimension=embed_size//num_heads
        
        assert (self.head_dimension*num_heads==embed_size),"Embed size needs to be divisible by number of heads"
        
        self.values=nn.Linear(self.head_dimension,self.head_dimension,bias=False)
        
        self.keys=nn.Linear(self.head_dimension,self.head_dimension,bias=False)
        
        self.queries=nn.Linear(self.head_dimension,self.head_dimension,bias=False)
        
        self.fc_out=nn.Linear(num_heads*self.head_dimension,embed_size)
        
    def forward(self,values,keys,queries,mask):
        
        num_examples_N=queries.shape[0]
        
        value_length,key_length,query_length=values.shape[1],keys.shape[1],queries.shape[1]
        
        #split embedding into self.num_heads pieces
        
        values=values.reshape(num_examples_N,value_length,self.num_heads,self.head_dimension)
        
        keys=keys.reshape(num_examples_N,key_length,self.num_heads,self.head_dimension)
        
        queries=queries.reshape(num_examples_N,query_length,self.num_heads,self.head_dimension)
        #op from queries * keys
        energy=torch.einsum("nqhd,nkhd->nhqk",[queries,keys])
        
        #query shape = N,query_len,head dim, same for keys 
        #keys shape=N,heads,query len,key len
        #energy shape : N,heads,querylen,keylen
        #for each word in target how much attention will be given to each word in input
        #torch.bmm is batch matrix multiply
        
        if mask is not None:
            energy=energy.masked_fill(mask==0,float("-1e28"))
        
        attention=torch.softmax(energy/(self.embed_size**(0.5)),dim=3)
        
        out=torch.einsum("nhql,nlhd->nqhd",[attention,values]).reshape(
            num_examples_N,query_length,self.num_heads*self.head_dimension)
        
        #attention shape=(n,heads,,querylen,keylen)
        #values shape=(n,value_len,heads,head_dim)
        #want n,querylen,heads,headdim
        
        out=self.fc_out(out)
        return out


# Transformer Block

# In[17]:


class TransformerBlock(nn.Module):
    def __init__(self,embed_size,heads,dropout,forward_expansion):
        super(TransformerBlock,self).__init__()
        
        self.attention=SelfAttention(embed_size,heads)
        
        self.norm1=nn.LayerNorm(embed_size)
        self.norm2=nn.LayerNorm(embed_size)
        
        self.feed_forward=nn.Sequential(
        nn.Linear(embed_size,forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size,embed_size)
        )
        
        self.dropout=nn.Dropout(dropout)
        
    def forward(self,value,key,query,mask):
        attention=self.attention(value,key,query,mask)
        x=self.dropout(self.norm1(attention+query))
        forward=self.feed_forward(x)
        out=self.dropout(self.norm2(forward+x))
        return out


# Encoder

# In[16]:


class Encoder(nn.Module):
    def __init__(
    self,
    source_vocab_size,
    embed_size,
    num_layers,
    num_heads,
    device,
    forward_expansion,
    dropout,
    max_length,):
        super(Encoder,self).__init__()
        
        self.embed_size=embed_size
        self.device=device
        self.word_embedding=nn.Embedding(source_vocab_size,embed_size)
        self.position_embedding=nn.Embedding(max_length,embed_size)
        
        self.layers=nn.ModuleList(
        [
            TransformerBlock(embed_size,
                            num_heads,
                            dropout=dropout,
                            forward_expansion=forward_expansion,
                            )
        ])
        
        self.dropout=nn.Dropout(dropout)
        
    def forward(self,x,mask):
            N,seq_length=x.shape
            
            positions=torch.arange(0,seq_length).expand(N,seq_length).to(self.device)
            
            out=self.dropout(self.word_embedding(x)+ self.position_embedding(positions))
            
            for layer in self.layers:
                out=layer(out,out,out,mask)
                
            return out


# In[34]:


class DecoderBlock(nn.Module):
    def __init__(self,embed_size,heads,forward_expansion,dropout,device):
        super(DecoderBlock,self).__init__()
        self.norm=nn.LayerNorm(embed_size)
        self.attention = SelfAttention(embed_size,heads)
        self.transformer_block=TransformerBlock(embed_size,heads,dropout,forward_expansion)
        self.dropout=nn.Dropout(dropout)
        
    def forward(self,x,value,key,src_mask,target_mask):
        attention=self.attention(x,x,x,target_mask)
        query=self.dropout(self.norm(attention+x))
        out=self.transformer_block(value,key,query,src_mask)
        return out


# In[26]:


class Decoder(nn.Module):
    def __init__(self,
                target_vocab_size,
                embed_size,
                num_layers,
                num_heads,
                forward_expansion,
                dropout,
                device,
                max_length,
                ):
        super(Decoder,self).__init__()
        self.device=device
        self.word_embedding=nn.Embedding(target_vocab_size,embed_size)
        self.position_embedding=nn.Embedding(max_length,embed_size)
        
        self.layers=nn.ModuleList(
        [DecoderBlock(embed_size,num_heads,forward_expansion,dropout,device)
        for _ in range(num_layers)])
        
        self.fc_out=nn.Linear(embed_size,target_vocab_size)
        self.dropout=nn.Dropout(dropout)
        
        
    def forward(self,x,enc_out,source_mask,target_mask):
        N,seq_length=x.shape
        positions=torch.arange(0,seq_length).expand(N,seq_length).to(self.device)
        x=self.dropout((self.word_embedding(x) + self.position_embedding(positions)))
        
        for layer in self.layers:
            x=layer(x,enc_out,enc_out,source_mask,target_mask)
            
        out=self.fc_out(x)
        
        return out


# In[32]:


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=512,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        device="cuda",
        max_length=100,
    ):

        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length,
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )

        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out


# In[38]:


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(
        device
    )
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to(
        device
    )
    out = model(x, trg[:, :-1])
    print(out.shape)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




