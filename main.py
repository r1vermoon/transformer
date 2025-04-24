import torch
import math
import spacy
from torch import nn

from torch import optim
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence
import time
from torchtext.datasets import Multi30k 
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
import math  
from collections import Counter  # 用于统计n-gram计数
import numpy as np 
import matplotlib.pyplot as plt 
from test import args

batch_size = 128
max_len = 256
d_model = 512
n_layers = 6
n_heads = 8
ffn_hidden = 2048
drop_prob = 0.1

init_lr = 1e-5
factor = 0.9
adam_eps = 5e-9
patience = 10
warmup = 100
epoch = 100
clip = 1.0
weight_decay = 5e-4
inf = float('inf')
print(args.epoch)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device:{device}")

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # '-1' means last dimension. 
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out
    
class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention,self).__init__()
        self.softmax=nn.Softmax(dim=-1)
    
    def forward(self,q,k,v,mask=None,e=1e-12):
        batch_size,head,length,d_tensor=k.size()
        k_t=k.transpose(2,3)
        score=(q@k_t)/math.sqrt(d_tensor)
        
        if mask is not None:
            score=score.masked_fill(mask==0,-10000)
            
        score=self.softmax(score)
        v=score@v
        
        return v,score
    
class PositionwiseFeedForward(nn.Module):
    
    def __init__(self,d_model,hidden,drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self,x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
class MultiHeadAttention(nn.Module):
    
    def __init__(self,n_head,d_model):
        super(MultiHeadAttention,self).__init__()
        self.n_head=n_head
        self.attention=ScaleDotProductAttention()
        self.w_q=nn.Linear(d_model,d_model)
        self.w_k=nn.Linear(d_model,d_model)
        self.w_v=nn.Linear(d_model,d_model)
        self.w_concat=nn.Linear(d_model,d_model) #linear after concat
    
    def split(self,tensor):
        batch_size,length,d_model=tensor.size()
        d_tensor=d_model//self.n_head
        tensor=tensor.view(batch_size,length,self.n_head,d_tensor).transpose(1,2)
        return tensor
    
    def concat(self, tensor):
        batch_size,head,length,d_tensor=tensor.size()
        d_model=head*d_tensor
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor
            
    def forward(self,q,k,v,mask=None):
        q,k,v=self.w_q(q),self.w_k(k),self.w_v(v)
        q,k,v=self.split(q),self.split(k),self.split(v)
        out, attention = self.attention(q, k, v, mask=mask)
        
        out = self.concat(out)
        out = self.w_concat(out)
        return out

class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, max_len,device):
        super(PositionalEncoding,self).__init__()

        self.encoding=torch.zeros((max_len,d_model), device=device)
        self.encoding.requires_grad=False

        pos=torch.arange(0,max_len,device=device)
        pos=pos.float().unsqueeze(dim=1)
        
        _2i=torch.arange(0,d_model,step=2,device=device).float()
        self.encoding[:,0::2]=torch.sin(pos/(10000**(_2i/d_model)))
        self.encoding[:,1::2]=torch.cos(pos/(10000**(_2i/d_model)))
        
    def forward(self,x):
        batch_size,seq_len=x.size()
        return self.encoding[:seq_len,:]
    
class TokenEmbedding(nn.Embedding):
    
    def __init__(self, vocab_size,d_model):
        super().__init__(vocab_size,d_model,padding_idx=1)
        
class TransformerEmbedding(nn.Module):
    
    def __init__(self, vocab_size,d_model,max_len,drop_prob,device):
        super().__init__()
        self.tok_emb=TokenEmbedding(vocab_size,d_model)
        self.pos_emb=PositionalEncoding(d_model,max_len,device)
        self.drop_out=nn.Dropout(p=drop_prob)
        
    def  forward(self,x):
        tok_emb=self.tok_emb(x)
        pos_emb=self.pos_emb(x)
        return self.drop_out(tok_emb+pos_emb)
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model,ffn_hidden,n_head,drop_prob):
        super().__init__()
        self.attention=MultiHeadAttention(d_model=d_model,n_head=n_head)
        self.norm1=LayerNorm(d_model=d_model)
        self.dropout1=nn.Dropout(p=drop_prob)
        
        self.ffn=PositionwiseFeedForward(d_model=d_model,hidden=ffn_hidden,drop_prob=drop_prob)
        self.norm2=LayerNorm(d_model=d_model)
        self.dropout2=nn.Dropout(p=drop_prob)
        
        
    def forward(self,x,src_mask):
        _x=x
        x=self.attention(q=x,k=x,v=x,mask=src_mask)
        x=self.dropout1(x)
        x=self.norm1(x+_x)
        
        _x=x
        x=self.ffn(x)
        x=self.dropout2(x)
        x=self.norm2(x+_x)
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self,d_model,ffn_hidden,n_head,drop_prob):
        super().__init__()
        self.attention=MultiHeadAttention(d_model=d_model,n_head=n_head)
        self.dropout1=nn.Dropout(p=drop_prob)
        self.norm1=LayerNorm(d_model=d_model)
        
        self.dec_enc_attention=MultiHeadAttention(d_model=d_model,n_head=n_head)
        self.dropout2=nn.Dropout(p=drop_prob)
        self.norm2=LayerNorm(d_model=d_model)
        
        self.ffn=PositionwiseFeedForward(d_model=d_model,hidden=ffn_hidden,drop_prob=drop_prob)
        self.dropout3=nn.Dropout(p=drop_prob)
        self.norm3=LayerNorm(d_model=d_model)
    
    def forward(self,dec,enc,trg_mask,src_mask):
        _x=dec
        x=self.attention(q=dec,k=dec,v=dec,mask=trg_mask)
        x=self.dropout1(x)
        x=self.norm1(x+_x)
        
        if enc is not None:
            _x=x
            x=self.dec_enc_attention(q=x,k=enc,v=enc,mask=src_mask)
            x=self.dropout2(x)
            x=self.norm2(x+_x)
            
        _x=x
        x=self.ffn(x)
        x=self.dropout3(x)
        x=self.norm3(x+_x)
        return x
    
class Encoder(nn.Module):
    def __init__(self, enc_voc_size,max_len,d_model,ffn_hidden,n_head,n_layers,drop_prob,device):
        super().__init__()
        self.emb=TransformerEmbedding(vocab_size=enc_voc_size,d_model=d_model,max_len=max_len,drop_prob=drop_prob,device=device)
        self.layers=nn.ModuleList([EncoderLayer(d_model=d_model,ffn_hidden=ffn_hidden,n_head=n_head,drop_prob=drop_prob) for _ in range(0, n_layers)])
    
    def forward(self,x,src_mask):
        x=self.emb(x)
        for layer in self.layers:
            x=layer(x,src_mask)
            return x
        
class Decoder(nn.Module):
    
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model, drop_prob=drop_prob,max_len=max_len, vocab_size=dec_voc_size, device=device)

        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model, ffn_hidden=ffn_hidden, n_head=n_head,drop_prob=drop_prob) for _ in range(n_layers)])
        
        self.linear=nn.Linear(d_model,dec_voc_size)
        
    def forward(self,trg,enc_src,trg_mask,src_mask):
        trg=self.emb(trg)
        for layer in self.layers:
            trg=layer(trg,enc_src,trg_mask,src_mask)
            
        output=self.linear(trg)
        return output
    
class Transformer(nn.Module):

    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, device):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               enc_voc_size=enc_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

        self.decoder = Decoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               dec_voc_size=dec_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

tokenizer_en = get_tokenizer('spacy', language='en_core_web_sm')
tokenizer_de = get_tokenizer('spacy', language='de_core_news_sm')
train_iter = Multi30k(root=".data", split='train', language_pair=('de', 'en'))

def yield_tokens(data_iter, tokenizer, language):
    for data_sample in data_iter:
        yield tokenizer(data_sample[language])
        
   
# 构建词汇表
vocab_de = build_vocab_from_iterator(
    yield_tokens(train_iter, tokenizer_de, language=0),
    specials=['<unk>', '<pad>', '<bos>', '<eos>'],
    min_freq=2
)
vocab_en = build_vocab_from_iterator(
    yield_tokens(train_iter, tokenizer_en, language=1),
    specials=['<unk>', '<pad>', '<bos>', '<eos>'],
    min_freq=2
)

# 设置默认未知词标记
vocab_de.set_default_index(vocab_de['<unk>'])
vocab_en.set_default_index(vocab_en['<unk>'])

src_pad_idx = 1
trg_pad_idx = 1
trg_sos_idx = 2
enc_voc_size = len(vocab_de)
dec_voc_size = len(vocab_en)
model = Transformer(src_pad_idx=src_pad_idx,trg_pad_idx=trg_pad_idx,trg_sos_idx=trg_sos_idx,
                    d_model=d_model,enc_voc_size=enc_voc_size, 
                    dec_voc_size=dec_voc_size,max_len=max_len,ffn_hidden=ffn_hidden,n_head=n_heads,
                    n_layers=n_layers, drop_prob=drop_prob,device=device).to(device)

optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)

def collate_batch(batch):
    
    de_batch, en_batch = [], []
    for de, en in batch:
        # 德语端添加 <bos> 和 <eos>
        de_processed = [vocab_de['<bos>']] + vocab_de(tokenizer_de(de)) + [vocab_de['<eos>']]
        # 英语端同理
        en_processed = [vocab_en['<bos>']] + vocab_en(tokenizer_en(en)) + [vocab_en['<eos>']]
        
        de_batch.append(torch.tensor(de_processed, dtype=torch.long))
        en_batch.append(torch.tensor(en_processed, dtype=torch.long))
    
    # 填充到相同长度
    de_padded = pad_sequence(de_batch, padding_value=vocab_de['<pad>'], batch_first=True)
    en_padded = pad_sequence(en_batch, padding_value=vocab_en['<pad>'], batch_first=True)

    return de_padded, en_padded

BATCH_SIZE = 128

# 重新加载数据集（因为迭代器只能遍历一次）
train_iter = Multi30k(split='train', language_pair=('de', 'en'))
valid_iter = Multi30k(split='valid', language_pair=('de', 'en'))


train_list = list(train_iter)


train_loader = DataLoader(
    list(train_iter),  # 转换为列表（Multi30k 是迭代器）
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_batch
)

valid_loader = DataLoader(
    list(valid_iter),
    batch_size=BATCH_SIZE,
    collate_fn=collate_batch
)

de_path='.data/datasets/Multi30k/test2016.de'
en_path='.data/datasets/Multi30k/test2016.en'
with open(de_path,'r', encoding="utf-8") as f: # 用utf-8 ecoding加载，大概了解下为什么
    test_de=f.readlines()
with open(en_path,'r', encoding="utf-8") as f:
    test_en=f.readlines()
    
test_list=[(de,en) for de,en in zip(test_de,test_en)]

test_loader = DataLoader(
    test_list, 
    batch_size=1,  # 也可以设为 1 逐条测试
    shuffle=False,
    collate_fn=collate_batch
)


optimizer = Adam(params=model.parameters(),
                 lr=init_lr,
                 weight_decay=weight_decay,
                 eps=adam_eps)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 verbose=True,
                                                 factor=factor,
                                                 patience=patience)

def train(model, train_loader, optimizer, criterion, clip):
    losssum=0
    model.train()
    
    for de,en in train_loader:  
        optimizer.zero_grad()
    
        #print(f"德语张量形状: {de.shape}")  # (seq_len, batch_size)
        #print(f"英语张量形状: {en.shape}")
        trg=en.to(device)  
        src=de.to(device)

        output = model(src, trg[:, :-1])
    
        loss = criterion(
                output.reshape(-1, output.shape[2]),
                trg[:, 1:].reshape(-1)
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        #print('loss :', loss.item())
    
        losssum = losssum+loss.item()
    last_loss=losssum/len(train_loader) 
    return last_loss


def bleu_stats(hypothesis, reference):
    stats = []
    stats.append(len(hypothesis))
    stats.append(len(reference))
    
    # 计算1-gram到4-gram的统计量
    for n in range(1, 5):
        s_ngrams = Counter(
            [tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) + 1 - n)]
        )
        r_ngrams = Counter(
            [tuple(reference[i:i + n]) for i in range(len(reference) + 1 - n)]
        )

        # 添加匹配的n-gram计数（交集）
        stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
        # 添加候选翻译中n-gram的总数
        stats.append(max([len(hypothesis) + 1 - n, 0]))
    return stats


def bleu(stats):
    if len(list(filter(lambda x: x == 0, stats))) > 0:
        return 0
    
    c, r = stats[:2]  # 获取候选和参考翻译长度
    
    # 计算1-4 gram的加权对数平均精确度
    log_bleu_prec = sum(
        [math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]
    ) / 4.
    
    # 计算长度惩罚因子并组合最终BLEU分数
    return math.exp(min([0, 1 - float(r) / c]) + log_bleu_prec)


def get_bleu(hypotheses, reference):
    stats = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    
    # 累加所有句对的统计量
    for hyp, ref in zip(hypotheses, reference):
        stats += np.array(bleu_stats(hyp, ref))
    
    # 计算并返回百分比形式的BLEU分数
    return 100 * bleu(stats)


def idx_to_word(x, vocab):
    words = []
    for i in x:
        word = vocab.get_itos()[i]  # 通过词汇表转换索引到单词
        if '<' not in word:  # 过滤掉包含'<'的特殊标记
            words.append(word)
    return " ".join(words)  # 拼接成字符串

def evaluate(model,loader,criterion):
    model.eval()
    losssum = 0
    batch_bleu = []
    with torch.no_grad(): 
        for de, en in loader: # 这个怎么还叫trainloader
            trg=en.to(device)  
            src=de.to(device)
            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output_reshape, trg)
            losssum += loss.item()

            total_bleu = []
            for j in range(batch_size):
                try:
                    trg_words = idx_to_word(trg[j], vocab_en)
                    output_words = output[j].max(dim=1)[1]
                    output_words = idx_to_word(output_words, vocab_en)
                    # bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())
                    bleu = 0.0 # 这里我注释掉了
                    total_bleu.append(bleu)
                except:
                    pass
                
            total_bleu = sum(total_bleu) 
            batch_bleu.append(total_bleu)

    batch_bleu = sum(batch_bleu)
    valid_loss=losssum/len(loader)
    
    return  valid_loss,batch_bleu

def run(total_epoch, best_loss):
    train_losses, test_losses, bleus = [], [], []
    for step in range(total_epoch):
        start_time = time.time()
        train_loss = train(model, train_loader, optimizer, criterion, clip)
        valid_loss, bleu = evaluate(model, valid_loader, criterion)
        end_time = time.time()

        if step > warmup:
            scheduler.step(valid_loss)

        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        #bleus.append(bleu)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        torch.save(model.state_dict(), 'final_model.pth')
        
#        if valid_loss < best_loss:
#            best_loss = valid_loss
#            torch.save(model.state_dict(), 'saved/model-{0}.pt'.format(valid_loss))

        plt.figure()
        plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
        plt.plot(range(1, len(test_losses)+1), test_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('result/loss_curve.png')  # 保存图像
        plt.close()  # 关闭图形，避免内存泄漏

        f = open('result/train_loss.txt', 'w')
        f.write(str(train_losses))
        f.close()

        #f = open('result/bleu.txt', 'w')
        #f.write(str(bleus))
        #f.close()

        f = open('result/test_loss.txt', 'w')
        f.write(str(test_losses))
        f.close()

        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')
#        print(f'\tBLEU Score: {bleu:.3f}')

def load_model(model_path):
    model = Transformer(src_pad_idx=src_pad_idx,trg_pad_idx=trg_pad_idx,trg_sos_idx=trg_sos_idx,
                    d_model=d_model,enc_voc_size=enc_voc_size, 
                    dec_voc_size=dec_voc_size,max_len=max_len,ffn_hidden=ffn_hidden,n_head=n_heads,
                    n_layers=n_layers, drop_prob=drop_prob,device=device).to(device)  
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


if __name__ == '__main__':
    run(total_epoch=epoch, best_loss=inf)
    model = load_model('final_model.pth')

    # 测试或推理
    # test_loss,bleu = evaluate(model, test_loader,criterion)
    # print(f"Test Loss: {test_loss:.4f}")
    src_sentence="Ein Mann mit einem orangefarbenen Hut, der etwas anstarrt."
    tgt_sentence=transformer_predict(model,src_sentence,vocab_de,vocab_en,device)
    print(tgt_sentence)