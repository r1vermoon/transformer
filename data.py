from conf import *
from torchtext.datasets import Multi30k 
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


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