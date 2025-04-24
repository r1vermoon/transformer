import torch

from torch.nn.utils.rnn import pad_sequence
from torchtext.datasets import Multi30k
from torch.utils.data import DataLoader


def collate_batch(vocab_en, vocab_de, tokenizer_en, tokenizer_de, batch):

    en_batch, de_batch  = [], []
    for de, en in batch:
        en_processed = [vocab_en['<bos>']] + vocab_en(tokenizer_en(en)) + [vocab_en['<eos>']]
        de_processed = [vocab_de['<bos>']] + vocab_de(tokenizer_de(de)) + [vocab_de['<eos>']]
        
        en_batch.append(torch.tensor(en_processed, dtype=torch.long))
        de_batch.append(torch.tensor(de_processed, dtype=torch.long))

    # 填充到相同长度
    en_padded = pad_sequence(en_batch, padding_value=vocab_en['<pad>'], batch_first=True)
    de_padded = pad_sequence(de_batch, padding_value=vocab_de['<pad>'], batch_first=True)

    return en_padded, de_padded

def load_dataloaders(de_path, en_path, batch_size, vocab_en, vocab_de, tokenizer_en, tokenizer_de):

    train_iter = Multi30k(split='train', language_pair=('de', 'en'))
    valid_iter = Multi30k(split='valid', language_pair=('de', 'en'))
    
    
    def wrapper_collate_batch(batch):
        return collate_batch(vocab_en, vocab_de, tokenizer_en, tokenizer_de, batch)

    train_loader = DataLoader(
        list(train_iter),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=wrapper_collate_batch
    )

    valid_loader = DataLoader(
        list(valid_iter),
        batch_size=batch_size,
        collate_fn=wrapper_collate_batch
    )

    # Multi30k has encoding errors, load this dataset manually
    with open(de_path,'r', encoding="utf-8") as f:
        test_de=f.readlines()
    with open(en_path,'r', encoding="utf-8") as f:
        test_en=f.readlines()
    test_list=[(de,en) for de,en in zip(test_de,test_en)]

    test_loader = DataLoader(
        test_list, 
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch
    )
    
    return train_loader, valid_loader, test_loader
