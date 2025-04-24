import spacy
from torchtext.datasets import Multi30k 
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator



def yield_tokens(data_iter, tokenizer, language):
    for data_sample in data_iter:
        yield tokenizer(data_sample[language])

def load_tokenizers():

    train_iter = Multi30k(root=".data", split='train', language_pair=('de', 'en'))

    # load tokenizer
    tokenizer_en = get_tokenizer('spacy', language='en_core_web_sm')
    tokenizer_de = get_tokenizer('spacy', language='de_core_news_sm')

    # build vocabulary
    vocab_en = build_vocab_from_iterator(
        yield_tokens(train_iter, tokenizer_en, language=1),
        specials=['<unk>', '<pad>', '<bos>', '<eos>'],
        min_freq=2
    )
    vocab_de = build_vocab_from_iterator(
        yield_tokens(train_iter, tokenizer_de, language=0),
        specials=['<unk>', '<pad>', '<bos>', '<eos>'],
        min_freq=2
    )

    # 设置默认未知词标记
    vocab_en.set_default_index(vocab_en['<unk>'])
    vocab_de.set_default_index(vocab_de['<unk>'])
    
    return vocab_en, vocab_de, tokenizer_en, tokenizer_de