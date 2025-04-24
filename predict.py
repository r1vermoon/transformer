import torch
from models.model.transformer import Transformer
from util.tokenizer import load_tokenizers

# device=
# src_sentence="Two young, White males are outside near many bushes."

def make_predict(src_sentence,vocab_en,vocab_de,tokenizer_en,model,device):
    # vocab_en,vocab_de,tokenizer_en,tokenizer_de=load_tokenizers()
    src_token=tokenizer_en(src_sentence)
    src_indices=[vocab_en[token] for token in src_token]
    print(src_token)
    print(src_indices)
    src_input=torch.tensor(src_indices).unsqueeze(0).to(device)
    trg_indices=[vocab_de['<bos>']]
    # print(trg_indices)
    
    for i in range(500):
        trg_input=torch.tensor(trg_indices).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output=model(src_input,trg_input)
            predict_output=output.argmax(2)[:,-1].item()
            trg_indices.append(predict_output)
            # print(trg_indices)
            if predict_output==vocab_de['<eos>']:
                break
        
    trg_sentence=' '.join([vocab_de.lookup_token(number) for number in trg_indices[1:-1]])
    return trg_sentence

vocab_en,vocab_de,tokenizer_en,tokenizer_de=load_tokenizers()
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
src_pad_idx = 1
trg_pad_idx = 1
trg_sos_idx = 2
max_len = 256
d_model = 512
n_layers = 6
n_heads = 8
ffn_hidden = 2048
drop_prob = 0.1

src_sentence= "Several men in hard hats are operating a giant pulley system."

model = Transformer(src_pad_idx=src_pad_idx,trg_pad_idx=trg_pad_idx,trg_sos_idx=trg_sos_idx,
                    d_model=d_model,enc_voc_size=len(vocab_en), 
                    dec_voc_size=len(vocab_de),max_len=max_len,ffn_hidden=ffn_hidden,n_head=n_heads,
                    n_layers=n_layers, drop_prob=drop_prob,device=device).to(device)
model.load_state_dict(torch.load("final_model_100.pth"))
model.eval()
trg_sentence=make_predict(src_sentence=src_sentence,vocab_en=vocab_en,vocab_de=vocab_de,tokenizer_en=tokenizer_en,model=model,device=device)
print(trg_sentence)