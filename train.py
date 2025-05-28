import time
import math

import torch
from torch import nn, optim
from torch.optim import Adam

from torch.nn.utils.rnn import pad_sequence


import matplotlib.pyplot as plt 

from models.model.transformer import Transformer
from util.bleu import idx_to_word, get_bleu
from util.epoch_timer import epoch_time
from util.tokenizer import load_tokenizers
from util.data import load_dataloaders

from config import get_config

def train(model, train_loader, optimizer, criterion, clip, device):
    losssum=0
    model.train()
    
    for de, en in train_loader:  
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


def evaluate(model, loader, criterion, batch_size, device):
    model.eval()
    losssum = 0
    batch_bleu = []
    with torch.no_grad(): 
        for en, de in loader:
            trg=de.to(device)  
            src=en.to(device)
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


def run(model, total_epoch, train_loader, batch_size,device,valid_loader, warmup, clip, src_pad_idx, init_lr, weight_decay, adam_eps, factor, patience):


    # create 
    criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)
    optimizer = Adam(params=model.parameters(),
                 lr=init_lr,
                 weight_decay=weight_decay,
                 eps=adam_eps)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                    verbose=True,
                                                    factor=factor,
                                                    patience=patience)    

    train_losses, test_losses, bleus = [], [], []
    for step in range(total_epoch):
        start_time = time.time()
        train_loss = train(model, train_loader, optimizer, criterion, clip, device)
        valid_loss, bleu = evaluate(model, valid_loader, criterion, batch_size, device)
        end_time = time.time()

        if step > warmup:
            scheduler.step(valid_loss)

        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        #bleus.append(bleu)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        if step % 50 == 0:
            torch.save(model.state_dict(), f'state_model_{step}.pth')
    
        
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

def load_model(model_path, src_pad_idx, trg_pad_idx, trg_sos_idx, d_model, enc_voc_size, dec_voc_size, max_len, ffn_hidden, n_heads, n_layers, drop_prob, device):
    model = Transformer(src_pad_idx=src_pad_idx,trg_pad_idx=trg_pad_idx,trg_sos_idx=trg_sos_idx,
                    d_model=d_model,enc_voc_size=enc_voc_size, 
                    dec_voc_size=dec_voc_size,max_len=max_len,ffn_hidden=ffn_hidden,n_head=n_heads,
                    n_layers=n_layers, drop_prob=drop_prob,device=device).to(device)  
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


if __name__ == '__main__':

    cfg=get_config()

    # # data related configs
    # batch_size = 128
    de_path='.data/datasets/Multi30k/test2016.de'
    en_path='.data/datasets/Multi30k/test2016.en'
    
    # # model related configs
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # src_pad_idx = 1
    # trg_pad_idx = 1
    # trg_sos_idx = 2
    # max_len = 256
    # d_model = 512
    # n_layers = 6
    # n_heads = 8
    # ffn_hidden = 2048
    # drop_prob = 0.1

    # # training realted configs
    # init_lr = 1e-5
    # factor = 0.9
    # adam_eps = 5e-9
    # patience = 10
    # warmup = 100
    # total_epoch = 500
    # clip = 1.0
    # weight_decay = 5e-4
    # inf = float('inf')

    # load tokenizer
    vocab_en, vocab_de, tokenizer_en, tokenizer_de = load_tokenizers()

    # load dataloaders
    train_loader, valid_loader, test_loader = load_dataloaders(de_path, en_path, cfg.batch_size, vocab_en, vocab_de, tokenizer_en, tokenizer_de)
    
    # create model
    model = Transformer(src_pad_idx=cfg.src_pad_idx,trg_pad_idx=cfg.trg_pad_idx,trg_sos_idx=cfg.trg_sos_idx,
                        d_model=cfg.d_model,enc_voc_size=len(vocab_en), 
                        dec_voc_size=len(vocab_de),max_len=cfg.max_len,ffn_hidden=cfg.ffn_hidden,n_head=cfg.n_heads,
                        n_layers=cfg.n_layers, drop_prob=cfg.drop_prob,device=cfg.device).to(cfg.device)



    run(model=model, total_epoch=cfg.total_epoch, train_loader=train_loader,batch_size=cfg.batch_size,device=cfg.device, valid_loader=valid_loader, warmup=cfg.warmup, clip=cfg.clip, src_pad_idx=cfg.src_pad_idx,
        init_lr=cfg.init_lr, weight_decay=cfg.weight_decay, adam_eps=cfg.adam_eps, factor=cfg.factor, patience=pad_sequence)
    
    torch.save(model.state_dict(), 'final_model.pth')
    # model = load_model('final_model.pth')

    # 测试或推理
    # test_loss,bleu = evaluate(model, test_loader,criterion)
    # print(f"Test Loss: {test_loss:.4f}")
    # src_sentence="Ein Mann mit einem orangefarbenen Hut, der etwas anstarrt."
    # tgt_sentence=transformer_predict(model,src_sentence,vocab_de,vocab_en,device)
    # print(tgt_sentence)
    