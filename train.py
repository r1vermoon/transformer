import math
import time

from torch import nn, optim
from torch.optim import Adam

from data import *
from models.model.transformer import Transformer
from util.bleu import idx_to_word, get_bleu
from util.epoch_timer import epoch_time

import matplotlib.pyplot as plt 


model = Transformer(src_pad_idx=src_pad_idx,trg_pad_idx=trg_pad_idx,trg_sos_idx=trg_sos_idx,
                    d_model=d_model,enc_voc_size=enc_voc_size, 
                    dec_voc_size=dec_voc_size,max_len=max_len,ffn_hidden=ffn_hidden,n_head=n_heads,
                    n_layers=n_layers, drop_prob=drop_prob,device=device).to(device)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)

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
    test_loss,bleu = evaluate(model, test_loader,criterion)
    print(f"Test Loss: {test_loss:.4f}")