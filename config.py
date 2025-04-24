import argparse
import torch

def get_config():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--batch_size", type=int, default=128)
    # parser.add_argument("--de_path", type=str, default='./data/datasets/Multi30k/test2016.de')
    # parser.add_argument("--en_path", type=str, default='./data/datasets/Multi30k/test2016.en')

    # model
    parser.add_argument("--device", type=str, default="cuda:1" )
    parser.add_argument("--src_pad_idx", type=int, default=1)
    parser.add_argument("--trg_pad_idx", type=int, default=1)
    parser.add_argument("--trg_sos_idx", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--ffn_hidden", type=int, default=2048)
    parser.add_argument("--drop_prob", type=float, default=0.1)

    # train
    parser.add_argument("--init_lr", type=float, default=1e-5)
    parser.add_argument("--factor", type=float, default=0.9)
    parser.add_argument("--adam_eps", type=float, default=5e-9)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--total_epoch", type=int, default=500)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--inf", type=float, default=float('inf'))

    args = parser.parse_args()
    
    # 参数合法性检查
    assert args.d_model % args.n_heads == 0
    assert args.drop_prob >= 0 and args.drop_prob < 1
    
    return args