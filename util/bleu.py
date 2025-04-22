import math
from collections import Counter

import numpy as np

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
