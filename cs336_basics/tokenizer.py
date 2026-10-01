import regex as re
from collections import Counter,defaultdict
import os
from pretokenization import find_chunk_boundaries , file_open

# 初始化词表
def initialize_vocab(special_tokens: list[str] ) -> dict[int,bytes]:
    vocab = {i: bytes([i]) for i in range(256)}
    vocab_len = 256
    # 添加special tokens
    for special_token in special_tokens :
        vocab[vocab_len] = special_token.encode("UTF-8")
        vocab_len = vocab_len + 1 
    return vocab

# 切分语料，返回预分词计数
def cut_text(special_tokens: list[str], text : str) -> dict[bytes,int] :
    # 在specialal tokens处切分文本
    if special_tokens != [] :

        escaped = [re.escape(tok) for tok in sorted(special_tokens, key=len, reverse=True)]
        pattern = "(?:" + "|".join(escaped) + ")"
        cuttedText = re.split(pattern,text)
        cuttedText = [p for p in cuttedText if p != ""]
    # 统计预分词数
    else :
        cuttedText = []
        cuttedText.append(text)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    # 拿到切分后文本的每个token
    tokens = [
    m.group()
    for text in cuttedText
    for m in re.finditer(PAT, text)
    ]
    # 编码为utf-8
    tokens_encoded = []
    for token in tokens :
        token_encoded = token.encode("UTF-8")
        tokens_encoded.append(token_encoded)
    counts = Counter(tokens_encoded)
    pretoken_cnt = dict(counts)

    return pretoken_cnt

def bpe_merge(pretokens: dict , vocab_size: int) -> tuple[dict,list]  :
    current_vocabsize = 257
    # 将bytes变为元组
    new_pretokens = defaultdict(int)
    for k, v in pretokens.items():
        parts = tuple(bytes([b]) for b in k)
        new_pretokens[parts] += v

    # 合并函数    
    def merge_pair(parts, pair, merged):
        result = []
        i = 0
        while i < len(parts):
            if i < len(parts) - 1 and parts[i] == pair[0] and parts[i+1] == pair[1]:
                result.append(merged)
                i += 2
            else:
                result.append(parts[i])
                i += 1
        return tuple(result)
    
    merged_list = []
    while current_vocabsize < vocab_size :
        # 统计频率
        pair_freq = Counter()
        for parts, v in new_pretokens.items() :
            for pair in zip(parts, parts[1:]) :
              pair_freq[pair] += v

        if not pair_freq :
            return dict(new_pretokens), merged_list
        # 取最大字典序
        max_pair, max_freq = max(
        pair_freq.items(),
        key=lambda kv: (kv[1], kv[0])
        )
        merged = max_pair[0] + max_pair[1]
        merged_list.append(tuple([max_pair[0], max_pair[1]]))

        updated_d = defaultdict(int)
        for parts, v in new_pretokens.items():
            new_parts = merge_pair(parts, max_pair, merged)
            updated_d[new_parts] += v

        new_pretokens = updated_d
        current_vocabsize += 1
    return dict(new_pretokens) , merged_list

def train_bpe(input_path: str | os.PathLike,
              vocab_size: int,
              special_tokens: list[str],
              **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]] :

    init_vocab = initialize_vocab(special_tokens)




def main():
    text = "low low low low low.lower lower lower lower lower newest newest newest"
    special_tokens =["<|endoftext|>"]
    cutted_text = cut_text(special_tokens, text)
    print(cutted_text)
    result, merge_list = bpe_merge(cutted_text, vocab_size=359)
    print(result)
    print("###########")
    print(merge_list)



if __name__ == "__main__":
    main()