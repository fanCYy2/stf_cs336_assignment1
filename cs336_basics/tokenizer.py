import regex as re
from collections import Counter,defaultdict
import os
from cs336_basics.pretokenization import pretokenizer

# 初始化词表
def initialize_vocab(special_tokens: list[str] ) -> dict[int,bytes]:
    vocab = {i: bytes([i]) for i in range(256)}
    vocab_len = 256
    # 添加special tokens
    for special_token in special_tokens :
        vocab[vocab_len] = special_token.encode("UTF-8")
        vocab_len = vocab_len + 1 
    return vocab

# 合并
def bpe_merge(pretokens: dict ,current_vocabsize : int, vocab_size: int) -> tuple[dict,list]  :
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
    merged_dict = {}
    while current_vocabsize < vocab_size :
        # 统计频率
        pair_freq = Counter()
        for parts, v in new_pretokens.items() :
            for pair in zip(parts, parts[1:]) :
                pair_freq[pair] += v

        if not pair_freq :
            return merged_dict, merged_list
        # 取最大字典序
        max_pair, max_freq = max(
        pair_freq.items(),
        key=lambda kv: (kv[1], kv[0])
        )
        merged = max_pair[0] + max_pair[1]
        merged_list.append(tuple([max_pair[0], max_pair[1]]))

        merged_dict[current_vocabsize] = merged
        updated_d = defaultdict(int)
        for parts, v in new_pretokens.items():
            new_parts = merge_pair(parts, max_pair, merged)
            updated_d[new_parts] += v

        new_pretokens = updated_d
        current_vocabsize += 1
   

    return merged_dict, merged_list

# 入口函数
def train_bpe(input_path: str | os.PathLike,
              vocab_size: int,
              special_tokens: list[str],
              **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]] :

    init_vocab = initialize_vocab(special_tokens)
    numProcesses = os.cpu_count() - 1
    pretokens = pretokenizer(input_path,numProcesses,special_tokens)
    merged_dict, merged_list = bpe_merge(pretokens,len(init_vocab),vocab_size)
    vocab = init_vocab | merged_dict

    return vocab, merged_list


def main():
    vocab, merged_list = train_bpe("tests/fixtures/tinystories_sample_5M.txt", 500, ["<|endoftext|>"])
    print(vocab)
    print("----------------------")
    print(merged_list)
    
if __name__ == "__main__":
    main()