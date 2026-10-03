import regex as re
from collections import Counter,defaultdict
import os
import json

from cs336_basics.pretokenization import pretokenizer
from cs336_basics.converter import bytes2str, str2bytes

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
def bpe_merge(pretokens: dict, current_vocabsize: int, vocab_size: int) -> tuple[dict, list]:
    # 将输入 pretokens 转为内部稳定编号结构
    # new_pretokens 现在按 id 存放当前 parts，id 就是下标
    # 初始时先合并相同 parts 的频次，保持与原逻辑一致
    temp_pretokens = defaultdict(int)
    for k, v in pretokens.items():
        parts = tuple(bytes([b]) for b in k)
        temp_pretokens[parts] += v

    new_pretokens = []   # list[tuple[bytes, ...]]，id = 下标
    freqs = []           # list[int]，与 new_pretokens 一一对应，表示该 pretoken 的频次

    for parts, v in temp_pretokens.items():
        new_pretokens.append(parts)
        freqs.append(v)

    # pair_to_ids: pair -> 包含该 pair 的 pretoken id 集合
    # pair_freq: pair -> 加权出现次数（同一 pretoken 内出现多次要重复计数）
    pair_to_ids = defaultdict(set)
    pair_freq = Counter()

    for pid, parts in enumerate(new_pretokens):
        v = freqs[pid]
        for pair in zip(parts, parts[1:]):
            pair_to_ids[pair].add(pid)
            pair_freq[pair] += v

    # 合并函数：与原逻辑完全一致
    def merge_pair(parts, pair, merged):
        result = []
        i = 0
        while i < len(parts):
            if i < len(parts) - 1 and parts[i] == pair[0] and parts[i + 1] == pair[1]:
                result.append(merged)
                i += 2
            else:
                result.append(parts[i])
                i += 1
        return tuple(result)

    merged_list = []
    merged_dict = {}

    while current_vocabsize < vocab_size:
        # 没有可合并的 pair 就结束
        if not pair_freq:
            return merged_dict, merged_list

        # 取最大字典序：先按频率，再按 pair 本身
        max_pair, max_freq = max(
            pair_freq.items(),
            key=lambda kv: (kv[1], kv[0])
        )
        merged = max_pair[0] + max_pair[1]

        merged_list.append(tuple([max_pair[0], max_pair[1]]))
        merged_dict[current_vocabsize] = merged

        # 只取出包含 max_pair 的 pretoken id 快照
        ids = list(pair_to_ids.get(max_pair, set()))

        for pid in ids:
            old_parts = new_pretokens[pid]
            new_parts = merge_pair(old_parts, max_pair, merged)

            # 理论上包含 max_pair 一定会发生变化；这里做安全判断
            if new_parts == old_parts:
                continue

            v = freqs[pid]

            # 1) 从索引中移除旧 parts 的所有 pair 贡献
            for pair in zip(old_parts, old_parts[1:]):
                pair_freq[pair] -= v
                if pair_freq[pair] == 0:
                    del pair_freq[pair]

                s = pair_to_ids.get(pair)
                if s is not None:
                    s.discard(pid)
                    if not s:
                        del pair_to_ids[pair]

            # 2) 更新该 id 的 parts，编号保持不变
            new_pretokens[pid] = new_parts

            # 3) 把新 parts 的 pair 贡献加入索引
            for pair in zip(new_parts, new_parts[1:]):
                pair_freq[pair] += v
                pair_to_ids[pair].add(pid)

        current_vocabsize += 1

    return merged_dict, merged_list

def save_data(file_path1 : str, file_path2: str, vocab : dict[int, bytes], merges: list[tuple[bytes,bytes]]) -> None :
    """
    存储数据,形式与参考实例一样.
    """
    reversed_vocab = {}
    for idx, bytes_ in vocab.items():
        reversed_vocab[bytes2str(bytes_)] = idx

    # store the vocabulary
    with open(file_path1, "w", encoding= "utf-8") as f :
        json.dump(reversed_vocab, f, ensure_ascii=False, indent=2)

    # store the merge list
    with open(file_path2, "w", encoding= "utf-8") as f :
        for i in merges :
            words = " ".join([bytes2str(i[0]), bytes2str(i[1])])
            f.write(words)
            f.write("\n")

def read_data(file_path1: str, file_path2: str) -> tuple[dict[int, bytes],list[tuple[bytes, bytes]]]:
    """
    事实上是save data的逆, 从文件中读出vocab 和 merge list
    """
    # 读字典
    with open(file_path1, "r", encoding= "utf-8") as f :
        vocab = json.load(f)

    vocab_ = dict(zip(vocab.values(),vocab.keys()))
    for idx in vocab_ :
        vocab_[idx] = str2bytes(vocab_[idx])
    # 读merge list
    merged_list = []
    with open(file_path2, "r", encoding= "utf-8") as f :
        for line in f :
            line = line.rstrip('\r\n')
            result = line.split(" ")
            lst = [str2bytes(result[0]), str2bytes(result[1])]
            merged_list.append(tuple(lst))

    return vocab_, merged_list
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
    vocab, merged_list = train_bpe("data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"])
    save_data("cs336_basics/vocab.json","cs336_basics/merges.txt", vocab, merged_list)
    
if __name__ == "__main__":
    main()