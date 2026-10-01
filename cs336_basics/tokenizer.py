import regex as re
from collections import Counter

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



# keys =[]
# for idx in tokens_encoded :
#     result = tuple(zip(idx, idx[1:]))

#     result_bytes = tuple(
#     tuple(bytes([x]) for x in pair)
#     for pair in result

# )
#     keys.append(result_bytes)

# print(keys)

def main():
    pretoken_cnt1 = cut_text(["<|endoftext|>"], "hello world")
    pretoken_cnt2 = cut_text([], "hello world")
    pretoken_cnt3 = cut_text(["<|endoftext|>"],"hello<|endoftext|>hello")
    pretoken_cnt4 = cut_text(["<|endoftext|>"],"<|endoftext|>hello<|endoftext|>")
    print(pretoken_cnt1)
    print(pretoken_cnt2)
    print(pretoken_cnt3)
    print(pretoken_cnt4)

if __name__ == "__main__":
    main()