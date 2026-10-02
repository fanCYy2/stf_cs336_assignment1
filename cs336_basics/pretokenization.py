import os
from typing import BinaryIO
import regex as re
from collections import Counter
import multiprocessing as mp

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

# 切分语料，返回预分词计数
def cut_text(special_tokens: list[str], text : str) -> dict[bytes,int] :
    # 在specialal tokens处切分文本
    if special_tokens != [] :

        escaped = [re.escape(tok) for tok in sorted(special_tokens, key=len, reverse=True)]
        pattern = "(?:" + "|".join(escaped) + ")"
        cuttedText = re.split(pattern,text)
        cuttedText = [p for p in cuttedText if p != ""]
    # 如果没有special token 原文本即为切分文本
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
    pretoken_cnt = dict(Counter(tokens_encoded))

    return pretoken_cnt

def worker(input_path : str, 
           start : int, 
           end : int, 
           special_tokens :list ) -> dict :
    
    with open(input_path,"rb") as f :
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        cuttedChunk = cut_text(special_tokens, chunk)
        return cuttedChunk
        
def pretokenizer(input_path :str, num_processes :int , special_tokens:list[str]) -> dict :
    # 打开文件，拿到切分边界列表
    boundaries_list = []
    with open(input_path, "rb") as f :
        split_special_tokens = ["<|endoftext|>"]
    
        boundaries = find_chunk_boundaries(f, num_processes ,split_special_tokens[0].encode("utf-8"))
        for start, end in zip(boundaries[:-1], boundaries[1:]) :
            boundaries_list.append((input_path,start,end,special_tokens))

    # 分发任务给子进程
    with mp.Pool(processes=num_processes) as pool :
        result = pool.starmap(worker,boundaries_list)
    # 统计计数
    pretoken_count = {}
    for d in result :
        for k ,v in d.items() :
            pretoken_count[k] = pretoken_count.get(k,0) + v
    
    return pretoken_count

if __name__ == "__main__" :
    # 打开文件，拿到切分边界列表
    pretokenizer("tests/fixtures/tinystories_sample_5M.txt",4,["<|endoftext|>"])