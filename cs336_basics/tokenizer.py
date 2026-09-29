import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# re.finditer(PAT, "hello world")
tokens = [m.group() for m in re.finditer(PAT, "hello world, 你好")]
tokens_encoded = []
for token in tokens :
    token_encoded = token.encode("UTF-8")
    tokens_encoded.append(token_encoded)

print(tokens)
print(tokens_encoded)

