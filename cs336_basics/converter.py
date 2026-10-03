from __future__ import annotations
from functools import lru_cache

@lru_cache
def bytes2uni() -> dict[int,str] :
    """
    单bytes的值至bytes所对应的unicode字符映射字典,其中不可见字符用了一些别的可见字符来代替
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d

@lru_cache
def uni2bytes() -> dict[str,int] :
    """
    上面的反向映射,也就是 bytes所对应的uni字符 至 单bytes的值 映射
    """
    d = bytes2uni()
    reversed_d = dict(zip(d.values(), d.keys()))
    return reversed_d

def bytes2str(bytes_ : bytes) -> str :
    """
    将一个bytes对象映射成字符串
    """
    str_list = []
    bytes2uni_ = bytes2uni()
    for d in bytes_ :
        str_list.append(bytes2uni_[d])
    s = ''.join(str_list)
    return s

def str2bytes(words : str) -> bytes :
    """
    与上面反向,将字符串映射为bytes
    """
    uni2bytes_ = uni2bytes()
    str2int_list = []
    for i in words :
        str2int_list.append(uni2bytes_[i])
    return bytes(str2int_list)

if __name__ == "__main__":
    # bytes_ = b" the"
    # print(bytes2str(bytes_))
    words = "Ġthe"
    print(str2bytes(words))
    # print(uni2bytes())