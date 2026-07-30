import json
import os
from collections import defaultdict, Counter
from typing import Union

import regex as re
# import re

def train_bpe(
        input_path: Union[str, os.PathLike],
        vocab_size: int,
        special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    训练字节级 BPE (Byte-Pair Encoding) 分词器
    核心流程：
    1. 初始化词表为所有可能的字节（0-255）。
    2. 读取输入语料，并根据特殊 Token 进行切分，确保特殊 Token 不参与统计。
    3. 使用 GPT-2 的预分词正则健语料库切分成单词，并统计每个单词的频率。
    4. 迭代进行“合并”操作，直到达到目标此表大小：
        合并策略：总是选择当前出现频率较高，且再字典序上最大的字节对。
    5. 使用倒排索引优化合并过程中的频率更新，确保速度。
    6. 将合并产生的 Token 加入词表，并最终加入特殊 Token

    :param input_path:
    :param vocab_size:
    :param special_tokens:
    :return:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab: 训练好的词汇表，映射 Token ID -> Token 字节序列。
            merges: BPE 合并规则列表，按生成顺序排列。
    """

    # 1. 初始化基础词表
    # 词表从 0~255 的字节开始，是 BPE 的基础单位。
    vocab = {i: bytes([i]) for i in range(256)}

    # 计算需要进行的合并次数。
    # 目标词表大小 = 基础词表数(256) + 特殊 token 数 + 需要新生成的 Token 数。
    num_merges = vocab_size - 256 - len(special_tokens)

    # 2. 读取语料，并按特殊 Token 切分
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 如果有特殊 Token，在开始统计前，用正则表达式从预料中“隔离”出来
    # 这能防止 BPE 规则将特殊 Token（如 <|endoftext|>）拆开或与普通文本混合。

    """
    For special tokens:
    在训练时，需要保证特殊 Token 不参与频率统计。
    代码逻辑：
        切割语料，在开始统计词频钱，用正则表达式将语料在特殊 Token 处切开。
        独立统计：只对切分出来的普通文本片段进行 BPE 统计。
        最后加入：训练结束后，强行将特殊 Token 加入词表（通常放最后），确保他们也有 ID。
    """
    if special_tokens:
        # 正则中，”|“ 表示或，这行表示将多个特殊 Token 用 | 连接，形成一个匹配任一 token 的正则模式
        special_regex = "|".join(re.escape(t) for t in special_tokens)
        # 使用 split 分割，将 text 按照特殊 Token 分割为若干部分。
        parts = re.split(f"{special_regex}", text)
        # 过滤从 parts 中提取出的特殊 Token 本身，只保留用于 BPE 训练的普通文本片段。
        train_segments = [p for p in parts if p not in special_tokens]
        # e.g.
        # text = "Hello World World<|endoftext|>Hello happy happy<|endoftext|>!"
        # parts = "Hello World World", "/</|endoftext/|/>", "Hello happy happy", "/</|endoftext/|/>", "!"
        # train_segments =  ['Hello World World', 'Hello happy happy', '!']
    else:
        # 没有特殊 Token，直接使用整个语料。
        train_segments = [text]

    # 3. 预分词 (Pre-tokenization) 并统计词频
    # 使用 GPT-2 的 BPE 预分词正则表达式
    # GPT-2 正则表达式的作用是执行“预分词”。规则是：
    #   (1)不允许跨越类型合并：比如它会把字母和标点符号分开。
    #   (2)保护空格：通常会把单词前面的空格和单词连在一起，作为一个整体。
    #   例如："Hello world! ..."
    #   会被切分为 ["Hello", " world", "!", " ..."]
    gpt2_pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    # raw_counts: 存储每个"单词"，(预分词后的结果)及其出现频率
    # 单词被表示为字节元组，例如"hello" -> (b'h', b'e', b'l', b'l', b'o')
    raw_counts = Counter()
    for segment in train_segments:
        # 对每个预料片段应用预分词正则，找到所有"单词“
        words = gpt2_pat.findall(segment)
        for word in words:
            # transform word to UTF-8 bytes, than compose as tuple and be the Counter's key, statistic this tuple frequnecy
            """
            e.g.
            for "Hi",
                word.encode("utf-8") -> b'Hi'
                for b in b"Hi" will got 72 and 105
                bytes([b]) decode the int to byte, like b'H' and b'i'
                than we compose them as Tuple (b'H', b'i')
            Why we must get Tuple?
                cause Counter's Key must be unvariable. so List can not be key, but Tuple can.

            e.g.
                words = ["Hi", " there", "!", ...]  
                raw_counts = {
                    (b'H', b'i'): 50,
                    (b' ', b't', b'h', b'e', b'r', b'e'): 100,
                    (b'!'): 50,
                    (b'\xe4',b'\xbd',b'\xa0',b'\xe5',b'\xa5',b'\xbd'):20, # 你好
                }
            """
            raw_counts[tuple(bytes([b]) for b in word.encode("utf-8"))] += 1

    # ---- 构造搞笑数据结构支持快读合并 ----
    # word_list: 存储单个单词的字节列表。使用 list 而不是 tuple，因为 BPE 合并会修改单词内部结构。
    # counts_list: 存储对应单词的频率。
    words_list = []
    counts_list = []
    for word_tuple, freq in raw_counts.items():
        # 转化为 list 以便后面修改
        words_list.append(list(word_tuple))
        counts_list.append(freq)

    # stats: 存储所有可能的相邻字节对 (pair) 及其全局出现频率。
    # 结构 {(byte_a, byte_b): frequency}
    stats = defaultdict(int)

    # indices: 倒排索引 reverse index。存储 pair -> {包含该 pair 的单词在 word_list 中的下标集合}
    # 这个结构是性能优化的关键，用于快速找到需要更新的单词。
    indices = defaultdict(set)

    # initialize 'stats' and 'indices'
    # 遍历所有唯一的单词
    for idx, word in enumerate(words_list):
        freq = counts_list[idx] # get the word's freq
        # range all the adjacent pair in the word
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            stats[pair] += freq     # 累加该 pair 的全局频率
            indices[pair].add(idx)  # 将当前单词的索引加入该 pair 的倒排列表中

    merges = [] # 用于存储生成的 BPE 合并规则，按顺序记录

    # 4. 迭代合并流程
    # 循环执行 'num_merges' 次，每次找到并应用一个最佳合并规则
    for _ in range(num_merges):
        # if 'stats' is empty (maybe means all the pair has been merged or frequency is 0), break the loop
        if not stats:
            break

        # 4.1 find best pair
        # target: find the Pair with the highest frequency and the largest lexicographical order in current 'stats'
        # max(stats.items(), key=lambda x: (x[1], x[0]))
        #   x[1] is frequency. max() func will select the highest freq priority
        #   x[0] is the Pair (tuple of bytes). if freq is same, max() func will compare the Pair's lexi index
        best_pair = max(stats.items(), key=lambda x: (x[1], x[0]))[0]

        # if best Pair's frequency has been 0 (maybe means all the pair has been merged), than break the loop
        if stats[best_pair] <= 0:
            break

        # record merge
        merges.append(best_pair)
        # build new Token (merged Token list)
        new_token = best_pair[0] + best_pair[1]

        # 4.2 get word need to update
        # use reverted index 'indices', get all the index of 'best_pair'
        # must copy a 'relevant_indices' as backup, cause the loop after will modify the 'indices' and 'stats'
        relevant_indices = list(indices[best_pair])

        # 4.3 traversal and update the word be influenced, statistics info and reverse index.
        for idx in relevant_indices:
            word = words_list[idx]  # get word
            freq = counts_list[idx] # get frequency

            # scan the word, find all position that 'best pair' enmerged
            i = 0
            while i < len(word) - 1:
                # check 'i' and 'i+1', whether match 'best_pair'
                if word[i] == best_pair[0] and word[i + 1] == best_pair[1]:
                    # match the 'best pair', execute merge

                    # 4.3.1. update the lod neighbor Pair's freq
                    # left neighbor pair: (word[i-1], word[i])
                    if i > 0:
                        prev_pair = (word[i-1], word[i])
                        stats[prev_pair] -= freq
                        if stats[prev_pair] == 0:
                            # if freq in prev_pair has been 0, remove the prev_pair in 'stats'
                            """
                            否则 stats 字典里依然会存在这个键：{(b'x', b'y'): 0}。
                            当训练快结束，或者剩下的所有对频率都降为 0 时，max 函数依然会扫描这些值为 0 的项。
                            根据平局规则，如果存在多个频率为 0 的项，max 会返回其中字典序最大的那一个，这是错误的
                            """
                            del stats[prev_pair]

                    # right neighbor pair: (word[i+1], word[i+2])
                    if i < len(word) - 2:
                        next_pair = (word[i+1], word[i+2])
                        stats[next_pair] -= freq
                        if stats[next_pair] == 0:
                            del stats[next_pair]

                    # 4.3.2. modify the word structure, replace (word[i], word[i+1]) with new_token
                    word[i] = new_token
                    del word[i+1]

                    # 4.3.3. add new neighbor pair's freq
                    # new left neighbor pair: (word[i-1], new_token)
                    # 3. 添加新产生的邻居 Pair 的频率和索引
                    #    - 新的左邻居：(word[i-1], new_token)
                    if i > 0:
                        new_prev = (word[i-1], word[i])
                        stats[new_prev] += freq
                        indices[new_prev].add(idx)

                    # new right neighbor pair: (new_token, word[i+1])
                    #    - 新的右邻居：(new_token, word[i+1]) (注意：word[i+1] 是旧的 word[i+2])
                    if i < len(word) - 1:
                        new_next = (word[i], word[i+1])
                        stats[new_next] += freq
                        indices[new_next].add(idx)

                    # 合并后，索引 i 指向的是新 Token。
                    # i 不需要移动（i+=1），因为我们刚刚修改了 word[i] 并且删除了 word[i+1]。
                    # 下一轮循环会检查新的 (word[i], word[i+1])，即 (new_token, old_word[i+2])
                    # 这可以处理像 A A A -> X A 这样的情况，正确地更新新的邻居对
                else:
                    # not match, continue scan
                    i += 1

        # 4.4 clean: remove the 'best_pair' which has been merged completely
        # after clean, this Pair will not exist in the 'stats' and 'indices'
        if best_pair in stats: del stats[best_pair]
        if best_pair in indices: del indices[best_pair]

    # 5. build the final vocab
    # add new token by BPE merged, ID is started at 256, and increasing by merged order.
    for pair in merges:
        new_id = len(vocab)
        vocab[new_id] = pair[0] + pair[1]

    # add special token
    for s_tok in special_tokens:
        s_bytes = s_tok.encode("utf-8")
        vocab[(len(vocab))] = s_bytes

    return vocab, merges

def bytes_to_unicode():
    """
    build a map, map the bytes of 0~255 as a group of visible Unicode characters
    this is standard method from GPT-2
    :return:
    """
    # bs: 所有可见的字符（安全字符）
    # cs: 对应的 Unicode 字符列表，初始时与 bs 一一对应。后续将不可见字符映射到256...

    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)        # 把缺失的字节补全（此时 bs 覆盖 0~255）
            cs.append(256+n)    # 危险字节映射到 256, 257, 258...
            """
            bs: [33, ..., 255, 0,   1,   2,   ...]
            cs: [33, ..., 255, 256, 257, 258, ...]
            """
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def save_tokenizer_files(vocab, merges, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # initialize map table
    byte_encoder = bytes_to_unicode()

    # vocab save
    # use byte_encoder transform bytes into visible String
    json_vocab = {
        k: "".join(byte_encoder[b] for b in v)
        for k, v in vocab.items()
    }
    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(json_vocab, f, indent=4)

    # save merge rule
    with open(os.path.join(out_dir, "merges.txt"), "w", encoding="utf-8") as f:
        for p1, p2 in merges:
            # transform p1 and p2
            s1 = "".join(byte_encoder[b] for b in p1)
            s2 = "".join(byte_encoder[b] for b in p2)
            f.write(f"{s1} {s2}\n")

def main():
    input_path = "data/TinyStoriesV2-GPT4-train.txt" # 你的原始文本路径
    vocab_size = 10_000

    special_tokens = ["<|endoftext|>"]
    output_dir = "data/TinyStoriesV2-GPT4-train"

    print(f"start train BPE tokenizer, target vocab size {vocab_size}")
    print("")

    # call the logic to train BPE
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)

    # save results
    save_tokenizer_files(vocab, merges, output_dir)


if __name__ == "__main__":
    main()
