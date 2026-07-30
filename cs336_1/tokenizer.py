import re
from collections import Iterable
from typing import Union

from IPython.terminal.shortcuts.filters import preceding_text

"""
for special token:
    推理/编码阶段 (tokenizer.encode)
        在模型使用分词器将文本转为 ID 时，必须优先匹配特殊 Token
    代码逻辑：
        正则匹配：构建一个包含所有特殊 Token 的正则表达式。
        优先级：先扫描文本，一旦发现特殊 Token，直接将其转为对应的 ID。
        普通处理：特殊 Token 之间的文本，再走正常的 GPT-2预分词和 BPE 合并流程。
"""

class BPETokenizer:
    """
    字节级 BPE (Byte-Pair Encoding) 分词器实现。
    该分词器将任意字符串编码为整数 ID 序列，并能将 ID 序列还原。
    它采用字节级处理，确保不会出现未知词 (OOV) 错误。
    """

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: Union[list[[str], None]] = None):
        """
        initialize tokenizer

        :param vocab: 词汇表，建立整数 ID 到字节块 (bytes) 的映射
        :param merges: 合并规则列表。列表中的每一项是一个二元组 (bytes_a, bytes_b),
                          表示在训练过程中 bytes_a 和 bytes_b 被合并的顺序
        :param special_tokens: 特殊标记列表（如 <|endoftext|>），这些标记不会被 BPE 规则拆分。
        """

        # 1. 建立双向映射，方便查表
        self.vocab = vocab  # ID -> 字节块
        self.id_to_byte = vocab
        self.byte_to_id = {v: k for k, v in vocab.items()}

        # 2. 将合并规则转换为 Rank 字典。
        # BPE 编码时，必须优先应用在训练阶段较早出现的合并规则
        # 字典结构为: {(bytes_a, bytes_b): 顺序索引}
        self.merges = {pair: i for i, pair in enumerate(merges)}
        self.special_tokens = special_tokens or []

        # 3. 构建特殊 Token 的正则表达式
        if self.special_tokens:
            # 关键：必须按照长度从长到短排序 (reverse=True)
            # 这样正则引擎会优先匹配最长的特殊标记，防止重叠标记 (如 <|a|><|b|>)被错误拆分。
            sorted_special = sorted(self.special_tokens, key=len, reverse=True)
            # 使用 re.escape 确保标记中的特殊字符 (如 | 或 [ ) 被当作普通字符处理
            special_pattern = "|".join(re.escape(t) for t in sorted_special)
            self.special_regex = re.compile(special_pattern)
        else:
            self.special_regex = None

        # 4. GPT-2 官方预分词正则表达式。
        # 它的作用是在应用 BPE 合并前，先将文本切分成单词、标点、数字等逻辑块。
        # 这样做是为了防止 BPE 规则跨越单词或标点（例如：防止将 "dog" 的末尾和 "." 合并）。
        self.gpt2_pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def encode(self, text: str) -> list[int]:
        """
        将输入的原始字符串编码为整数 ID 列表。

        该方法的核心逻辑是：
        1. 作为一个"协调者"，她负责处理文本中的特殊标记 (Special Tokens) 和普通文本。
        2. 特殊标记 (如 <|endoftext|>) 被视为原子，直接映射 ID，不参与 BPE 的拆分和合并。
        3. 普通文本片段则被交给底层逻辑执行预分词和 BPE 算法。

        :param
            text: 需要编码的原始字符串 (例如 "Hello <|end|> World")
        :return:
            list[int]: 编码后的整数 ID 列表
        """

        # 1. 边界情况检查
        # if input is None or empty string, return empty list
        # that's cause aviod erro whtn the next logic process the empty context
        if not text:
            return []

        # 2. situation A Fast Path
        # if we not define any special tag when initialize (or special tag list is empty)
        # so, all the text could be a section of continuous "normal text"
        # we call inner function _encode_text_segment
        if not self.special_regex:
            return self._encode_text_segment(text)

        # 3. situation B process the complex text with special tokens
        # current text may contain both normal text and special tokens, we need to split it.
        tokens = []

        # las_pos for reload the position of previous match end, help us locate the gap between "special token"
        # last_pos 用于记录上一次匹配结束的位置，帮助我们定位“特殊标记”之间的“缝隙”。
        last_pos = 0

        # 使用 finditer 遍历文本中所有含特殊标记模式的匹配项
        # finditer 的好处是它提供了 match.start() 和 match.end()
        # 这让我们能够精确地知道特殊标记在哪里开始，在哪里结束
        for match in self.special_regex.finditer(text):

            # 3.1 提取并处理“前置普通文本”
            # 区间是 [last_pos, match.start]
            # " hello <|endoftext|> world"
            # 这段文本是两个特殊标记(or 开头到第一个标记、最后一个标记结束到末尾)之间的普通文本
            pre_text = text[last_pos:match.start()]

            # 如果两个标记之间确实有文字 (length > 0)
            if pre_text:
                # 调用 BPE 逻辑。 _encode_text_segment 会执行：
                # 1st. GPT-2 预分词正则切分
                # 2nd. 字节化。
                # 3rd. 按照 merges 规则进行贪婪合并
                tokens.extend(self._encode_text_segment(pre_text))
                # pre_tokens: [1,2,3,...] self._encode_text_segment: [4,5,6]
                # tokens.extend -> [1,2,3,...,4,5,6]
                # token.append() : [1,2,3,...,[4,5,6]]

            # 3.2 处理“当前特殊标记”
            # match.group() 拿到的就是被识别出来的特殊标记字符串 (如 "<|endoftext|>")
            special_tok = match.group()

            # core principle: special token not participate in BPE merge!
            # we encode special token with UTF-8 bytes, then find its ID in vocab
            # notice: these tokens has been append the vocab, when stage of trian_bpe
            tokens.append(self.byte_to_id[special_tok.encode("utf-8")])

            # 3.3 更新游标
            # 将游标移动到当前匹配项得末尾，为寻找下一个片段做准备。
            last_pos = match.end()

        # 4. 处理 "收尾文本"
        # 如果最后一个特殊标记后面还有文字（例如 "Hello<|end|>World" 中的 "World"），
        # 或者整个文本根本没有特殊标记匹配（虽然逻辑上 Case A 已处理，但这里是双重保险），
        # 我们需要处理从 last_pos 到字符串末尾的所有剩余字符。
        remaining_text = text[last_pos:]
        if remaining_text:
            # 剩余部分同样作为普通文本片段进行 BPE 编码
            tokens.extend(self._encode_text_segment(remaining_text))

        # 返回拼接好的所有 ID 列表
        return tokens


    def _encode_text_segment(self, text: str) -> list[int]:
        """
        inner core function, merge the text without special tokens by BPE rule

        :param text:
        :return:
        """
        ids = []
        # use GPT-2 regularization to pre_tokenization, split text into word/punctuation chunk
        # e.g. "Hello world!" -> ["Hello", " world", "!"]
        pre_tokens = self.gpt2_pat.findall(text)

        for p_tok in pre_tokens:
            # 1st. transform current fragment into bytes list, and treat every byte as an independent "part"
            # e.g. "hello" -> [b'h', b'e', b'l', b'l', b'o']
            byte_parts = [bytes([b]) for b in p_tok.encode("utf-8")]

            # 2nd. repeatedly perform the merge operation, until no valid merge rules are left.
            while len(byte_parts) >= 2:
                # 当前列表的所有相邻对钟，寻找合并优先级最高 (Rank 最小) 的一对，即按照构造 merge 时，添加 pair 的顺序进行合并
                best_pair = None
                min_rank = float("inf")

                for i in range(len(byte_parts) - 1):
                    pair = (byte_parts[i], byte_parts[i + 1])
                    if pair in self.merges:
                        rank = self.merges[pair]
                        if rank < min_rank:
                            min_rank = rank
                            best_pair = pair

                # 如果找不到任何可以合并的规则，退出当前片段的合并过程
                if best_pair is None:
                    break

                # 3rd. perform the merge operation
                # traversal the current list, replace all the best_pair with the merged long byte block
                # e.g. [b'H', b'e', b'l', b'l', b'o', b'H', b'e'] -> [b'He', b'l', b'l', b'o', b'He']
                new_byte_parts = []
                i = 0

                while i < len(byte_parts):
                    # 如果当前两个部分匹配最高优规则
                    if i < len(byte_parts) - 1 and (byte_parts[i], byte_parts[i+1]) == best_pair:
                        new_byte_parts.append(best_pair[0] + best_pair[1])
                        i += 2
                    # not match, push current char into new_byte_parts
                    else:
                        new_byte_parts.append(byte_parts[i])
                        i += 1
                # update the list, continue the next loop to merge the next best_pair
                byte_parts = new_byte_parts

            # 4th. transform the final byte blocks into IDs in vocab
            for part in byte_parts:
                ids.append(self.byte_to_id[part])

        return ids

    def decode(self, ids: list[int]) -> str:
        """
        将 ID 列表解码为原始字符串。
        :param ids:
        :return:
        """

        # 1. find byte blocks in vocab by ID
        byte_segments = [self.id_to_byte[i] for i in ids]

        # 2. concatenate all byte blocks into a complete byte stream in order
        full_bytes = b"".join(byte_segments)

        # 3. decode byte stream into UTF-8 string
        # 使用 error = "replace" 非常关键：因为 BPE 可能会生成不完整的字节序列
        # (e.g. 3 字节的中文字符只产生了一部分)，此时不报错二十插入替换符号 ().

        return full_bytes.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """
        内存高效的迭代解码器

        :param
            iterable: 一个可迭代的字符串对象 (例如文件句柄)
        :return:
            一个生成器，逐个阐述解码后的 ID。用于处理无法一次性读入内存的大文件。
        """
        for chunk in iterable:
            # 对每一个文本进行编码，并通过 yield 吐出结果
            yield from self.encode(chunk)


