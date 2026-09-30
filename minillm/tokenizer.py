"""从零实现的 BPE 分词器(字节对编码)。

流程与 GPT-2 一致,但完全自实现:
1. 预切分:按换行符切成行单元;英文单词/数字串/缩写作为初始符号,
   中文等其他非 ASCII 字符天然逐字符切开(见 _TOKEN_RE);
   空格用哨兵字符 U+00A0 表示,保证子词可以跨空格合并且解码可逆;
2. 迭代合并:统计语料中所有行内相邻符号对的频率,反复合并最高频对,
   直到达到目标合并次数;词表始终包含每个单字符,保证任意文本可退到字符级;
3. 编码:对新文本先做预切分,再按学习到的合并优先级(merge rank)全局贪心合并,
   只执行"合并结果仍是词表成员"的规则;仍未登录的符号拆成单字符兜底;
4. 解码:token id -> 字符串直接拼接(哨兵还原为空格),无损往返。

支持 JSON 持久化,便于训练后复用。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

SPACE = "\u00a0"  # 空格哨兵,参与合并/存储,解码时还原为普通空格
# GPT-2 风格预分词:缩写、字母序列、数字串、单个其他非空白字符。
# 注意 [^\sA-Za-z0-9] 会把中文等所有非 ASCII 字符也匹配为"单字符",
# 因此无需再为非 ASCII 单独写分支——它们天然逐字符切开。
_TOKEN_RE = re.compile(r"""'[sS]|'(t|re|ve|ll|m|d)|[A-Za-z]+|[0-9]+|[^\sA-Za-z0-9]""")

# ----------------------------------------------------------------------
# 兜底原子符号集(BMP 全集):即使训练语料中从未出现,也必须进词表的字符。
# ----------------------------------------------------------------------
# 原因:BPE 的无损性依赖"任何 token 都能退到单字符";若某字符不在词表,
# 编码含该字符的新文本时只能产出 <unk>,往返被破坏。
#
# 为什么直接并入整个 BMP(U+0000..U+FFFF)?
#   逐个枚举"常用字符"(ASCII 标点、CJK 基本区、常见符号……)永远会漏:
#   emoji 在 U+1F300+(星平面)、日文假名/韩文/扩展汉字各有自己的区段,
#   手工白名单不可能穷尽。BMP 只有 65536 个码点,全部入表后:
#     * 任何 BMP 字符(拉丁、希腊、西里尔、假名、CJK、符号……)零 OOV;
#     * 星平面字符(emoji 等)可由代理对 U+D800..U+DFFF 两个 token 拼回
#       ——这正是 UTF-16 的编码方式,decode 时对孤立代理做 surrogatepair
#       重组即可无损还原(见 decode 实现);
#   于是"字符级 BPE + 有限兜底集"升级为**事实上的全 Unicode 无损方案**,
#   工程上等价于字节级 BPE 的可读性(GPT-2 用 256 个字节原子达到同样目的,
#   代价是每个非 ASCII 字符都变成 2~3 个乱码字节 token;我们保留字符原子,
#   中文一个字就是一个 token,序列更短、embedding 语义更直接)。
#
# 成本:词表多 ~6.5 万 id。对本工程的 512 维小模型,embedding 增加约
#   65536*512*2(输入+输出权重)= ~67M 参数——所以默认配置里我们把
#   兜底集做成开关(Config.full_bmp_atoms),教学演示用小词表更快;
#   scripts/train_tokenizer.py --full-bmp 可打开全集换取真正零 OOV。
_BMP_EXCLUDE = set("\n\r")  # 换行/回车是预切分边界,不进入词表(见 split_word)


def _bmp_atoms() -> str:
    """BMP 全部可打印码点(U+0000..U+FFFF),剔除换行类控制符与已用的哨兵冲突。

    注意 U+00A0(NBSP)已被用作空格哨兵 SPACE——它本来就在 BMP 里,
    入表一次即可,无需特判:哨兵与普通字符共享同一张词表。

    实现细节(两个坑):
      1. 孤立代理码点(D800-DFFF)在 Python 字符串里是"合法字符",但无法
         被 UTF-8 编码 —— 它们作为 emoji 的编码载体必须留在**内存词表**中,
         由 save() 的 \\uXXXX 转义负责落盘,这里不做剔除;
      2. Cc/Cf/Co/Zl/Zp 等控制与格式类字符(如 U+0000 NUL、U+FEFF BOM)
         几乎不会出现在正常文本里,却会让词表虚胖并可能干扰 JSON/终端输出,
         直接排除。代价:这类字符会退化为 <unk>(实践中无影响)。
    """
    return "".join(
        ch for ch in (chr(c) for c in range(0x10000))
        if ch not in _BMP_EXCLUDE and unicodedata.category(ch)[0] not in ("C", "Z")
    )


_BASE_ATOMS = _bmp_atoms()

# 数字子串预分词上限:连续数字切成最多 N_DIGITS 位一块(如 '1234567' ->
# ['123','456','7'])。这是 GPT-2 官方 [0-9]+ 单次长匹配规则的**有意偏离**:
# 长数字串整体作为一个初始符号会进入词表,但它不是原子字符——一旦语料外
# 出现更长的同类数字串,整词无法退到字符级,产生不可逆的 <unk>。限制长度
# 后所有预符号都是短片段,任意数字串都能由它们拼回,保证无损。
N_DIGITS = 3


def _split_supplementary(word: str) -> list[str]:
    """把含星平面字符(U+10000+,主要是 emoji)的词拆成 UTF-16 代理对符号。

    为什么需要这一步:词表只覆盖 BMP(U+0000..U+FFFF),而 emoji 位于
    星平面,单个 python 字符 chr(0x1F389) 无法直接作为符号入表。
    解决方式借用 UTF-16 的标准做法:一个星平面码点 = 高代理(D800-DBFF)
    + 低代理(DC00-DFFF) 两个 16 位单元。我们把这两个单元各自当作一个
    "BMP 符号"(它们已在 _bmp_atoms 兜底集里),编码时产出 2 个 token,
    解码时把"孤立高代理 + 紧随其后的孤立低代理"重新拼回原字符 —— 全程
    无损且不需要引入任何词表外 id。
    注意:代理单元永远不会单独与别的符号发生 BPE 合并(split_word 已把
    它们隔离成相邻的两个符号,merge 阶段它们组合出的字符串不在词表中,
    会被 banned 机制跳过),因此该方案与既有合并规则完全兼容。
    """
    out: list[str] = []
    for ch in word:
        cp = ord(ch)
        if cp >= 0x10000:  # 星平面:转成 (高代理, 低代理) 两个 BMP 符号
            cp -= 0x10000
            hi = chr(0xD800 + (cp >> 10))       # 高 10 位 -> D800..DBFF
            lo = chr(0xDC00 + (cp & 0x3FF))     # 低 10 位 -> DC00..DFFF
            out.extend([hi, lo])
        else:
            out.append(ch)
    return out


def split_word(word: str) -> list[str]:
    """把一个行单元拆成初始符号:正则切分 + 未覆盖字符逐字兜底;空格转哨兵。

    保证: "".join(split_word(w)).replace(SPACE," ") == w(编解码一一对应)。
    """
    out: list[str] = []
    i, n = 0, len(word)
    while i < n:
        ch = word[i]
        if ch == " ":
            out.append(SPACE)
            i += 1
            continue
        m = _TOKEN_RE.match(word, i)
        if m and m.start() == i and m.end() > i:
            tok = m.group()
            # 数字子串按固定宽度分块(见 N_DIGITS 注释),其余原样入列
            if tok[0].isdigit() and len(tok) > N_DIGITS:
                out.extend(tok[j:j + N_DIGITS] for j in range(0, len(tok), N_DIGITS))
            else:
                out.append(tok)
            i = m.end()
        else:
            # 正则未覆盖的字符(星平面 emoji、除 \r\t\n 外的其他空白等):
            # 逐字符兜底。先走 _split_supplementary —— 星平面字符会被拆成
            # UTF-16 代理对两个 BMP 符号(它们都在词表里),BMP 字符原样入列。
            ch2 = word[i]
            if ord(ch2) >= 0x10000:
                out.extend(_split_supplementary(ch2))
            elif ch2.isspace():
                # 罕见空白(U+2009 窄空格、\f 换页等):保留为原子本身。
                # 这些码点都在 BMP 兜底集内,可无损回退;不能映射到 SPACE,
                # 否则 decode 会把窄空格错还原成普通空格,破坏往返。
                out.append(ch2)
            else:
                out.append(ch2)   # 理论上不可达(BMP 非空白已被正则覆盖),保险兜底
            i += 1
    # 出口统一做星平面后处理:正则分支可能把 emoji 整字符匹配下来
    # (`[^\sA-Za-z0-9]` 会命中 U+1F389),这里把它们拆成代理对,
    # 保证 train/encode 两条路径看到的符号空间完全一致(BMP only)。
    return _split_surrogate_runs(out)


def basic_symbols(text: str) -> list[str]:
    """返回词表需要的全部"原子符号":语料中出现过的单字符 + 兜底常用字符集。

    空格映射为哨兵,换行是切分边界不入词表。
    词表必须包含每一个可能的原子符号:BPE 的无损性依赖"任何 token 都能退到
    字符级"。若某个字符不在词表中,编码未见过的文本时就只能产出 <unk>,
    往返被破坏——因此除了语料实际出现的字符,还要并入 _BASE_ATOMS 兜底集。
    """
    seen: set[str] = {ch for ch in _BASE_ATOMS if ch != "\n"}
    for ch in text:
        if ch == "\n":          # 换行是切分边界,不进入词表
            continue
        seen.add(SPACE if ch == " " else ch)
    return sorted(seen)


def pretokenize(text: str) -> list[str]:
    """把文本切成行单元(空行丢弃;行内保留空格,由 split_word 处理)。

    注意:换行是"切分边界",本身不进入任何符号——因此含 \\n 的多行文本
    encode+decode 后换行会丢失。这是 GPT-2 式按行 BPE 的固有取舍,对本工程
    "一行一句"的训练语料完全无损;若要严格无损请逐行编解码后再自行拼接换行。
    """
    parts = text.split("\n")
    return [p for p in parts if p]


def _split_surrogate_runs(symbols: list[str]) -> list[str]:
    """把符号列表中的"星平面字符"就地拆成 UTF-16 代理对两个 BMP 符号。

    为什么放在这里而不是 split_word 内部:
      * split_word 的正则 `[^\\sA-Za-z0-9]` 会把 🎉 整字符匹配下来,
        之前写在 else 分支里的代理拆分永远不会被触发(实测 bug);
      * 训练阶段(BPETokenizer.train)直接调用 split_word 构建词表初始
        符号 —— 如果只改编码路径,emoji 会以整字符形态进入 merge 统计,
        却不在 BMP 词表原子集里,合并/查表行为不一致。
      * 统一在 split_word 的**出口**做后处理,train 与 encode 两条路径
        自动共享同一套符号空间:词表原子 = BMP 全集 ∪ 语料原子,
        emoji 永远以 (高代理, 低代理) 两个合法 id 出现,往返无损。
    """
    if not any(len(s) == 1 and ord(s) >= 0x10000 for s in symbols):
        return symbols  # 快速路径:绝大多数行不含星平面字符,零开销返回
    out: list[str] = []
    for s in symbols:
        if len(s) == 1 and ord(s) >= 0x10000:
            out.extend(_split_supplementary(s))   # 单字符 emoji -> 代理对
        else:
            # 多字符符号理论上不含星平面(正则产物都是 BMP),保险起见仍检查
            out.append(s if all(ord(c) < 0x10000 for c in s)
                       else "".join(_split_supplementary(s)))
    return out


class BPETokenizer:
    EOS_ID = 0  # 特殊 token <eos> 固定为 id 0

    def __init__(self, vocab: dict[str, int], merges: list[tuple[str, str]]):
        self.vocab = vocab                      # token 字符串 -> id
        self.merges = merges                    # 有序合并规则(按学习顺序)
        # 兜底 <unk>:仅当文本含训练语料之外的字符时使用(正常数据不会出现)
        if "<unk>" not in self.vocab:
            self.vocab["<unk>"] = len(self.vocab)
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        # merge_rank[(a,b)] = 规则 (a,b) 的学习次序;次序越小越优先应用
        self._merge_rank = {pair: i for i, pair in enumerate(merges)}
        self.special_tokens = {"<eos>", "<unk>"}

    # ------------------------------------------------------------------ 训练
    @classmethod
    def train(
        cls,
        text: str,
        vocab_size: int = 8192,
        verbose: bool = True,
    ) -> "BPETokenizer":
        """在语料上训练 BPE。

        vocab_size 控制"新增合并次数"上限(与 GPT-2 计数方式一致);
        最终词表大小 = 原子符号数 + 实际合并数(+<eos>/<unk>),可能略大于该值,
        调用方应以 tokenizer.vocab_size 为准来配置模型 embedding 维度。
        """
        word_freq = Counter(pretokenize(text))            # 行 -> 出现次数
        ws = {w: split_word(w) for w in word_freq}        # 工作区:行 -> 符号列表

        vocab: dict[str, int] = {"<eos>": 0}
        next_id = 1
        # 第一步:所有原子符号(单字符)入表,这是无损往返的根基
        for s in basic_symbols(text):
            if s not in vocab:
                vocab[s] = next_id
                next_id += 1

        # 第二步:迭代合并最高频相邻符号对,直到达到合并次数上限或无对可合并。
        # 工程加速技巧:维护"符号对频次堆"(最大堆),而不是每轮全量重扫语料。
        #   - heap 里可能有过期条目(某对的真实频次已变小),弹出时对比 cur[pair]
        #     与堆中记录值,不一致则丢弃并重新入堆 —— 惰性删除(lazy deletion);
        #   - pair_pos[(a,b)] = 包含该对的行集合,合并后只需更新受影响行。
        import heapq
        pair_freq: Counter = Counter()
        pair_pos: dict[tuple[str, str], set[str]] = {}
        for w, syms in ws.items():
            f = word_freq[w]
            for i in range(len(syms) - 1):
                p = (syms[i], syms[i + 1])
                pair_freq[p] += f
                pair_pos.setdefault(p, set()).add(w)
        # 最大堆用负频次模拟;同频按字典序保证确定性
        heap = [(-c, p) for p, c in pair_freq.items()]
        heapq.heapify(heap)

        max_merges = max(0, vocab_size - 1)
        merges: list[tuple[str, str]] = []
        while len(merges) < max_merges and heap:
            # 2.1 弹出堆顶;若与当前真实频次不符说明是过期条目,刷新后重新入堆
            neg_c, (a, b) = heapq.heappop(heap)
            cur = pair_freq.get((a, b), 0)
            if cur == 0:
                continue
            if -neg_c != cur:
                heapq.heappush(heap, (-cur, (a, b)))
                continue
            best_cnt = cur
            merged = a + b
            merges.append((a, b))
            if merged not in vocab:
                vocab[merged] = next_id
                next_id += 1
            # 2.2 只在包含该对的行里做一次线性扫描替换(O(W) 而非 O(V*W)),
            #     并同步更新受影响行的所有相邻对计数
            affected = list(pair_pos.pop((a, b)))
            for w in affected:
                syms = ws[w]
                old_pairs = [(syms[i], syms[i + 1]) for i in range(len(syms) - 1)]
                new = []
                i = 0
                n = len(syms)
                while i < n:
                    if i < n - 1 and syms[i] == a and syms[i + 1] == b:
                        new.append(merged)
                        i += 2
                    else:
                        new.append(syms[i])
                        i += 1
                ws[w] = new
                f = word_freq[w]
                # 旧对计数减、新对计数加,并把新出现的对入 pair_pos/堆
                for p in old_pairs:
                    pair_freq[p] -= f
                    if pair_freq[p] <= 0:
                        del pair_freq[p]
                    sset = pair_pos.get(p)
                    if sset is not None:
                        sset.discard(w)
                        if not sset:
                            del pair_pos[p]
                newsyms = ws[w]
                for i in range(len(newsyms) - 1):
                    p = (newsyms[i], newsyms[i + 1])
                    if p not in pair_freq:
                        heapq.heappush(heap, (-f, p))
                    pair_freq[p] = pair_freq.get(p, 0) + f
                    pair_pos.setdefault(p, set()).add(w)
            if verbose and len(merges) % 2000 == 0:
                print(f"[tokenizer] merges={len(merges)} vocab={len(vocab)} "
                      f"last=({a!r},{b!r})->{merged!r} count={best_cnt}")

        return cls(vocab, merges)

    # ------------------------------------------------------------------ 编解码
    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        """把文本编码为 token id 序列(按行切分,逐行做 BPE 合并)。"""
        ids: list[int] = []
        for word in pretokenize(text):
            ids.extend(self._encode_word(word))
        if add_eos:
            ids.append(self.EOS_ID)
        return ids

    def _encode_word(self, word: str) -> list[int]:
        symbols = split_word(word)
        # 全局贪心:反复应用 rank 最小(最早学到)的相邻合并规则。
        # 关键约束:只有当合并结果本身也是词表成员时才执行——
        # 否则语料外的词(如 'hello' 被 'he'+'llo' 拼成词表外符号)会卡死在
        # 一个查不到 id 的中间态,导致整词退化为 <unk>。
        banned: set[tuple[str, str]] = set()
        while len(symbols) > 1:
            best_rank, best_i = 1 << 30, -1
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                if pair in banned:
                    continue
                r = self._merge_rank.get(pair)
                if r is not None and r < best_rank:
                    best_rank, best_i = r, i
                    if r == 0:
                        break
            if best_i < 0:
                break
            merged = symbols[best_i] + symbols[best_i + 1]
            if merged in self.vocab:
                symbols[best_i:best_i + 2] = [merged]
            else:
                # 规则会产出词表外符号:本次编码内禁用它再重试(不改全局状态)
                banned.add((symbols[best_i], symbols[best_i + 1]))

        unk = self.vocab.get("<unk>")
        ids: list[int] = []
        stack: list[str] = list(reversed(symbols))  # 栈式处理,避免深递归
        while stack:
            s = stack.pop()
            sid = self.vocab.get(s)
            if sid is not None:
                ids.append(sid)
                continue
            # 未登录符号兜底:拆成单字符继续尝试(词表含全部原子符号,
            # 正常只会命中一次拆分;<unk> 仅在文本含语料外字符时出现)。
            if len(s) > 1:
                stack.extend(reversed(list(s)))
            elif unk is not None:
                ids.append(unk)
        return ids

    def decode(self, ids: list[int]) -> str:
        """token id -> 文本:<eos>/<unk> 跳过,哨兵还原为空格,代理对重组。

        三步特殊处理,每一步都对应编码端的一个可逆设计:
          1. <eos>(id=0)是文档分隔符,不属于文本内容,直接丢弃;
          2. SPACE 哨兵(U+00A0)在 split_word 里代替普通空格参与合并,
             这里 replace 回真正的空格;
          3. 星平面字符(emoji)被编码成"高代理+低代理"两个 token
             (见 _split_supplementary),这里扫描字符串流,把
             D800-DBFF 紧跟 DC00-DFFF 的相邻对重新合成原始码点。
             孤立代理(理论不会出现,除非模型采样出畸形序列)原样保留,
             不会抛异常——保证 decode 对任意 id 序列都是全函数。
        """
        parts = []
        for i in ids:
            if i == self.EOS_ID:
                continue
            t = self.id_to_token.get(i, "")
            if t in ("<unk>", "<eos>"):   # 特殊符号不产出文本
                continue
            parts.append(t.replace(SPACE, " "))
        s = "".join(parts)
        # ---- 代理对重组:UTF-16 surrogate pair -> 星平面单字符 ----
        out: list[str] = []
        k, n = 0, len(s)
        while k < n:
            c = s[k]
            cp = ord(c)
            if 0xD800 <= cp <= 0xDBFF and k + 1 < n:      # 高代理
                lo = ord(s[k + 1])
                if 0xDC00 <= lo <= 0xDFFF:                 # 紧随低代理 => 合法代理对
                    out.append(chr(0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00)))
                    k += 2
                    continue
            out.append(c)
            k += 1
        return "".join(out)

    def decode_stream(self, ids: list[int]) -> str:
        """增量解码(服务场景),等价于 decode。"""
        return self.decode(ids)

    # ------------------------------------------------------------------ 持久化
    # JSON 文件必须是合法 UTF-8,而"孤立代理字符"(D800-DFFF)无法被
    # UTF-8 编码器接受(它们只在成对出现时才有意义)。词表里恰好包含
    # 这些作为原子符号的代理码点,因此存盘前把每个特殊字符转义成
    # "\uXXXX" 形式(json.dumps(ensure_ascii=True) 会自动做这件事),
    # 读回时 json 解析又会还原为原字符 —— 词表内容一字不差。
    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "vocab": self.vocab,
            "merges": [[a, b] for a, b in self.merges],
        }
        with open(path, "w", encoding="utf-8") as f:
            # ensure_ascii=True:所有非 ASCII 字符(含代理项)写成 \uXXXX 转义
            json.dump(data, f, ensure_ascii=True)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)  # \uXXXX 转义在解析时自动还原为真实字符
        return cls(data["vocab"], [tuple(m) for m in data["merges"]])

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def __repr__(self) -> str:
        return f"BPETokenizer(vocab_size={self.vocab_size}, merges={len(self.merges)})"
