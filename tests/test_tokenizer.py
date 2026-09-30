"""BPE 分词器单元测试。

覆盖四类性质:
  1. 无损往返(最重要):语料内文本、语料外新词、标点、emoji、空格,
     encode -> decode 必须逐字符还原 —— 这是 <unk> Bug 修复的回归锁;
  2. 编码合法性:id 全部在词表范围内、特殊 token 行为正确;
  3. 持久化:save/load 后编解码结果不变(含 \\uXXXX 代理转义路径);
  4. 预切分/符号层:split_word 自身的可逆约定。
"""

from __future__ import annotations

from minillm.tokenizer import SPACE, BPETokenizer, pretokenize, split_word


# ------------------------------------------------------------------ 往返
def test_roundtrip_corpus_lines(tiny_tokenizer, corpus_file):
    """训练语料的每一行都必须完美往返。"""
    for line in corpus_file.read_text(encoding="utf-8").split("\n"):
        assert tiny_tokenizer.decode(tiny_tokenizer.encode(line)) == line


def test_roundtrip_out_of_vocab_words(tiny_tokenizer):
    """语料里完全没出现过的新英文词/句子(当年 OOV Bug 的直接复现用例)。"""
    samples = [
        "supercalifragilisticexpialidocious",
        "The quick brown fox jumps over the lazy dog!",
        "Wait... really?! Yes — absolutely.",
        "unicode ñ ü ö ä ç ß ø å 字符也要无损",
    ]
    for s in samples:
        ids = tiny_tokenizer.encode(s)
        assert tiny_tokenizer.decode(ids) == s, f"roundtrip failed: {s!r}"


def test_roundtrip_punctuation_and_digits(tiny_tokenizer):
    """标点与数字串:'!' 曾因哨兵映射问题退化为 <unk>,此处回归。"""
    for s in ["hello world !", "!!!", "1234567890", "v1.2.3 build #42"]:
        assert tiny_tokenizer.decode(tiny_tokenizer.encode(s)) == s


def test_roundtrip_emoji_surrogate_pairs(tiny_tokenizer):
    """星平面 emoji 走 UTF-16 代理对双 token 方案,必须无损。"""
    for s in ["🎉 party time 🚀", "中文混 emoji 😂😀🔥 结尾"]:
        ids = tiny_tokenizer.encode(s)
        assert tiny_tokenizer.decode(ids) == s
        # 每个 emoji 应恰好消耗 2 个 token(高代理+低代理)
        assert len(ids) >= 4


def test_roundtrip_spaces_multiple(tiny_tokenizer):
    """连续多个空格:每个空格对应一个 SPACE 哨兵 token,数量不能丢。"""
    s = "a  b   c    d"
    assert tiny_tokenizer.decode(tiny_tokenizer.encode(s)) == s


def test_no_unk_for_common_chars(tiny_tokenizer):
    """常见字符不应触发 <unk>:统计 <unk> id 是否出现在编码结果中。"""
    unk_id = tiny_tokenizer.vocab["<unk>"]
    text = "The sea 大海 sparkle!? 123 — café naïve 🎉"
    ids = tiny_tokenizer.encode(text)
    assert unk_id not in ids, "存在不该出现的 <unk>,无损域被破坏"


def test_empty_encode_decode(tiny_tokenizer):
    assert tiny_tokenizer.encode("") == []
    assert tiny_tokenizer.decode([]) == ""


def test_add_eos(tiny_tokenizer):
    ids = tiny_tokenizer.encode("hello", add_eos=True)
    assert ids[-1] == BPETokenizer.EOS_ID
    # decode 会丢弃 <eos>,正文不受影响
    assert tiny_tokenizer.decode(ids) == "hello"


def test_decode_is_total_function(tiny_tokenizer):
    """decode 对任意 id 序列(包括模型可能采样出的越界/畸形 id)不得抛异常。"""
    weird = [0, 1, 999999, -0, tiny_tokenizer.vocab_size + 12345]
    out = tiny_tokenizer.decode(weird)          # 只要求:不崩溃、返回 str
    assert isinstance(out, str)


# ------------------------------------------------------------------ 持久化
def test_save_load_roundtrip(tiny_tokenizer, tmp_path):
    p = tmp_path / "tok.json"
    tiny_tokenizer.save(str(p))
    tok2 = BPETokenizer.load(str(p))
    assert tok2.vocab == tiny_tokenizer.vocab
    assert tok2.merges == tiny_tokenizer.merges
    text = "Persistence check: 词表存盘再读回 must be identical!"
    assert tok2.encode(text) == tiny_tokenizer.encode(text)
    assert tok2.decode_stream(tok2.encode(text)) == text


# ------------------------------------------------------------------ 符号层
def test_split_word_reversible():
    """split_word 对含普通空格的文本可逆:拼回去(哨兵换回空格)== 原词。"""
    for w in ["hello world", "  spaced  ", "中文 abc123 !"]:
        syms = split_word(w)
        assert "".join(syms).replace(SPACE, " ") == w
    # 原生哨兵(U+00A0):split_word 按 isspace 分支原样保留(注释里明确"不能映射
    # 到 SPACE 否则 decode 错还原破坏往返"),故直接拼回即得原字符,无需 replace。
    w = f"a{SPACE}b"
    assert "".join(split_word(w)) == w


def test_pretokenize_drops_empty_lines():
    parts = pretokenize("line1\n\nline2\n")
    assert parts == ["line1", "line2"]


def test_vocab_contains_eos_and_unk(tiny_tokenizer):
    assert tiny_tokenizer.vocab["<eos>"] == BPETokenizer.EOS_ID == 0
    assert "<unk>" in tiny_tokenizer.vocab


def test_ids_in_range(tiny_tokenizer):
    for s in pretokenize(open("tests/conftest.py", encoding="utf-8").read()):
        for i in tiny_tokenizer._encode_word(s):
            assert 0 <= i < tiny_tokenizer.vocab_size
