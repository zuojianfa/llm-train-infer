# 教程 01：BPE 分词器 —— 把文字变成 token

对应代码：`minillm/tokenizer.py`，入口 `scripts/train_tokenizer.py`。

## 1. 为什么需要分词器？
模型只能处理数字。字符级太碎（序列长）、词级又无法处理未登录词（OOV）。
BPE（Byte Pair Encoding）折中：**从单字符出发，反复合并语料中最频繁的相邻对**，
高频词组获得短 id，罕见词自动退回字符组合——任何文本都能编码，永不失败。

## 2. 训练三阶段
1. **预切分**：按 Unicode 类别把文本切成"原子"（字母/数字连续段、每个标点独立、空格附着前词）。中文每个汉字天然是独立原子。
2. **统计合并**：反复找出现次数最多的相邻符号对，记录规则 `(a,b)->ab`，直到达到目标词表大小。
3. **建表**：基础词表 = 全部原子字符 + `<unk>` + 特殊符；merge 规则表供编码时回放。

## 3. 无损往返与 OOV 兜底（本工程的关键修复）
朴素 BPE 的坑：编码一个语料外的词时，若盲目套用 merge 规则会拼出**词表里不存在的符号**，
只能退化成 `<unk>`，解码后原文丢失。本工程的策略：
- **只有当 a+b 本身是词表成员时，才应用该条 merge 规则**；
- 否则保留更细的符号，直到每片都是词表成员。
于是 encode→decode 对任意 Unicode 文本（含 emoji、标点、生僻字）**逐字符可逆**。
回归测试见 `tests/test_tokenizer.py`（当年 Bug 的复现用例被永久锁定）。

## 4. 使用
```bash
python scripts/train_tokenizer.py --data data/train.txt --vocab-size 8192 --out out/tokenizer.json
```
