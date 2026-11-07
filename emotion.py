"""Fish Audio 情绪标签：识别、整理、去除。

纯 Python，不依赖 AstrBot，方便单独测试。分两套语法：

**S2 系列**——方括号写情绪/语气，例如 ``[happy] 今天天气真好``、``[sad][whispering] 我想你了``。

**Drama 3**——方括号是"导演指令"（自然语言，任意语言都行），另外多一种只改局部几个词的
成对尖括号标签：``<whisper>`` ``<emphasis>`` ``<soft>`` ``<fast>`` ``<slow>`` ``<stress>``
``<forceful>``，例如 ``[轻声安慰]没关系，<soft>慢慢来。</soft>``。
成对标签只在 ``EmotionRules.paired`` 为真时识别（由选择 drama-3 模型自动打开）。

这个模块负责三件事：

1. 认出哪些方括号是情绪标签（不误伤 ``[CQ:at,...]``、``[1]``、``arr[i]`` 之类）；
2. 送去合成前整理标签（全角括号换成半角、限制数量；成对标签换成占位符保护起来，
   免得被 Markdown 清理吃掉 ``<>``）；
3. 不转语音、发文字时把标签去掉（代码块和行内代码里的标签不动）。
"""

import re
from dataclasses import dataclass

# ---------- Fish Audio 文档里列出的标签（小写） ----------
_BASIC = (
    "happy sad angry excited calm nervous confident surprised satisfied delighted "
    "scared worried upset frustrated depressed empathetic embarrassed disgusted moved "
    "proud relaxed grateful curious sarcastic"
).split()
_ADVANCED = (
    "disdainful unhappy anxious hysterical indifferent uncertain doubtful confused "
    "disappointed regretful guilty ashamed jealous envious hopeful optimistic pessimistic "
    "nostalgic lonely bored contemptuous sympathetic compassionate determined resigned"
).split()
_TONES = ("in a hurry tone", "shouting", "screaming", "whispering", "soft tone", "emphasis")
_EFFECTS = (
    "laughing", "chuckling", "sobbing", "crying loudly", "sighing", "groaning",
    "panting", "gasping", "yawning", "snoring", "clear throat",
)
_SPECIAL = ("audience laughing", "background laughter", "crowd laughing", "break", "long-break")
# 文档示例里出现过的其他写法
_SEEN = (
    "friendly", "narrator", "mysterious", "relieved", "enthusiastic", "encouraging",
    "urgent", "furious", "terrified", "ecstatic", "interested",
)
# 文档里的强度修饰：[slightly sad] [very excited] [extremely angry]
_MODIFIERS = ("slightly", "very", "extremely", "a bit", "somewhat")

KNOWN_TAGS = frozenset((*_BASIC, *_ADVANCED, *_TONES, *_EFFECTS, *_SPECIAL, *_SEEN))

# ---------- Drama 3 的成对尖括号标签 ----------
# Fish Audio 文档 "Change only specific words" 一节列出的完整集合，只有这 7 个。
# 它们只改被包住那几个词的读法，所以必须成对出现：<soft>慢慢来。</soft>
PAIRED_TAGS = frozenset((
    "whisper", "emphasis", "soft", "fast", "slow", "stress", "forceful",
))

MODES = ("any", "whitelist")

# 方括号（半角或全角）里 1~40 个字符，不含嵌套括号和换行。
# 前面不能紧跟字母/数字/下划线/右圆括号（排除 arr[i]、f(x)[0]），后面不能紧跟 "("（排除 Markdown 链接）。
# 前面紧跟 "]" 的情况在 _tags() 里单独判断：只有它前面是上一个标签时才算（[sad][whispering]），m[i][j] 不算。
_TAG = re.compile(r"(?<![A-Za-z0-9_\)])[\[［]([^\[\]［］\r\n]{1,40})[\]］](?!\()")
_HAS_LETTER = re.compile(r"[^\W\d_]")  # 至少有一个字母/汉字，排除 [1]、[2-3]、[...]
_CODE_SPLIT = re.compile(r"(```.*?```|`[^`\n]+`)", re.DOTALL)

# 单个尖括号标签（开或闭都匹配）。只认 PAIRED_TAGS 里的名字，大小写不敏感。
_ANY_PAIRED = re.compile(
    r"<\s*(/)?\s*(" + "|".join(sorted(PAIRED_TAGS, key=len, reverse=True)) + r")\s*>",
    re.IGNORECASE,
)
# 占位符：合成前把成对标签换成它，避开 markdown 清理（_MD_MARK 会吃掉 ">"）。
# 用控制字符，既不会被 \w 匹配，也不会被 URL / Emoji / 空白规则碰到。
_MASK = "\x00paired{}\x00"
_MASK_RE = re.compile(r"\x00paired(\d+)\x00")


@dataclass(frozen=True)
class EmotionRules:
    mode: str = "any"  # any：任何短方括号都当标签（S2 支持自然语言描述）；whitelist：只认已知标签
    extra: frozenset = frozenset()  # 白名单模式下额外认可的标签（小写）
    max_tags: int = 6  # 一条语音最多保留几个标签；0 = 不限制
    paired: bool = False  # 是否识别 drama-3 的成对尖括号标签（<whisper>…</whisper>）


def make_rules(mode: str = "any", extra=(), max_tags: int = 6, paired: bool = False) -> EmotionRules:
    mode = mode if mode in MODES else "any"
    cleaned = set()
    for item in extra or ():
        tag = str(item).strip().strip("[]［］").strip().lower()
        if tag:
            cleaned.add(" ".join(tag.split()))
    try:
        limit = max(0, int(max_tags))
    except (TypeError, ValueError):
        limit = 6
    return EmotionRules(
        mode=mode, extra=frozenset(cleaned), max_tags=limit, paired=bool(paired)
    )


def is_tag(inner: str, rules: EmotionRules) -> bool:
    """方括号里的内容是不是情绪标签。"""
    text = " ".join((inner or "").split())
    if not text or not _HAS_LETTER.search(text):
        return False
    low = text.lower()
    if low.startswith(("cq:", "at:")) or "://" in low or low.startswith("www."):
        return False
    if rules.mode != "whitelist":
        return True
    known = KNOWN_TAGS | rules.extra  # 只算一次，别在循环里反复合并
    if low in known:
        return True
    return any(low.startswith(mod + " ") and low[len(mod) + 1 :] in known for mod in _MODIFIERS)


def _tags(text: str, rules: EmotionRules):
    prev_end = -1  # 上一个被认出的标签的结尾位置
    for m in _TAG.finditer(text):
        if m.start() > 0 and text[m.start() - 1] in "]］" and m.start() != prev_end:
            continue  # 紧跟在一个不是标签的 "]" 后面，比如 m[i][j]
        if is_tag(m.group(1), rules):
            prev_end = m.end()
            yield m


def count(text: str, rules: EmotionRules) -> int:
    return sum(1 for _ in _tags(text or "", rules))


def normalize(text: str, rules: EmotionRules) -> str:
    """送去合成的版本：标签保留，但统一成半角 [xxx]，超过 max_tags 的多余标签丢掉。"""
    text = text or ""
    out, pos, kept = [], 0, 0
    for m in _tags(text, rules):
        out.append(text[pos : m.start()])
        if rules.max_tags <= 0 or kept < rules.max_tags:
            kept += 1
            out.append("[" + " ".join(m.group(1).split()) + "]")
        pos = m.end()
    out.append(text[pos:])
    return "".join(out)


def strip(text: str, rules: EmotionRules) -> str:
    """把标签去掉。标签位于行首 / 空白之后时，连它后面的一个空格一起去掉，避免留下多余空格。"""
    text = text or ""
    out, pos, last = [], 0, ""
    for m in _tags(text, rules):
        seg = text[pos : m.start()]
        if seg:
            out.append(seg)
            last = seg[-1]
        end = m.end()
        if (not last or last.isspace()) and end < len(text) and text[end] in " \t":
            end += 1
        pos = end
    out.append(text[pos:])
    return "".join(out)


def strip_outside_code(text: str, rules: EmotionRules) -> str:
    """发文字时用：代码块和行内代码里的标签保持原样，其余地方去掉标签。

    方括号标签和成对尖括号标签（rules.paired 为真时）都会去掉。
    两种标签都在同一个切分里处理，保证代码块内的都不动。
    """
    text = text or ""
    if not text:
        return text
    wants_brackets = rules is not None and ("[" in text or "［" in text)
    wants_paired = rules is not None and rules.paired and "<" in text
    if not wants_brackets and not wants_paired:
        return text
    parts = _CODE_SPLIT.split(text)
    for i in range(0, len(parts), 2):  # 偶数位是普通文字，奇数位是代码
        segment = parts[i]
        if wants_brackets:
            segment = strip(segment, rules)
        if wants_paired:
            segment = strip_paired(segment, rules)
        parts[i] = segment
    return "".join(parts)


# ---------- 成对尖括号标签（Drama 3） ----------
def count_paired(text: str, rules: "EmotionRules | None") -> int:
    """成对尖括号标签有几个（开闭各算一个）。未开启 paired 时恒为 0。"""
    if rules is None or not rules.paired or not text:
        return 0
    return sum(1 for _ in _ANY_PAIRED.finditer(text))


def mask_paired(text: str, rules: "EmotionRules | None") -> tuple:
    """合成前把成对标签换成占位符，返回 (替换后的文本, 原始标签列表)。

    为什么需要这一步：``main._prepare()`` 里的 ``_MD_MARK`` 会去掉 ``<>``，如果不管，
    ``<soft>慢慢来。</soft>`` 到了 Fish 那边就只剩 ``soft慢慢来。/soft``，指令失效。
    换成不含特殊字符的占位符后，markdown 清理、去 Emoji、去 @ 都不会碰它，
    `unmask_paired()` 在合成前一刻还原。

    代码块里的尖括号不动（和方括号标签的规则一致）。
    """
    text = text or ""
    if rules is None or not rules.paired or "<" not in text:
        return text, []
    found: list = []

    def _swap(segment: str) -> str:
        def _repl(m: re.Match) -> str:
            found.append(m.group(0))
            return _MASK.format(len(found) - 1)

        return _ANY_PAIRED.sub(_repl, segment)

    parts = _CODE_SPLIT.split(text)
    for i in range(0, len(parts), 2):
        parts[i] = _swap(parts[i])
    return "".join(parts), found


def unmask_paired(text: str, tags) -> str:
    """把 `mask_paired()` 的占位符还原成原始标签。"""
    text = text or ""
    if not tags or "\x00" not in text:
        return text

    def _repl(m: re.Match) -> str:
        try:
            index = int(m.group(1))
        except (TypeError, ValueError):
            return m.group(0)
        return tags[index] if 0 <= index < len(tags) else ""

    return _MASK_RE.sub(_repl, text)


def strip_paired(text: str, rules: "EmotionRules | None") -> str:
    """把成对尖括号标签整个去掉（开标签、闭标签、包住的文字都保留文字本身）。"""
    if rules is None or not rules.paired or not text or "<" not in text:
        return text or ""
    return _ANY_PAIRED.sub("", text)


def has_unpaired(text: str, rules: "EmotionRules | None") -> bool:
    """有没有落单的成对标签（只有开没有闭，或者反过来）。

    drama-3 遇到落单标签不会报错，但被包住的效果会作用到整段甚至整条，属于要提醒的情况。
    """
    if rules is None or not rules.paired or not text:
        return False
    opened: dict = {}
    for m in _ANY_PAIRED.finditer(text):
        name = m.group(2).lower()
        is_close = bool(m.group(1))
        depth = opened.get(name, 0)
        if is_close:
            opened[name] = max(0, depth - 1)
        else:
            opened[name] = depth + 1
    return any(depth > 0 for depth in opened.values())
