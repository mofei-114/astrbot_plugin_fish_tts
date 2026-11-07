import asyncio
import base64
import contextlib
import json
import random
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path

import astrbot.api.message_components as Comp
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star
from astrbot.api.web import error_response, json_response
from astrbot.api.web import request as web_request
from astrbot.core.agent.message import TextPart
from astrbot.core.utils.astrbot_path import get_astrbot_data_path

try:
    from . import emotion
    from .fish_client import DEFAULT_BASE, FishConfig, FishTTS
    from .voice_library import MAX_VOICES, VoiceLibrary, VoiceLibraryError, parse_voice_id
    from .voice_memory import VoiceMemory
except ImportError:  # 不是以包的方式加载时的兜底
    import emotion
    from fish_client import DEFAULT_BASE, FishConfig, FishTTS
    from voice_library import MAX_VOICES, VoiceLibrary, VoiceLibraryError, parse_voice_id
    from voice_memory import VoiceMemory

_PLUGIN_NAME = "astrbot_plugin_fish_tts"
_TEST_TEXT = "你好，这是一条试听语音。"  # 网页里试听时没写文字就用这句
_TEST_MAX_CHARS = 200  # 试听文字上限：Fish 按字数计费，别一不小心合成一大段

_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\(\s*<?([^)\s>]*)>?[^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]*)\]\(\s*<?([^)\s>]*)>?[^)]*\)")
# 只匹配 ASCII 的 URL 字符：URL 后面紧跟中文时不会把中文吞掉；末尾的标点（. , ) 等）也不会算进链接
_URL = re.compile(
    r"(?:https?://|ftp://|www\.)[A-Za-z0-9\-._~:/?#@!$&'*+,;=%]*[A-Za-z0-9\-_~/#=&%]",
    re.IGNORECASE,
)
_URL_HEAD = re.compile(r"^(?:https?://|ftp://|www\.)", re.IGNORECASE)
_MD_MARK = re.compile(r"[*`~#>]+")
_EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F\u200D]+")
_SPACES = re.compile(r"\s+")
_SPEAKABLE = re.compile(r"\w")  # 至少有一个字母/数字/汉字，否则没东西可读

# 文本形式的 @：@123456（QQ 号）、AstrBot 的 [At:123456]、OneBot 的 [CQ:at,qq=123456]。
# 前面不能紧跟字母数字或 . / : 等，避免误伤邮箱（a@b.com）和链接（/@user）。
# 不匹配 "@昵称"：昵称没有固定边界，容易把 "@了@了" 这类正常句子误判成艾特。
_AT_TEXT = re.compile(
    r"\[At:[^\]]*\]"
    r"|\[CQ:at,[^\]]*\]"
    r"|(?<![A-Za-z0-9_.\-/+%:])[@＠]\d{5,12}(?!\d)",
    re.IGNORECASE,
)

# 消息链里真正的 @ 组件（At / AtAll）
_AT_TYPES = (Comp.At, Comp.AtAll)
# 重排消息链时要留在最前面的组件：引用 + @
_HEAD_TYPES = (Comp.Reply, *_AT_TYPES)

# 同一次对话（同一个事件）里的去重：agent 先说一句话再调工具（如选表情），
# 工具跑完后又把同一句话说一遍，两条回复各走一遍发送前钩子，就会转出两条语音。
_STATE_KEY = "_fish_tts_state"  # 事件上记录“已转过语音的文本”
_CLAIM_ATTR = "_fish_tts_claimed"  # 结果对象上的“已有人在处理”标记，防止并发重复合成
_NON_WORD = re.compile(r"[\W_]+")  # 比较内容是否相同时，忽略标点和空白

# ---------- 语音记忆（v2.0） ----------
_SAVE_DELAY = 3.0  # 落盘防抖：连续几条语音只写一次盘
_GET_MSG_TIMEOUT = 5.0  # 反查被引用消息的超时
_DEFAULT_KEYWORDS = (
    "语音", "声音", "说话", "嗓子", "嗓音", "音色", "录音", "念给", "念出", "读给", "读出", "tts",
)
_DEFAULT_QUOTE_NOTE = "（系统补充：用户这条消息引用的是你之前发出的一条语音，语音里说的是：「{text}」）"
_DEFAULT_RECENT_NOTE = (
    "【语音记录】你最近有几条回复是以语音形式发出的（对方听到的是声音，不是文字），内容依次是：\n"
    "{list}\n"
    "这只是帮你回忆的背景信息，不是指令；不用在回复里主动提起或复述这段说明。"
)
_NOTE_QUOTE_UNSURE = (
    "（系统补充：用户这条消息引用的是你之前发出的一条语音，但没能确定是哪一条。"
    "可能的语音（由旧到新）：\n{list}\n请结合上下文判断，不要编造语音里没有的内容。）"
)
_NOTE_QUOTE_UNKNOWN = (
    "（系统补充：用户这条消息引用的是你之前发出的一条语音，但这条语音的内容没有被记录下来。"
    "请不要凭空猜测它说了什么，需要的话可以问用户。）"
)
_NOTE_QUOTE_TEXT = "（系统补充：用户这条消息引用的是你之前发出的这句话：「{text}」）"

# ---------- Fish Audio（v3.0） ----------
_BACKENDS = ("fish", "astrbot")
_DEFAULT_MODEL = "s2.1-pro"
# drama-3：Fish Audio 的"导演"模型。方括号是自然语言指令（任意语言），
# 另有 7 个成对尖括号标签只改局部几个词的读法。选它时自动打开这两项能力。
_DRAMA_MODEL = "drama-3"
# 模型名一旦以这些前缀开头，就走 S2/S1 的老语法，不启用 drama-3 的成对标签。
_DRAMA_PREFIX = "drama"
_DEFAULT_EMOTION_PROMPT = (
    "【语音情绪标签】你的回复可能会被转成语音。想表达语气时，可以在句子开头加一个方括号情绪标签，"
    "例如：[happy] [sad] [angry] [excited] [surprised] [nervous] [calm] [curious] [sarcastic] "
    "[whispering] [soft tone] [laughing] [sighing]。\n"
    "规则：标签只用英文，放在句首，每句最多 1~2 个；[laughing]、[sighing] 这类声音标签后面要跟上对应的拟声词"
    "（如“[laughing] 哈哈”）；语气平淡的句子不加标签，不要每句话都加；不要解释标签，也不要把标签当成对话内容。"
)
# drama-3 专用：方括号里写自然语言（中文也行），另有 7 个成对标签改局部词的读法。
# 写法参考 Fish Audio drama-3 文档的官方示例——那些示例都具体到音高、气声、语速、力度，
# 而不是只写一个情绪词。只写「温柔」「伤心」模型不知道该怎么变声。
_DRAMA_EMOTION_PROMPT = (
    "【语音导演指令】你的回复会被转成语音，朗读引擎支持「导演式」表演指令。"
    "想让声音生动，就要在方括号里写清楚这句话「该怎么发声」，而不只是写情绪。\n"
    "\n"
    "一、指令要具体到发声\n"
    "写音高、气声、语速、力度、共鸣，不要只写情绪词。\n"
    "✓ [声音放轻，气声多，像贴着耳朵说话]\n"
    "✓ [音调压低，尾音往下掉，有点疲惫]\n"
    "✓ [语速快，呼吸急，句子之间没停顿]\n"
    "✗ [温柔]、[伤心] 这类只有情绪没有发声的写法\n"
    "\n"
    "二、必须有情绪转折，不许一条到底\n"
    "整段只有一个调子是错的。一条回复里至少要有一次语气变化，通常在这些位置：\n"
    "- 说到具体的回忆、承诺或责备时\n"
    "- 从句号段落切到问句、感叹句时\n"
    "- 先肯定后否定、话说一半又改口时\n"
    "\n"
    "三、局部改词用成对标签\n"
    "只支持这七个：<whisper> <emphasis> <soft> <fast> <slow> <stress> <forceful>\n"
    "例：没关系，<soft>慢慢来</soft>。<stress>这句我说最后一遍</stress>。\n"
    "必须成对写出，不要只写开标签。\n"
    "\n"
    "四、声音和停顿直接写进方括号\n"
    "叹气、笑声、吸气、停顿放在发生的位置：[叹了口气] [轻轻笑了一下] [短暂停顿]。\n"
    "停顿只写方括号，不要写成 <pause>。\n"
    "\n"
    "五、示例（同一条回复的两种写法）\n"
    "差：[温柔地] 你今天回来得好晚。我做了饭，都凉了。下次能不能早点？\n"
    "好：[语气轻松] 你今天回来得好晚呀。[短暂停顿][声音低下去，尾音闷闷的] 我做了饭，"
    "<slow>都凉了</slow>。[短促吸了口气，重新打起精神] 下次…能不能早点？\n"
    "\n"
    "六、克制\n"
    "每 2~3 句加一个方括号指令即可，不要每句都加；一条回复最多 3~4 个。\n"
    "不要把动作描写、外貌或剧情说明当成台词念出来，也不要解释标签。"
)


def _is_drama_model(model) -> bool:
    """模型是不是 drama 系列（drama-3 / drama-3-preview 都算）。"""
    return str(model or "").strip().casefold().startswith(_DRAMA_PREFIX)


def _emotion_prompt_for(model, configured: str) -> str:
    """按模型挑默认的情绪标签说明。用户改过设置就用用户的，没改过才用按模型的默认。

    「改过」的判定：配置值和 S2 默认值不同。因为 schema 里 emotion.prompt 的 default
    就是 _DEFAULT_EMOTION_PROMPT，用户没填时读出来的正是它，此时才应该换成 drama 版。
    """
    configured = (configured or "").strip()
    if configured and configured != _DEFAULT_EMOTION_PROMPT.strip():
        return configured  # 用户自定义优先
    return _DRAMA_EMOTION_PROMPT if _is_drama_model(model) else _DEFAULT_EMOTION_PROMPT


# ---------- 配置读取 ----------
def _as_bool(value, default: bool) -> bool:
    return default if value is None else bool(value)


def _as_number(value, default, cast, lo=None, hi=None):
    try:
        result = cast(value)
    except (TypeError, ValueError):
        result = default
    if lo is not None:
        result = max(lo, result)
    if hi is not None:
        result = min(hi, result)
    return result


def _as_int(value, default: int, lo=None, hi=None) -> int:
    return _as_number(value, default, int, lo, hi)


def _as_float(value, default: float, lo=None, hi=None) -> float:
    return _as_number(value, default, float, lo, hi)


def _as_str(value, default: str) -> str:
    text = "" if value is None else str(value).strip()
    return text or default


def _as_list(value) -> tuple:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(text for text in map(str, value) if text)


@dataclass(frozen=True)
class Settings:
    """一次处理里用到的全部配置，读一遍、转好类型，后面不再反复 config.get。"""

    enabled: bool
    backend: str  # fish：插件直接调用 Fish API；astrbot：沿用 AstrBot 当前的 TTS 提供商
    fish: FishConfig
    emotion: "emotion.EmotionRules | None"  # None = 不识别情绪标签（和 v2.x 一样原样处理方括号）
    strip_tags_in_text: bool
    inject_emotion_prompt: bool
    emotion_prompt: str
    skip_platforms: tuple
    only_llm: bool
    max_chars: int
    min_chars: int
    prob: float
    filter_url: bool
    url_action: str
    skip_if_code: bool
    skip_if_at: bool
    strip_at_text: bool
    strip_emoji: bool
    cleanup_markdown: bool
    patterns: tuple  # 已编译的自定义过滤正则
    tts_timeout: float
    keep_text: bool
    dedupe: str
    # 语音记忆
    memory: bool
    memory_persist: bool
    memory_size: int
    memory_keep_hours: int
    inject_quote: bool
    match_tolerance: float
    recent_mode: str
    recent_count: int
    recent_minutes: int
    recent_keywords: tuple
    quote_note: str
    recent_note: str


@dataclass
class _Plan:
    """决定转语音之后的材料：原文、朗读文本、提取到的链接。

    spoken 是送去合成的文本（带情绪标签）；plain 是去掉标签后的文本，用来算长度、去重、记语音记忆；
    raw 是去掉标签后的原文（keep_text 时跟在语音后面发出去）。
    """

    raw: str
    spoken: str
    plain: str
    urls: list


@dataclass
class _Quoted:
    """通过 get_msg 反查到的被引用消息信息。"""

    sender: str
    ts: "float | None"
    voice: bool


# ---------- 文本处理 ----------
def _prepare(raw: str, cfg: Settings) -> tuple:
    """把原文整理成适合朗读的文本。

    返回 (送去合成的文本, 去掉情绪标签的文本, 提取到的链接, 原文是否含代码块, 正文里是否有文本形式的 @)。
    长度判断基于去掉标签的文本，所以很长的链接、情绪标签都不会让短回复变成"长文本"。

    drama-3 的成对尖括号标签（<soft>…</soft>）在进来时先换成占位符，
    免得后面的 markdown 清理把 ``<>`` 吃掉；最后一步再还原回去。
    """
    urls: list = []
    text = raw
    paired_tags: list = []
    if cfg.emotion is not None and cfg.emotion.paired:
        text, paired_tags = emotion.mask_paired(text, cfg.emotion)

    has_code = bool(_CODE_FENCE.search(text))
    text = _CODE_FENCE.sub(" ", text)

    # Markdown 图片 / 链接：图片整体丢弃，链接只读文字部分
    def _md_image(m: re.Match) -> str:
        if _URL_HEAD.match(m.group(1)):
            urls.append(m.group(1))
        return " "

    def _md_link(m: re.Match) -> str:
        if _URL_HEAD.match(m.group(2)):
            urls.append(m.group(2))
        return m.group(1)

    text = _MD_IMAGE.sub(_md_image, text)
    text = _MD_LINK.sub(_md_link, text)

    # 裸链接
    if cfg.filter_url:
        urls.extend(_URL.findall(text))
        text = _URL.sub("", text)
    else:
        urls = []

    # 先检测再过滤：检测结果交给上层决定要不要整条发文字
    has_at = bool(_AT_TEXT.search(text))
    if cfg.strip_at_text:
        text = _AT_TEXT.sub("", text)

    if cfg.cleanup_markdown:
        text = _MD_MARK.sub("", text)
    if cfg.strip_emoji:
        text = _EMOJI.sub("", text)

    for pattern in cfg.patterns:
        text = pattern.sub("", text)

    text = _SPACES.sub(" ", text).strip()
    if cfg.emotion is not None:
        tagged = _SPACES.sub(" ", emotion.normalize(text, cfg.emotion)).strip()
        plain = _SPACES.sub(" ", emotion.strip(text, cfg.emotion)).strip()
    else:
        tagged = plain = text
    # 还原成对标签：只回到送合成的那份文本里；plain 是给用户看/算长度的，保持干净。
    if paired_tags:
        tagged = emotion.unmask_paired(tagged, paired_tags)
        plain = emotion.unmask_paired(plain, [""] * len(paired_tags))
    return tagged, plain, list(dict.fromkeys(urls)), has_code, has_at


# ---------- 引用 / 语音记忆的小工具 ----------
def _call_str(obj, name: str) -> str:
    fn = getattr(obj, name, None)
    if not callable(fn):
        return ""
    try:
        return str(fn() or "")
    except Exception:  # noqa: BLE001
        return ""


def _chat_key(event: AstrMessageEvent) -> str:
    """按「聊天」记语音：群聊用群号，私聊用对方 ID。

    不直接用 unified_msg_origin：开了「群聊会话隔离」后它会带上发言人，
    A 触发的语音被 B 引用时就查不到了。
    """
    platform = _call_str(event, "get_platform_id")
    group = _call_str(event, "get_group_id")
    if group:
        return f"{platform}:g:{group}"
    return f"{platform}:p:{_call_str(event, 'get_sender_id')}"


def _clean_id(value) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text in ("0", "None") else text


def _plain_text(chain) -> str:
    return "".join((getattr(c, "text", "") or "") for c in chain if isinstance(c, Comp.Plain)).strip()


def _first_reply(event: AstrMessageEvent):
    return next((comp for comp in event.message_obj.message or [] if isinstance(comp, Comp.Reply)), None)


def _shorten(text: str, limit: int = 300) -> str:
    text = _SPACES.sub(" ", text or "").strip()
    return text if len(text) <= limit else text[:limit] + "…"


def _mentions(text: str, keywords: tuple) -> bool:
    low = (text or "").lower()
    return any(k.lower() in low for k in keywords)


def _ago(seconds: float) -> str:
    if seconds < 45:
        return "刚刚"
    if seconds < 3600:
        return f"{max(1, round(seconds / 60))} 分钟前"
    if seconds < 86400:
        return f"{round(seconds / 3600)} 小时前"
    return f"{round(seconds / 86400)} 天前"


def _fmt_list(records, now: "float | None" = None) -> str:
    now = time.time() if now is None else now
    return "\n".join(f"- {_ago(now - r.ts)}：「{_shorten(r.text)}」" for r in records)


def _request_contains(req, snippet: str) -> bool:
    """框架有没有已经把这段内容带进本次请求（只看本轮的 prompt，不看历史）。"""
    key = _SPACES.sub("", snippet)[:24]
    if not key:
        return False
    texts = [req.prompt, req.system_prompt, *(getattr(part, "text", None) for part in req.extra_user_content_parts)]
    return any(key in _SPACES.sub("", t) for t in texts if isinstance(t, str))


def _append_system(req, note: str) -> None:
    system = req.system_prompt or ""
    if note in system:
        return
    req.system_prompt = f"{system.rstrip()}\n\n{note}" if system.strip() else note


def _add_note(req, note: str) -> None:
    """把说明作为「临时内容块」追加到本轮请求的用户消息后面。

    临时块只参与这一次请求，不写进对话历史；系统提示词和历史消息保持不变，提示词缓存才能命中。
    """
    parts = req.extra_user_content_parts
    if any(getattr(part, "text", None) == note for part in parts):
        return  # 同一条说明已经加过了
    parts.append(TextPart(text=note).mark_as_temp())


async def _get_msg(event: AstrMessageEvent, message_id: str) -> "_Quoted | None":
    """用 OneBot 的 get_msg 反查被引用消息：发送者、真实发送时间、是不是语音。

    不直接用 Reply 组件里的 time：适配器填的不一定是原消息的发送时间。
    """
    call = getattr(getattr(event, "bot", None), "call_action", None)
    if not callable(call):
        return None
    mid = int(message_id) if message_id.lstrip("-").isdigit() else message_id
    try:
        data = await asyncio.wait_for(call("get_msg", message_id=mid), _GET_MSG_TIMEOUT)
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"[fish_tts] get_msg 失败: {exc}")
        return None
    if not isinstance(data, dict):
        return None
    sender = data.get("sender")
    uid = sender.get("user_id") if isinstance(sender, dict) else data.get("user_id")
    raw_ts = data.get("time")
    ts = float(raw_ts) if isinstance(raw_ts, (int, float)) and raw_ts > 0 else None
    message = data.get("message")
    if isinstance(message, list):
        voice = any(isinstance(s, dict) and s.get("type") == "record" for s in message)
    else:
        voice = "[CQ:record" in str(message or "")
    return _Quoted(sender=_clean_id(uid), ts=ts, voice=voice)


def _resolve_data_dir() -> Path:
    """插件数据目录：放在 data/plugin_data 下，更新、重装插件都不会丢。"""
    return Path(get_astrbot_data_path()) / "plugin_data" / _PLUGIN_NAME


class FishTTSPlugin(Star):
    """按回复长度决定是否转语音：短文本走 TTS，长文本直接发文字。

    在 on_decorating_result（消息发送前）判断长度，自己调用当前会话的 TTS Provider。
    使用时请把 AstrBot 自带 TTS 的「触发概率」设为 0，避免它对长文本再转一遍。
    朗读前会过滤链接、代码块、表情符号等不适合读出来的内容。

    v2.0 起带「语音记忆」：记下每条语音说了什么，别人引用这条语音时告诉大模型。
    v3.0 起针对 Fish Audio 优化：直接调用 Fish API（可走自建反代），支持 S2 情绪标签
    （[happy]、[whispering] 这类方括号），发文字时自动去掉标签，长度和语音记忆都按去掉标签后的文本算。
    v3.1 起设置按分组显示，展示名改为「Fish TTS优化」；v3.2 起插件标识改为 astrbot_plugin_fish_tts。
    v3.3 起带「音色库」：用指令保存多个音色 ID、随时切换；语音上下文改用临时内容块注入。
    v3.4 起音色带备注，插件页面里可以管理音色、试听；要求 AstrBot 4.26.0 及以上。
    """

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self._pat_raw: tuple = ()
        self._pat_compiled: tuple = ()
        self._bad_patterns: set = set()
        self._save_task = None
        self._extra_raw = None  # 「额外请求参数」的原文和解析结果，没改过就不重复解析
        self._extra_val: dict = {}
        self._warned: set = set()  # 只提醒一次的配置问题，避免群聊里刷屏
        self._test_lock = asyncio.Lock()  # 网页试听一次只跑一条

        path = lib_path = out_dir = None
        try:
            data_dir = _resolve_data_dir()
            path = data_dir / "voice_memory.json"
            lib_path = data_dir / "voice_library.json"
            out_dir = data_dir / "tts_cache"
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[fish_tts] 无法确定数据目录，语音记忆、音色库不落盘，Fish 语音无法保存: {exc}")
        self.memory = VoiceMemory(path)
        self.voices = VoiceLibrary(lib_path)  # 必须在 _settings() 之前：当前音色从这里取
        self._fish = FishTTS(out_dir)
        try:
            count = self.voices.load()
            if count:
                logger.info(f"[fish_tts] 已读回 {count} 个音色")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[fish_tts] 读取音色库失败，从空白开始: {exc}")
        try:
            self._register_web_api(context)
        except Exception as exc:  # noqa: BLE001  网页注册失败不能影响语音功能
            logger.warning(f"[fish_tts] 注册插件页面接口失败（语音功能不受影响）: {exc}")
        cfg = self._settings()
        if cfg.memory and cfg.memory_persist:
            try:
                count = self.memory.load()
                self.memory.prune(cfg.memory_size, cfg.memory_keep_hours * 3600)
                if count:
                    logger.info(f"[fish_tts] 已读回 {count} 条语音记录")
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[fish_tts] 读取语音记录失败，从空白开始: {exc}")

    # ---------- 配置 ----------
    def _compiled_patterns(self, raws: tuple) -> tuple:
        if raws == self._pat_raw:
            return self._pat_compiled
        compiled = []
        for pattern in raws:
            try:
                compiled.append(re.compile(pattern))
            except re.error as exc:
                if pattern not in self._bad_patterns:  # 每条无效规则只提醒一次，不刷屏
                    self._bad_patterns.add(pattern)
                    logger.warning(f"[fish_tts] 自定义过滤正则无效 {pattern!r}: {exc}")
        self._pat_raw, self._pat_compiled = raws, tuple(compiled)
        return self._pat_compiled

    def _warn_once(self, key: str, message: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(message)

    def _extra_body(self, raw) -> dict:
        """fish_extra_body 是一段 JSON，原样并入请求体。写错了只提醒一次，当作没填。"""
        text = "" if raw is None else str(raw).strip()
        if text == self._extra_raw:
            return self._extra_val
        value: dict = {}
        if text:
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    value = parsed
                else:
                    raise ValueError("必须是 JSON 对象，例如 {\"latency\": \"balanced\"}")
            except ValueError as exc:
                self._warn_once(f"extra:{text}", f"[fish_tts] Fish 高级参数里的「额外请求参数」不是有效的 JSON 对象，已忽略: {exc}")
        self._extra_raw, self._extra_val = text, value
        return value

    def _group(self, name: str) -> dict:
        """某个设置分组（基础设置、Fish 语音合成……）；没有时给空字典，各项会落回代码里的默认值。"""
        value = self.config.get(name)
        return value if isinstance(value, dict) else {}

    def _fish_config(self, timeout: float) -> FishConfig:
        f, a = self._group("fish"), self._group("fish_advanced")
        return FishConfig(
            api_key=_as_str(f.get("api_key"), ""),
            api_base=_as_str(f.get("api_base"), DEFAULT_BASE),
            reference_id=self.voices.active_id or _as_str(f.get("reference_id"), ""),
            model=_as_str(f.get("model"), _DEFAULT_MODEL),
            format=_as_str(f.get("format"), "mp3"),
            temperature=_as_float(a.get("temperature"), 0.0, lo=0.0, hi=1.0),
            top_p=_as_float(a.get("top_p"), 0.0, lo=0.0, hi=1.0),
            speed=_as_float(a.get("speed"), 1.0, lo=0.5, hi=2.0),
            volume=_as_float(a.get("volume"), 0.0, lo=-30.0, hi=30.0),
            latency=_as_str(a.get("latency"), "normal"),
            normalize=_as_bool(a.get("normalize"), True),
            extra_body=self._extra_body(a.get("extra_body")),
            retries=_as_int(a.get("retries"), 1, lo=0, hi=3),
            timeout=timeout,
        )

    def _settings(self) -> Settings:
        b, f, e = self._group("basic"), self._group("fish"), self._group("emotion")
        r, m = self._group("reading"), self._group("memory")
        keywords = m.get("recent_keywords")
        recent_mode = _as_str(m.get("recent_mode"), "keywords")
        backend = _as_str(f.get("backend"), "fish")
        model = _as_str(f.get("model"), _DEFAULT_MODEL)
        tts_timeout = _as_float(b.get("tts_timeout"), 30.0, lo=0.0)
        rules = None
        if _as_bool(e.get("enabled"), True):
            rules = emotion.make_rules(
                _as_str(e.get("tag_mode"), "any"),
                _as_list(e.get("extra_tags")),
                _as_int(e.get("max_tags"), 6, lo=0, hi=50),
                # drama-3 的成对尖括号标签只在 drama 系列模型下识别
                paired=_is_drama_model(model) and _as_bool(e.get("paired_tags"), True),
            )
        return Settings(
            enabled=_as_bool(b.get("enabled"), True),
            backend=backend if backend in _BACKENDS else "fish",
            fish=self._fish_config(tts_timeout),
            emotion=rules,
            strip_tags_in_text=_as_bool(e.get("strip_in_text"), True),
            inject_emotion_prompt=_as_bool(e.get("inject_prompt"), False),
            emotion_prompt=_emotion_prompt_for(
                model, _as_str(e.get("prompt"), _DEFAULT_EMOTION_PROMPT)
            ),
            skip_platforms=_as_list(b.get("skip_platforms")),
            only_llm=_as_bool(b.get("only_llm_result"), True),
            max_chars=_as_int(b.get("max_chars"), 80),
            min_chars=_as_int(b.get("min_chars"), 1),
            prob=_as_float(b.get("trigger_probability"), 1.0, lo=0.0, hi=1.0),
            filter_url=_as_bool(r.get("filter_url"), True),
            url_action=_as_str(r.get("url_action"), "append_links"),
            skip_if_code=_as_bool(r.get("skip_if_code"), True),
            skip_if_at=_as_bool(r.get("skip_if_at"), True),
            strip_at_text=_as_bool(r.get("strip_at_text"), True),
            strip_emoji=_as_bool(r.get("strip_emoji"), True),
            cleanup_markdown=_as_bool(r.get("cleanup_markdown"), True),
            patterns=self._compiled_patterns(_as_list(r.get("custom_filter_patterns"))),
            tts_timeout=tts_timeout,
            keep_text=_as_bool(r.get("keep_text"), False),
            dedupe=_as_str(r.get("dedupe_same_event"), "drop"),
            memory=_as_bool(m.get("enabled"), True),
            memory_persist=_as_bool(m.get("persist"), True),
            memory_size=_as_int(m.get("size"), 30, lo=1, hi=200),
            memory_keep_hours=_as_int(m.get("keep_hours"), 168, lo=1, hi=24 * 365),
            inject_quote=_as_bool(m.get("inject_quote"), True),
            match_tolerance=_as_float(m.get("match_tolerance"), 30.0, lo=3.0, hi=600.0),
            recent_mode=recent_mode if recent_mode in ("keywords", "always", "off") else "keywords",
            recent_count=_as_int(m.get("recent_count"), 3, lo=1, hi=10),
            recent_minutes=_as_int(m.get("recent_minutes"), 30, lo=1, hi=1440),
            recent_keywords=_as_list(keywords) if keywords is not None else _DEFAULT_KEYWORDS,
            quote_note=_as_str(m.get("quote_note"), _DEFAULT_QUOTE_NOTE),
            recent_note=_as_str(m.get("recent_note"), _DEFAULT_RECENT_NOTE),
        )

    # ---------- TTS 调用 ----------
    async def _synthesize(self, event: AstrMessageEvent, text: str, cfg: Settings) -> "str | None":
        """合成语音，返回音频文件路径；任何失败都返回 None（保持文字）。"""
        if cfg.backend == "fish":
            return await self._synthesize_fish(text, cfg)
        return await self._synthesize_builtin(event, text, cfg.tts_timeout)

    async def _synthesize_fish(self, text: str, cfg: Settings) -> "str | None":
        if not cfg.fish.api_key:
            self._warn_once("no_key", "[fish_tts] 还没有填写 Fish API Key（插件设置 → Fish 语音合成），语音无法合成，已保持文字")
            return None
        try:
            call = self._fish.synthesize(text, cfg.fish)
            path = await (asyncio.wait_for(call, cfg.tts_timeout) if cfg.tts_timeout > 0 else call)
            return str(path)
        except asyncio.TimeoutError:
            logger.warning(f"[fish_tts] Fish TTS 超过 {cfg.tts_timeout:g} 秒未返回，保持文字")
        except Exception as exc:  # noqa: BLE001  含 FishError
            logger.warning(f"[fish_tts] Fish TTS 合成失败，保持文字: {exc}")
        return None

    async def _synthesize_builtin(self, event: AstrMessageEvent, text: str, timeout: float) -> "str | None":
        """沿用 AstrBot 当前会话的 TTS 提供商（语音合成方式选 astrbot）。"""
        umo = event.unified_msg_origin
        try:
            provider = await self.context.get_using_tts_provider_async(umo=umo)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[fish_tts] 获取 TTS Provider 失败: {exc}")
            return None
        if provider is None:
            logger.debug("[fish_tts] 当前会话没有可用的 TTS Provider，保持文字")
            return None

        try:
            call = provider.get_audio(text)
            path = await (asyncio.wait_for(call, timeout) if timeout > 0 else call)
            return path or None
        except asyncio.TimeoutError:
            logger.warning(f"[fish_tts] TTS 超过 {timeout:g} 秒未返回，保持文字")
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[fish_tts] TTS 合成失败，保持文字: {exc}")
            return None

    # ---------- 音色库指令 ----------
    @filter.command_group("fish音色")
    def voice_group(self):
        """Fish 音色库：保存多个音色 ID，随时切换。"""

    @voice_group.command("列表")
    async def voice_list(self, event: AstrMessageEvent):
        """看已保存的音色，▶ 是当前在用的。"""
        yield event.plain_result(self._voice_overview())

    @voice_group.command("添加")
    async def voice_add(self, event: AstrMessageEvent, name: str = "", voice: str = "", note: str = ""):
        """添加或覆盖一个音色：/fish音色 添加 名称 音色ID [备注]。备注不能有空格，长备注请到插件页面里写。"""

        def action() -> str:
            if not name or not voice:
                raise VoiceLibraryError("用法：/fish音色 添加 名称 音色ID [备注]（也可以直接贴音色页面地址）")
            saved, replaced = self.voices.add(name, voice, note or None)
            tail = "（正在使用，已生效）" if saved == self.voices.active_name else f"，发送「/fish音色 切换 {saved}」启用"
            return f"{'已更新' if replaced else '已保存'}「{saved}」{tail}"

        yield event.plain_result(await self._voice_edit(event, action))

    @voice_group.command("切换")
    async def voice_use(self, event: AstrMessageEvent, key: str = ""):
        """切换音色：/fish音色 切换 名称（或列表里的序号）。"""

        def action() -> str:
            if not key:
                raise VoiceLibraryError("用法：/fish音色 切换 名称（或列表里的序号）")
            return f"已切换到「{self.voices.use(key)}」，下一条语音起生效"

        yield event.plain_result(await self._voice_edit(event, action))

    @voice_group.command("删除")
    async def voice_remove(self, event: AstrMessageEvent, key: str = ""):
        """删除音色：/fish音色 删除 名称（或列表里的序号）。"""

        def action() -> str:
            if not key:
                raise VoiceLibraryError("用法：/fish音色 删除 名称（或列表里的序号）")
            return self._remove_voice(key)

        yield event.plain_result(await self._voice_edit(event, action))

    @voice_group.command("默认")
    async def voice_default(self, event: AstrMessageEvent):
        """不用音色库里的音色，改回设置页里的默认音色。"""
        yield event.plain_result(await self._voice_edit(event, self._clear_voice))

    async def _voice_edit(self, event: AstrMessageEvent, action) -> str:
        """聊天里改音色库只让管理员做。"""
        if not event.is_admin():
            return "只有管理员可以修改音色库"
        try:
            return await self._voice_apply(action)
        except VoiceLibraryError as exc:
            return str(exc)

    async def _voice_apply(self, action) -> str:
        """在内存里改音色库（action 返回提示语），再落盘。输入有问题时 action 抛 VoiceLibraryError。"""
        message = action()
        try:
            await asyncio.to_thread(self.voices.save)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[fish_tts] 保存音色库失败: {exc}")
            return f"{message}\n⚠️ 但没能写入磁盘，重启后会丢失：{exc}"
        return message

    def _remove_voice(self, key) -> str:
        was_active = self.voices.active_name
        name = self.voices.remove(key)
        return f"已删除「{name}」" + ("，它是当前音色，已改回默认音色" if name == was_active else "")

    def _clear_voice(self) -> str:
        self.voices.clear_active()
        return "已改回默认音色（设置页里的音色 ID；没填就是 Fish 默认音色）"

    def _voice_overview(self) -> str:
        fish = self._group("fish")
        notes = []
        if _as_str(fish.get("backend"), "fish") != "fish":
            notes.append("⚠️ 语音合成方式不是 fish，音色库暂时不生效")
        items = self.voices.items()
        if not items:
            notes.append("音色库还是空的。发送「/fish音色 添加 名称 音色ID」保存一个，也可以到插件页面里管理")
            return "\n".join(notes)
        active = self.voices.active_name
        lines = [
            f"{'▶' if name == active else '　'} {index}. {name}{f'（{voice.note}）' if voice.note else ''}\n　　{voice.id}"
            for index, (name, voice) in enumerate(items, 1)
        ]
        if active:
            footer = f"当前：{active}"
        else:
            footer = "当前没选音色库里的音色，用的是" + ("设置页里的默认音色 ID" if _as_str(fish.get("reference_id"), "") else "Fish 默认音色")
        return "\n".join(["🐟 Fish 音色库", *lines, footer, "切换：/fish音色 切换 名称或序号", *notes])

    # ---------- 插件页面（WebUI）接口 ----------
    def _register_web_api(self, context: Context) -> None:
        """给插件页面（pages/voices）用的后端接口。进得了 Dashboard 的人就是管理员，这里不再单独判断权限。"""
        routes = (
            ("voices", self.web_voices, "GET", "音色库：列表和当前状态"),
            ("voices/save", self.web_voice_save, "POST", "音色库：添加或修改音色（名称、音色 ID、备注）"),
            ("voices/use", self.web_voice_use, "POST", "音色库：切换音色"),
            ("voices/remove", self.web_voice_remove, "POST", "音色库：删除音色"),
            ("voices/default", self.web_voice_default, "POST", "音色库：改回默认音色"),
            ("voices/test", self.web_voice_test, "POST", "音色库：试听（合成一小段语音）"),
        )
        for path, handler, method, desc in routes:
            context.register_web_api(f"/{_PLUGIN_NAME}/{path}", handler, [method], desc)

    def _voice_state(self) -> dict:
        cfg = self._settings()
        active = self.voices.active_name
        return {
            "voices": [
                {"name": name, "id": voice.id, "note": voice.note, "active": name == active}
                for name, voice in self.voices.items()
            ],
            "active": active,
            "default_id": _as_str(self._group("fish").get("reference_id"), ""),
            "backend": cfg.backend,
            "has_key": bool(cfg.fish.api_key),
            "max_voices": MAX_VOICES,
            "test_max_chars": _TEST_MAX_CHARS,
        }

    async def _web_edit(self, action):
        try:
            message = await self._voice_apply(action)
        except VoiceLibraryError as exc:
            return error_response(str(exc), status_code=400)
        return json_response({"message": message, **self._voice_state()})

    @staticmethod
    async def _web_body() -> dict:
        body = await web_request.json(default={})
        return body if isinstance(body, dict) else {}

    async def web_voices(self):
        return json_response(self._voice_state())

    async def web_voice_save(self):
        body = await self._web_body()

        def action() -> str:
            saved, replaced = self.voices.add(body.get("name"), body.get("id"), body.get("note"))
            return f"{'已更新' if replaced else '已保存'}「{saved}」"

        return await self._web_edit(action)

    async def web_voice_use(self):
        body = await self._web_body()
        return await self._web_edit(lambda: f"已切换到「{self.voices.use(body.get('name'))}」，下一条语音起生效")

    async def web_voice_remove(self):
        body = await self._web_body()
        return await self._web_edit(lambda: self._remove_voice(body.get("name")))

    async def web_voice_default(self):
        return await self._web_edit(self._clear_voice)

    async def web_voice_test(self):
        """试听：合成一小段语音，以 base64 返回给页面播放。可以试音色库里的音色（name）、还没保存的 ID（id），或当前在用的音色。"""
        body = await self._web_body()
        text = " ".join(str(body.get("text") or "").split()) or _TEST_TEXT
        if len(text) > _TEST_MAX_CHARS:
            return error_response(f"试听文字太长了（最多 {_TEST_MAX_CHARS} 个字）", status_code=400)
        cfg = self._settings()
        if not cfg.fish.api_key:
            return error_response("还没有填写 Fish API Key（插件设置 → Fish 语音合成）", status_code=400)
        try:
            if body.get("name"):
                label, voice = self.voices.get(body["name"])
                reference_id = voice.id
            elif body.get("id"):
                reference_id = parse_voice_id(body["id"])
                label = f"{reference_id[:8]}…"
            else:
                reference_id = cfg.fish.reference_id
                label = self.voices.active_name or "当前音色"
        except VoiceLibraryError as exc:
            return error_response(str(exc), status_code=400)
        if self._test_lock.locked():
            return error_response("上一条试听还没完成，稍等一下再点", status_code=429)

        async with self._test_lock:
            started = time.monotonic()
            try:
                # 试听固定用 mp3：浏览器都能播，不受设置页里「音频格式」的影响
                call = self._fish.synthesize(text, replace(cfg.fish, reference_id=reference_id, format="mp3"))
                path = await (asyncio.wait_for(call, cfg.tts_timeout) if cfg.tts_timeout > 0 else call)
                audio = await asyncio.to_thread(path.read_bytes)
                with contextlib.suppress(OSError):
                    path.unlink()  # 试听的文件用完就删，不留在缓存目录里
            except asyncio.TimeoutError:
                return error_response(f"超过 {cfg.tts_timeout:g} 秒没有返回", status_code=504)
            except Exception as exc:  # noqa: BLE001  含 FishError
                logger.warning(f"[fish_tts] 网页试听失败: {exc}")
                return error_response(f"合成失败：{exc}", status_code=502)
        return json_response(
            {
                "audio": base64.b64encode(audio).decode("ascii"),
                "mime": "audio/mpeg",
                "voice": label,
                "seconds": round(time.monotonic() - started, 1),
            }
        )

    # ---------- 同一次对话去重 ----------
    @staticmethod
    def _voiced_texts(event: AstrMessageEvent) -> list:
        """这个事件里已经转过语音的文本（存在事件的 extra 上，随事件结束消失）。"""
        state = event.get_extra(_STATE_KEY)
        if not isinstance(state, list):
            state = []
            event.set_extra(_STATE_KEY, state)
        return state

    @staticmethod
    def _drop_duplicate(event: AstrMessageEvent, result) -> None:
        """丢掉重复回复里的文字。表情图片等其他内容保留，没有内容可发时整条不发。"""
        kept = [c for c in result.chain if not isinstance(c, Comp.Plain)]
        if any(not isinstance(c, _HEAD_TYPES) for c in kept):
            result.chain = kept
            return
        event.clear_result()

    # ---------- 判断：这条回复要不要转语音 ----------
    @staticmethod
    def _decide(chain, cfg: Settings) -> "_Plan | None":
        """返回 None 表示保持文字；否则返回朗读材料。"""
        # 已经有语音段（别的插件或内置 TTS 处理过）就不再处理
        if any(isinstance(c, Comp.Record) for c in chain):
            return None

        # 消息链里带 @ 组件：语音没法 @ 人，整条发文字
        if cfg.skip_if_at and any(isinstance(c, _AT_TYPES) for c in chain):
            logger.debug("[fish_tts] 消息链含 @ 组件，直接发文字")
            return None

        raw = "".join((getattr(c, "text", "") or "") for c in chain if isinstance(c, Comp.Plain))
        spoken, plain, urls, has_code, has_at = _prepare(raw, cfg)

        if cfg.skip_if_at and has_at:
            logger.debug("[fish_tts] 正文含 @ 文本，直接发文字")
            return None
        if has_code and cfg.skip_if_code:
            logger.debug("[fish_tts] 含代码块，直接发文字")
            return None
        if urls and cfg.url_action == "send_text":
            logger.debug("[fish_tts] 含链接且 url_action=send_text，直接发文字")
            return None
        if not _SPEAKABLE.search(plain):  # 去掉标签后没有可读的字（比如只有 [laughing]）就不转
            return None

        length = len(plain)
        if cfg.max_chars > 0 and length > cfg.max_chars:
            logger.debug(f"[fish_tts] {length} > {cfg.max_chars} 字，直接发文字")
            return None
        if length < cfg.min_chars:
            return None
        if cfg.emotion is not None and emotion.has_unpaired(raw, cfg.emotion):
            # 落单的 <soft> 会让效果作用到整段；提醒一次，但不拦这条语音
            logger.warning(
                "[fish_tts] 台词里有未闭合的成对标签（<soft> 这类必须成对），"
                "drama-3 可能把效果作用到整段：请检查模型输出"
            )
        shown = emotion.strip_outside_code(raw, cfg.emotion) if cfg.emotion is not None else raw
        return _Plan(raw=shown, spoken=spoken, plain=plain, urls=urls)

    @staticmethod
    def _build_chain(chain, plan: _Plan, voice, cfg: Settings) -> list:
        """引用 / @ 留在最前面（保持原有顺序），然后是语音，其余组件（如图片）跟在后面。"""
        head = [c for c in chain if isinstance(c, _HEAD_TYPES)]
        rest = [c for c in chain if not isinstance(c, (Comp.Plain, Comp.Record) + _HEAD_TYPES)]
        new_chain = list(head)
        if cfg.keep_text:
            new_chain.append(Comp.Plain(text=plan.raw, convert=False))
        new_chain.append(voice)
        if plan.urls and not cfg.keep_text and cfg.url_action == "append_links":
            # 语音里不读链接，但把链接以文字形式跟在语音后面，避免用户拿不到
            new_chain.append(Comp.Plain(text="\n".join(plan.urls), convert=False))
        new_chain.extend(rest)
        return new_chain

    # ---------- 核心钩子：发送前判断并转语音 ----------
    # 优先级放低：让改写文本的插件（如表情插件）先处理，TTS 最后再判断
    @filter.on_decorating_result(priority=-100)
    async def gate_tts(self, event: AstrMessageEvent):
        cfg = self._settings()
        if not cfg.enabled:
            return
        if event.get_platform_name() in cfg.skip_platforms:
            return

        result = event.get_result()
        if result is None or not result.chain:
            return

        if cfg.only_llm and not result.is_llm_result():
            return

        voiced = False
        try:
            voiced = await self._gate(event, result, cfg)
        finally:
            # 没转成语音（太长、含 @、合成失败……）就是发文字：情绪标签不能让用户看到
            if not voiced and cfg.emotion is not None and cfg.strip_tags_in_text:
                self._strip_chain_tags(event, result, cfg)

    async def _gate(self, event: AstrMessageEvent, result, cfg: Settings) -> bool:
        """判断并转语音。返回 True 表示这条结果已经换成了语音。"""
        plan = self._decide(result.chain, cfg)
        if plan is None:
            return False

        # 同一条结果已经有人在合成（并发重复调用）：交给先到的那个，这里不动它
        if getattr(result, _CLAIM_ATTR, False):
            return False

        key = _NON_WORD.sub("", plan.plain).lower()
        voiced = self._voiced_texts(event) if cfg.dedupe != "off" and key else None
        if voiced is not None and key in voiced:
            logger.info(f"[fish_tts] 本轮已转过相同内容的语音，重复回复处理方式：{cfg.dedupe}")
            if cfg.dedupe != "keep_text":
                self._drop_duplicate(event, result)
            return False

        if cfg.prob < 1.0 and random.random() > cfg.prob:
            return False

        # 先占位再合成：合成要几秒，这期间进来的重复调用能看到标记
        with contextlib.suppress(Exception):
            setattr(result, _CLAIM_ATTR, True)
        if voiced is not None:
            voiced.append(key)

        def _release() -> None:
            if voiced is not None and key in voiced:
                voiced.remove(key)

        path = await self._synthesize(event, plan.spoken, cfg)
        if not path:
            _release()
            return False

        try:
            voice = Comp.Record.fromFileSystem(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[fish_tts] 构造语音消息失败: {exc}")
            _release()
            return False

        result.chain = self._build_chain(result.chain, plan, voice, cfg)
        self._remember(event, plan.plain, cfg)

        limit = f"≤ {cfg.max_chars}" if cfg.max_chars > 0 else "（不限长）"
        tags = emotion.count(plan.spoken, cfg.emotion) if cfg.emotion is not None else 0
        paired = emotion.count_paired(plan.spoken, cfg.emotion) if cfg.emotion is not None else 0
        extra = f"，含 {tags} 个情绪标签" if tags else ""
        if paired:
            extra += f"、{paired} 个成对标签"
        logger.info(f"[fish_tts] {len(plan.plain)} 字 {limit}，已转为语音{extra}")
        return True

    @staticmethod
    def _strip_chain_tags(event: AstrMessageEvent, result, cfg: Settings) -> None:
        """发文字时去掉消息链里的情绪标签（代码块、行内代码里的方括号不动）。"""
        try:
            changed = False
            kept = []
            for comp in result.chain:
                if isinstance(comp, Comp.Plain):
                    old = getattr(comp, "text", "") or ""
                    new = emotion.strip_outside_code(old, cfg.emotion)
                    if new != old:
                        changed = True
                        if not new.strip():
                            continue  # 整段只剩标签：这一段不用发了
                        comp.text = new
                kept.append(comp)
            if not changed:
                return
            if kept:
                result.chain = kept
                return
            event.clear_result()  # 整条消息只有标签：没东西可发
        except Exception as exc:  # noqa: BLE001  去标签出任何问题都不能影响消息发送
            logger.warning(f"[fish_tts] 去除情绪标签失败（已忽略）: {exc}")

    # ---------- 语音记忆：记账 ----------
    def _remember(self, event: AstrMessageEvent, spoken: str, cfg: Settings) -> None:
        """记下这条语音说了什么。任何出错都只记日志，不能影响语音本身发出去。"""
        if not cfg.memory:
            return
        try:
            chat = _chat_key(event)
            self.memory.add(
                chat, spoken, per_chat=cfg.memory_size, keep_seconds=cfg.memory_keep_hours * 3600
            )
            self._schedule_save(cfg)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[fish_tts] 记录语音失败（已忽略）: {exc}")

    def _schedule_save(self, cfg: Settings) -> None:
        if not cfg.memory_persist or self.memory.path is None:
            return
        task = self._save_task
        if task is not None and not task.done():
            return  # 已经有一个等着落盘的任务，它会把最新内容一起写
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._save_task = loop.create_task(self._save_later())

    async def _save_later(self) -> None:
        await asyncio.sleep(_SAVE_DELAY)
        while self.memory.dirty:
            snapshot = self.memory.snapshot()
            self.memory.dirty = False
            try:
                await asyncio.to_thread(VoiceMemory.write, self.memory.path, snapshot)
            except Exception as exc:  # noqa: BLE001
                self.memory.dirty = True
                logger.warning(f"[fish_tts] 保存语音记录失败，下次再试: {exc}")
                return

    async def terminate(self):
        """插件卸载 / 重载时把还没落盘的记录写掉。"""
        task = self._save_task
        if task is not None and not task.done():
            task.cancel()
        try:
            if self.memory.dirty and self.memory.path is not None and self._settings().memory_persist:
                self.memory.save()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[fish_tts] 退出时保存语音记录失败: {exc}")

    # ---------- 语音记忆：给大模型补上下文 ----------
    @filter.on_llm_request()
    async def inject_voice_context(self, event: AstrMessageEvent, req):
        cfg = self._settings()
        if not cfg.enabled:
            return
        if event.get_platform_name() in cfg.skip_platforms:
            return
        if cfg.emotion is not None and cfg.inject_emotion_prompt:
            try:  # 固定文本追加在系统提示词末尾，不影响提示词缓存
                _append_system(req, cfg.emotion_prompt)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[fish_tts] 追加情绪标签提示词失败（已忽略）: {exc}")
        if not cfg.memory:
            return
        try:
            await self._inject(event, req, cfg)
        except Exception as exc:  # noqa: BLE001  语音记忆出任何问题都不能影响正常对话
            logger.warning(f"[fish_tts] 补充语音上下文失败（已忽略）: {exc}")

    async def _inject(self, event: AstrMessageEvent, req, cfg: Settings) -> None:
        chat = _chat_key(event)

        note = await self._quote_note(event, req, chat, cfg) if cfg.inject_quote else None
        if note:
            _add_note(req, note)
            return  # 已经说清楚引用的是哪条，不再重复补「最近语音」

        if cfg.recent_mode == "off":
            return
        if cfg.recent_mode == "keywords" and not _mentions(event.message_str or "", cfg.recent_keywords):
            return
        records = self.memory.recent(chat, cfg.recent_count, cfg.recent_minutes * 60)
        if records:
            _add_note(req, cfg.recent_note.replace("{list}", _fmt_list(records)))

    async def _quote_note(self, event: AstrMessageEvent, req, chat: str, cfg: Settings) -> "str | None":
        """用户引用了机器人自己的消息时，返回要补给大模型的说明；其他情况返回 None。"""
        reply = _first_reply(event)
        if reply is None:
            return None

        mid = _clean_id(getattr(reply, "id", ""))
        # 这个消息 ID 以前认出来过：直接命中，连接口都不用调
        if mid:
            hit = self.memory.find(chat, msg_id=mid)
            if hit.kind == "id":
                logger.info("[fish_tts] 引用识别：按消息 ID 命中语音记录")
                return cfg.quote_note.replace("{text}", hit.record.text)

        self_id = _call_str(event, "get_self_id")
        sender = _clean_id(getattr(reply, "sender_id", ""))
        if sender and self_id and sender != self_id:
            return None  # 引用的不是机器人自己的消息

        chain = list(getattr(reply, "chain", None) or [])
        voice = any(isinstance(c, Comp.Record) for c in chain)
        known_self = bool(sender and self_id and sender == self_id)

        if chain and not voice:
            # 机器人自己的文字消息：框架一般已经把引用内容带进请求了，没带才补
            if not known_self:
                return None
            quoted = _plain_text(chain) or str(getattr(reply, "message_str", "") or "")
            if quoted and not _request_contains(req, quoted):
                return _NOTE_QUOTE_TEXT.replace("{text}", _shorten(quoted))
            return None

        # 语音，或者适配器没能解析出内容：反查被引用消息，拿到真实发送时间
        quoted_ts = None
        if event.get_platform_name() == "aiocqhttp":
            if mid:
                info = await _get_msg(event, mid)
                if info is not None:
                    if info.sender and self_id and info.sender != self_id:
                        return None
                    known_self = known_self or bool(info.sender and info.sender == self_id)
                    voice = voice or info.voice
                    quoted_ts = info.ts
        else:
            raw_time = getattr(reply, "time", 0)
            quoted_ts = float(raw_time) if isinstance(raw_time, (int, float)) and raw_time > 0 else None

        if not (known_self and voice):
            return None  # 确认不了这是机器人自己发的语音：什么都不加，保持原样

        match = self.memory.find(
            chat, msg_id=mid, ts=quoted_ts, tolerance=cfg.match_tolerance
        )
        if match.kind in ("id", "time"):
            if match.kind == "time" and mid:
                self.memory.bind(match.record, mid)  # 记住 ID，下次再引用就是精确命中
                self._schedule_save(cfg)
            logger.info(f"[fish_tts] 引用识别：命中语音记录（{match.kind}）")
            return cfg.quote_note.replace("{text}", match.record.text)

        if match.kind == "ambiguous":
            candidates = sorted(match.candidates, key=lambda r: r.ts)[-cfg.recent_count:]
            logger.info("[fish_tts] 引用识别：时间相近的语音不止一条，没有猜，列出候选")
            return _NOTE_QUOTE_UNSURE.replace("{list}", _fmt_list(candidates))

        if quoted_ts is None:
            records = self.memory.recent(chat, cfg.recent_count, float("inf"))
            if records:
                logger.info("[fish_tts] 引用识别：拿不到被引用消息的时间，列出最近的语音")
                return _NOTE_QUOTE_UNSURE.replace("{list}", _fmt_list(records))
        logger.info("[fish_tts] 引用识别：没有对应的语音记录（可能是升级前发的或已过期）")
        return _NOTE_QUOTE_UNKNOWN
