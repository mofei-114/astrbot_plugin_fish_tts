"""端到端验证 _prepare()：把 main.py 的文本处理链路跑一遍，不需要 AstrBot 运行时。

main.py 顶部 import astrbot，这里用 stub 顶替掉，只取 _prepare 及其依赖。
"""

import os
import sys
import types
import unittest
from dataclasses import dataclass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import emotion  # noqa: E402


def _install_astrbot_stubs():
    """给 main.py 造一套最小可 import 的 astrbot 假模块。"""
    if "astrbot.api" in sys.modules:
        return

    class _Logger:
        def __getattr__(self, name):
            return lambda *a, **k: None

    astrbot = types.ModuleType("astrbot")
    api = types.ModuleType("astrbot.api")
    api.logger = _Logger()
    api.AstrBotConfig = dict

    comp = types.ModuleType("astrbot.api.message_components")

    class _Comp:
        pass

    for name in ("Plain", "Record", "Reply", "At", "AtAll", "Image"):
        setattr(comp, name, type(name, (_Comp,), {}))
    api.message_components = comp

    event = types.ModuleType("astrbot.api.event")

    class _Group:
        """command_group 的返回值要能继续挂 .command()，这里把它做成自闭合的装饰器。"""

        def __call__(self, fn):
            fn.command = self.command
            return fn

        def command(self, *a, **k):
            return lambda fn: fn

    event.AstrMessageEvent = type("AstrMessageEvent", (), {})
    event.filter = types.SimpleNamespace(
        on_decorating_result=lambda **k: (lambda f: f),
        on_llm_request=lambda **k: (lambda f: f),
        command=lambda *a, **k: (lambda f: f),
        command_group=lambda *a, **k: _Group(),
        llm_tool=lambda *a, **k: (lambda f: f),
        custom_filter=lambda *a, **k: (lambda f: f),
    )

    star = types.ModuleType("astrbot.api.star")
    star.Context = type("Context", (), {})
    star.Star = type("Star", (), {"__init__": lambda self, ctx=None: None})

    web = types.ModuleType("astrbot.api.web")
    web.error_response = lambda *a, **k: None
    web.json_response = lambda *a, **k: None
    web.request = types.SimpleNamespace()

    agent = types.ModuleType("astrbot.core.agent.message")
    agent.TextPart = type("TextPart", (), {})

    path_mod = types.ModuleType("astrbot.core.utils.astrbot_path")
    path_mod.get_astrbot_data_path = lambda: "."

    core = types.ModuleType("astrbot.core")
    utils = types.ModuleType("astrbot.core.utils")

    for name, mod in {
        "astrbot": astrbot,
        "astrbot.api": api,
        "astrbot.api.message_components": comp,
        "astrbot.api.event": event,
        "astrbot.api.star": star,
        "astrbot.api.web": web,
        "astrbot.core": core,
        "astrbot.core.agent": types.ModuleType("astrbot.core.agent"),
        "astrbot.core.agent.message": agent,
        "astrbot.core.utils": utils,
        "astrbot.core.utils.astrbot_path": path_mod,
    }.items():
        sys.modules[name] = mod


_install_astrbot_stubs()
import main as plugin_main  # noqa: E402


@dataclass
class FakeSettings:
    """只填 _prepare 真正会读的字段。"""

    emotion: object
    filter_url: bool = True
    strip_at_text: bool = True
    cleanup_markdown: bool = True
    strip_emoji: bool = True
    patterns: tuple = ()


def settings_for(model="drama-3", paired=True, mode="any", max_tags=6):
    rules = emotion.make_rules(
        mode, (), max_tags, paired=plugin_main._is_drama_model(model) and paired
    )
    return FakeSettings(emotion=rules)


class PrepareEndToEndTests(unittest.TestCase):
    def test_paired_tags_survive_markdown_cleanup(self):
        """核心回归：<soft> 不能被 cleanup_markdown 吃掉。"""
        cfg = settings_for()
        tagged, plain, urls, has_code, has_at = plugin_main._prepare(
            "[轻声安慰]没关系，<soft>慢慢来。</soft>", cfg
        )
        self.assertEqual(
            tagged, "[轻声安慰]没关系，<soft>慢慢来。</soft>",
            "送合成的文本必须保留成对标签",
        )
        self.assertEqual(plain, "没关系，慢慢来。", "给用户看的文本必须干净")
        self.assertNotIn("<", plain)
        self.assertFalse(has_code)
        self.assertFalse(has_at)

    def test_mixed_directive_and_paired(self):
        cfg = settings_for()
        text = "[有点累了，声音低下去]<slow>今天好长。</slow><emphasis>真想睡了</emphasis>"
        tagged, plain, *_ = plugin_main._prepare(text, cfg)
        self.assertIn("[有点累了，声音低下去]", tagged)
        self.assertIn("<slow>今天好长。</slow>", tagged)
        self.assertIn("<emphasis>真想睡了</emphasis>", tagged)
        self.assertEqual(plain, "今天好长。真想睡了")

    def test_s2_model_does_not_treat_angle_brackets_as_tags(self):
        """S2 模型下尖括号不是标签，不会按标签剥除。

        注意：上游的 markdown 清理（_MD_MARK）本来就会去掉 ``>``，
        S2 下 ``<soft>`` 会变成 ``<soft``——这是上游既有行为，不是本次改动引入的。
        本次只保证 drama-3 下成对标签被保护起来（见上面的用例）。
        """
        cfg = settings_for(model="s2.1-pro")
        _tagged, plain, *_ = plugin_main._prepare("<soft>你好</soft>", cfg)
        self.assertEqual(plain, "<soft你好</soft")

    def test_length_counts_only_speakable_text(self):
        """标签不计入长度，否则带指令的短句会被误判成长文本。"""
        cfg = settings_for()
        text = "[非常温柔地，像在耳边轻声说话]<soft>嗯。</soft>"
        tagged, plain, *_ = plugin_main._prepare(text, cfg)
        self.assertEqual(plain, "嗯。")
        self.assertEqual(len(plain), 2)
        self.assertIn("<soft>", tagged)

    def test_code_block_untouched_end_to_end(self):
        """行内代码里的尖括号不当作成对标签（与方括号标签规则一致）。

        它们不会被 mask 保护，因此照常受 markdown 清理影响（上游既有行为）。
        这里锁定的是"代码里的标签不被识别为标签"，不是"代码内容原样保留"。
        """
        cfg = settings_for()
        tagged, plain, _urls, has_code, _has_at = plugin_main._prepare(
            "[温柔]看这段 `<soft>不动</soft>` 代码", cfg
        )
        self.assertEqual(plain, "看这段 <soft不动</soft 代码")
        self.assertIn("[温柔]", tagged, "方括号指令仍保留给合成")
        self.assertNotIn("[温柔]", plain, "方括号指令仍从可见文本去掉")
        self.assertFalse(has_code)

    def test_paired_disabled_by_config(self):
        cfg = settings_for(paired=False)
        _tagged, plain, *_ = plugin_main._prepare("<soft>你好</soft>", cfg)
        # 关掉后尖括号按普通字符处理，会被 markdown 清理去掉 ">"（上游既有行为）
        self.assertEqual(plain, "<soft你好</soft")

    def test_mask_does_not_leak_into_plain(self):
        cfg = settings_for()
        _tagged, plain, *_ = plugin_main._prepare("<soft>a</soft>", cfg)
        self.assertNotIn("\x00", plain)


if __name__ == "__main__":
    unittest.main(verbosity=2)
