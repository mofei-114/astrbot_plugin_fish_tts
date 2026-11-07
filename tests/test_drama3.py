"""drama-3 适配的单元测试：成对尖括号标签 + 模型相关的情绪提示词。

只测纯函数（emotion.py 和 main.py 里不依赖 AstrBot 的部分），
不需要网络、不需要 AstrBot 运行时。跑法：

    python -m pytest tests -q
    python tests/test_drama3.py          # 不装 pytest 也能跑
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import emotion  # noqa: E402


def drama_rules(**kwargs):
    return emotion.make_rules("any", (), kwargs.pop("max_tags", 6), paired=True, **kwargs)


def s2_rules(**kwargs):
    return emotion.make_rules("any", (), kwargs.pop("max_tags", 6), **kwargs)


class PairedStripTests(unittest.TestCase):
    """发文字时成对标签要被整对去掉。"""

    def test_paired_tags_removed_from_visible_text(self):
        r = drama_rules()
        self.assertEqual(
            emotion.strip_outside_code("[轻声安慰]没关系，<soft>慢慢来。</soft>", r),
            "没关系，慢慢来。",
        )

    def test_all_seven_paired_tags(self):
        r = drama_rules()
        for name in sorted(emotion.PAIRED_TAGS):
            text = f"前<{name}>中</{name}>后"
            self.assertEqual(emotion.strip_outside_code(text, r), "前中后", name)

    def test_paired_tags_case_insensitive(self):
        r = drama_rules()
        self.assertEqual(
            emotion.strip_outside_code("<WHISPER>小声</Whisper>", r), "小声"
        )

    def test_only_paired_tags_when_enabled(self):
        """S2 规则下尖括号不是标签，原样保留。"""
        r = s2_rules()
        self.assertEqual(
            emotion.strip_outside_code("<soft>慢慢来</soft>", r), "<soft>慢慢来</soft>"
        )

    def test_unknown_angle_tag_left_alone(self):
        """不在七白名单里的尖括号（比如 HTML）不动。"""
        r = drama_rules()
        self.assertEqual(
            emotion.strip_outside_code("<div>文字</div>", r), "<div>文字</div>"
        )

    def test_paired_inside_code_untouched(self):
        r = drama_rules()
        text = "看 `<soft>x</soft>` 和 ``<fast>y</fast>``"
        self.assertEqual(emotion.strip_outside_code(text, r), text)

    def test_paired_outside_code_stripped_inside_kept(self):
        r = drama_rules()
        text = "<soft>正式</soft> 和 `<fast>代码</fast>`"
        self.assertEqual(
            emotion.strip_outside_code(text, r), "正式 和 `<fast>代码</fast>`"
        )


class PairedMaskTests(unittest.TestCase):
    """合成前用占位符保护成对标签，避免被 markdown 清理吃掉。"""

    def test_mask_and_unmask_roundtrip(self):
        r = drama_rules()
        text = "[温柔]没关系，<soft>慢慢来。</soft>"
        masked, tags = emotion.mask_paired(text, r)
        self.assertNotIn("<soft>", masked)
        self.assertEqual(len(tags), 2)
        self.assertEqual(emotion.unmask_paired(masked, tags), text)

    def test_mask_is_markdown_safe(self):
        """占位符里不能有 < > * ` # 这些会被 markdown 清理删掉的字符。"""
        r = drama_rules()
        masked, _ = emotion.mask_paired("<soft>x</soft>", r)
        for bad in "<>*`#~":
            self.assertNotIn(bad, masked)

    def test_no_mask_without_paired(self):
        r = s2_rules()
        text = "<soft>x</soft>"
        masked, tags = emotion.mask_paired(text, r)
        self.assertEqual(masked, text)
        self.assertEqual(tags, [])

    def test_no_mask_inside_code(self):
        r = drama_rules()
        text = "`<soft>x</soft>`"
        masked, tags = emotion.mask_paired(text, r)
        self.assertEqual(masked, text)
        self.assertEqual(tags, [])

    def test_unmask_empty_tags_noop(self):
        self.assertEqual(emotion.unmask_paired("普通文本", []), "普通文本")


class PairBalanceTests(unittest.TestCase):
    """落单的成对标签要能被发现。"""

    def test_unclosed_soft_detected(self):
        self.assertTrue(emotion.has_unpaired("<soft>没闭合", drama_rules()))

    def test_extra_closing_detected(self):
        self.assertFalse(emotion.has_unpaired("多一个</soft>", drama_rules()))

    def test_balanced_ok(self):
        self.assertFalse(emotion.has_unpaired("<soft>a</soft><fast>b</fast>", drama_rules()))

    def test_nested_same_tag_ok(self):
        self.assertFalse(
            emotion.has_unpaired("<soft>a<soft>b</soft>c</soft>", drama_rules())
        )

    def test_no_check_when_disabled(self):
        self.assertFalse(emotion.has_unpaired("<soft>没闭合", s2_rules()))


class PairCountTests(unittest.TestCase):
    def test_count_open_and_close(self):
        r = drama_rules()
        self.assertEqual(emotion.count_paired("<soft>a</soft>", r), 2)

    def test_count_zero_when_disabled(self):
        self.assertEqual(emotion.count_paired("<soft>a</soft>", s2_rules()), 0)


class BracketTagRegressionTests(unittest.TestCase):
    """改动不能破坏原有的方括号识别。"""

    def test_basic(self):
        r = drama_rules()
        self.assertEqual(emotion.strip("[sad][whispering] 我想你了", r), "我想你了")
        self.assertEqual(emotion.count("[sad][whispering] 我想你了", r), 2)

    def test_code_untouched(self):
        """代码段里的方括号不动，代码段外的照常去掉。"""
        r = drama_rules()
        self.assertEqual(
            emotion.strip_outside_code("看 `arr[0]` 和 [happy]好", r), "看 `arr[0]` 和 好"
        )
        self.assertEqual(
            emotion.strip_outside_code("看 `[happy]在代码里`", r), "看 `[happy]在代码里`"
        )

    def test_markdown_link_untouched(self):
        r = drama_rules()
        self.assertEqual(
            emotion.strip("[标题](http://a.b) [happy]真棒", r), "[标题](http://a.b) 真棒"
        )

    def test_natural_language_directive_kept_for_synthesis(self):
        """drama-3 的中文自然语言指令要保留在合成文本里、从可见文本里去掉。"""
        r = drama_rules()
        spoken = emotion.normalize("[温柔地，像贴近对方说话]你来啦。", r)
        plain = emotion.strip("[温柔地，像贴近对方说话]你来啦。", r)
        self.assertIn("[温柔地，像贴近对方说话]", spoken)
        self.assertEqual(plain, "你来啦。")


class DramaPromptTests(unittest.TestCase):
    """模型选 drama-3 时应换成 drama-3 版提示词。"""

    def setUp(self):
        # main.py 顶部 import astrbot，测试环境里没有，这里只取需要的纯函数
        import importlib.util

        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main.py"
        )
        src = open(path, encoding="utf-8").read()
        # 只截取模块级的常量与两个纯函数，避开 astrbot 导入
        start = src.index("_DRAMA_MODEL =")
        end = src.index("# ---------- 配置读取 ----------")
        ns = {}
        exec(compile(src[start:end], "<main-slice>", "exec"), ns)
        self.ns = ns

    def test_drama_model_detected(self):
        is_drama = self.ns["_is_drama_model"]
        for model in ("drama-3", "Drama-3", " drama-3 ", "drama-3-preview", "DRAMA"):
            self.assertTrue(is_drama(model), model)
        for model in ("s2.1-pro", "s2-pro", "s2.1-pro-free", "s1", "", None):
            self.assertFalse(is_drama(model), model)

    def test_prompt_switches_by_model(self):
        pick = self.ns["_emotion_prompt_for"]
        fallback = "S2 默认说明"
        drama = pick("drama-3", fallback)
        self.assertIn("导演", drama)
        self.assertIn("<whisper>", drama)
        self.assertNotEqual(drama, fallback)
        self.assertEqual(pick("s2.1-pro", fallback), fallback)

    def test_drama_prompt_lists_all_seven_tags(self):
        prompt = self.ns["_DRAMA_EMOTION_PROMPT"]
        for name in emotion.PAIRED_TAGS:
            self.assertIn(f"<{name}>", prompt, name)

    def test_drama_prompt_forbids_unpaired(self):
        self.assertIn("必须成对", self.ns["_DRAMA_EMOTION_PROMPT"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
