"""Japanese G-Maze distractor generator (Boyce et al. 2020 design).

Usage: python maze_japanese.py input.txt output.txt [--seed N] [-p params.txt]
"""


import argparse
import ast
import base64
import csv
import gzip
import logging
import math
import os
import pickle
import random
import re
import time
import urllib.request
import urllib.error
from collections import Counter
from typing import Dict, List, Optional

import torch
import wordfreq
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import fugashi
    _FUGASHI_AVAILABLE = True
except ImportError:
    _FUGASHI_AVAILABLE = False
    logging.warning("fugashi not installed; raw Japanese input will not be auto-tokenized.")


DEFAULT_MODEL_NAME = "rinna/japanese-gpt2-medium"

DEFAULT_PARAMS: Dict = {
    "min_pmi":     14.0,
    "min_abs":     21.0,
    "num_to_test":  80,        # max candidates per slot
    "max_repeat":    0,        # max reuses of exact form (0 = stem-cap only)
    "model_name": DEFAULT_MODEL_NAME,
    "grade_level":   6,        # kanji filter: 0 = off, 1-6 = elementary grade
}

_SURPRISAL_CEILING = 35.0


_POOL_CACHE_VERSION = 18


_FREQ_BIN_WIDTH  = 0.1   # log-freq units per bin
_FREQ_BIN_RADIUS = 10    # ±10 bins = ±1.0 log-freq around target

_GRADE_FREQ_FLOOR: Dict[int, float] = {
    0: -15.0, 1: 10.0, 2: 9.0, 3: 8.0, 4: 7.0, 5: 6.0, 6: 5.0,
}


_JP_RE = re.compile(
    r'[\u3041-\u3096\u30A1-\u30F6\u4E00-\u9FFF\u3400-\u4DBF\uF900-\uFAFF\u30FC]'
)
_KANJI_RE    = re.compile(r'[\u4E00-\u9FFF\u3400-\u4DBF\uF900-\uFAFF]')
_KATAKANA_RE = re.compile(r'[\u30A1-\u30F6\u30FC]')

_SMALL_KANA_START = re.compile(r'^[ぁぃぅぇぉっゃゅょァィゥェォッャュョ]')

_SMALL_TSU_END = re.compile(r'[っッ]$')

_PUNCT_FINAL = re.compile(r'[。！？]$')


_CASE_PARTICLES = frozenset({"が", "を", "に", "で", "は"})

# Only が and を give CATEGORICAL double-particle violations within a clause.
# に, で, は can grammatically repeat (time+location, topic+contrastive, etc.),
# so they only warrant tier-2 treatment in violation strategy selection.
_STRONG_DUPLICATE_PARTICLES = frozenset({"が", "を"})

_ALL_PARTICLES = [
    "が", "を", "に", "で", "は", "の", "と",
    "から", "まで", "へ", "も",
]


_CONTENT_POS = frozenset({
    "名詞", "動詞", "形容詞", "形容動詞", "副詞",
    "連体詞", "接続詞", "感動詞", "接頭詞", "接頭辞",
})

# Verbs whose lemma marks them as compound-forming when preceded by a noun
# (サ変 like 勉強する) or by another verb in te-form (aspectual auxiliaries).
_COMPOUND_VERB_LEMMAS = frozenset({
    "する", "為る",
    "できる", "出来る",
    "なる", "成る",
    "いる", "居る",
    "ある", "有る", "在る",
    "おく", "置く",
    "みる", "見る",
    "くる", "来る",
    "しまう", "仕舞う",
    "いく", "行く", "ゆく",
})

_VERB_DICT_TAILS = frozenset("うくぐすつぬぶむる")

_VERB_PAST_TAILS = frozenset("ただ")

_ADJ_TAILS = frozenset("い")


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

_BLOCKLIST_URLS = [
    "https://raw.githubusercontent.com/LDNOOBWV2/"
    "List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words_V2/master/data/ja.txt",
    "https://raw.githubusercontent.com/LDNOOBWV2/"
    "List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words_V2/main/data/ja.txt",
]
_BLOCKLIST_CACHE = os.path.join(_SCRIPT_DIR, ".blocklist_ja_cache.txt")

_SUPPLEMENT_B64 = (
    "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"
)


def _decode_supplement() -> set:
    """Decode the gzip+base64 encoded supplement blocklist."""
    try:
        raw = gzip.decompress(base64.b64decode(_SUPPLEMENT_B64))
        return {w for w in raw.decode("utf-8").splitlines() if w.strip()}
    except Exception as exc:
        logging.warning("Failed to decode supplement blocklist: %s", exc)
        return set()


def _fetch_external_blocklist() -> set:
    """Download the LDNOOBW Japanese blocklist, cache it, and return as a set."""
    if os.path.isfile(_BLOCKLIST_CACHE):
        try:
            with open(_BLOCKLIST_CACHE, "r", encoding="utf-8") as f:
                words = {line.strip() for line in f if line.strip()
                         and not line.startswith("#")}
            if words:
                logging.info("Loaded %d external blocklist words from cache",
                             len(words))
                return words
        except OSError:
            pass

    for url in _BLOCKLIST_URLS:
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "maze-japanese/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            words = {line.strip() for line in raw.splitlines() if line.strip()}
            if words:
                try:
                    with open(_BLOCKLIST_CACHE, "w", encoding="utf-8") as f:
                        f.write("\n".join(sorted(words)) + "\n")
                    print(f"  ✓ Downloaded & cached {len(words)} "
                          f"blocklist words from LDNOOBW")
                except OSError:
                    print(f"  ✓ Downloaded {len(words)} blocklist words "
                          f"(cache write failed)")
                return words
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            logging.debug("Blocklist download failed for %s: %s", url, exc)
            continue

    logging.warning(
        "Could not download external blocklist (no internet?). "
        "Using encoded supplement only."
    )
    return set()


_EXTERNAL_BLOCKED = _fetch_external_blocklist()

_SUPPLEMENT_BLOCKED = _decode_supplement()

_MORPHOLOGICAL_BLOCKED = {
    "第", "超", "新", "大", "小", "全", "各", "本", "同", "前", "次",
    "元", "現", "旧", "約", "毎", "再", "未", "非", "無", "不",
    "的", "性", "化", "式", "用", "製", "付", "済", "中",
    "こと", "もの", "ところ", "ため", "はず", "わけ", "まま",
    "ほう", "ほか",
    "出", "入", "上", "下",
    "たち", "タチ",
    "あり", "なり", "つき", "かかり", "つかい", "かけ",
    "はじめ", "おわり", "つづき", "まとめ", "ならび",
    "出し", "だし",
    "だっ", "じゃっ", "ちゃっ", "とっ", "がっ", "やっ", "わっ",
    "でし", "だろ", "でしょ",
    "もん",   # spoken もの — too casual for experimental stimuli
}

_BLOCKED_WORDS = frozenset(
    _EXTERNAL_BLOCKED | _SUPPLEMENT_BLOCKED | _MORPHOLOGICAL_BLOCKED
)


_BOUND_POS_SUBS = frozenset({
    "接頭詞", "接尾詞", "接頭辞", "接尾辞",   # IPADic labels
    "接頭", "接尾",                             # substring matches
    "非自立",                                   # dependent (こと, もの)
    "数", "助数詞",                             # numbers & counters
    "形式名詞",                                 # formal nouns (UniDic)
})

_EXCLUDED_DISTRACTOR_POS = frozenset({
    "接続詞",   # conjunctions
    "接頭詞",   # prefixes
    "接尾辞",   # suffixes (UniDic)
    "接尾詞",   # suffixes (IPADic)
    "数詞",     # numerals
    "助数詞",   # counters
})

_ABSTRACT_NOUN_SUBS = frozenset({
    "副詞可能",     # adverbial nouns (前, 後, 時)
    "非自立可能",   # formal / dependent nouns
    "形状詞可能",   # adjectival nouns overlapping na-adj
    "助数詞可能",   # counter-capable nouns
})


_FORMAL_NOUNS = frozenset({
    "事", "こと", "もの", "物", "者", "方", "所", "ところ",
    "前", "後", "上", "下", "中", "間", "時", "頃",
    "為", "ため", "はず", "わけ", "まま", "よう", "そう",
})
_FORMAL_NOUN_CAP = 2   # max uses per formal-noun stem across the run
_STEM_VARIETY_CAP = 2  # max uses per ANY content stem (prevents 世界×10)

_BORING_WORDS = frozenset({
    "こと", "事", "もの", "物", "とき", "時", "ため", "為",
    "ほう", "方", "わけ", "ほか", "他", "たち", "あと", "後",
    "まえ", "前", "うち", "なか", "中",
})
_BORING_PENALTY = 5.0  # extra bits added to surprisal threshold

_KATAKANA_BONUS = -0.3

_STANDALONE_SINGLE_CHAR = frozenset({
    "犬", "猫", "馬", "鳥", "魚", "虫", "花", "木", "山", "川",
    "海", "空", "雨", "雪", "風", "星", "月", "日", "火", "水",
    "金", "土", "石", "竹", "米", "肉", "酒", "茶", "薬", "糸",
    "紙", "絵", "歌", "音", "声", "色", "光", "影", "夢", "嘘",
    "顔", "目", "耳", "口", "手", "足", "頭", "腕", "胸", "腹",
    "骨", "血", "汗", "涙", "傷", "命", "死", "毒", "鬼", "神",
    "王", "姫", "侍", "僧", "客", "敵", "友", "親", "子", "妻",
    "夫", "兄", "姉", "弟", "妹", "孫", "姓", "名", "歳", "春",
    "夏", "秋", "冬", "朝", "昼", "夜", "今", "昔", "先", "奥",
    "横", "隣", "裏", "表", "底", "端", "角", "穴", "橋", "塔",
    "門", "庭", "畑", "森", "島", "国", "村", "町", "城", "寺",
    "駅", "店", "家", "部", "席", "床", "壁", "窓", "鍵", "箱",
    "皿", "鏡", "笛", "剣", "弓", "盾", "罠", "旗", "札", "鐘",
})


def _is_katakana_heavy(word: str) -> bool:
    """True if >50% of the word's script characters are katakana."""
    chars = _JP_RE.findall(word)
    if not chars:
        return False
    return sum(1 for c in chars if _KATAKANA_RE.match(c)) / len(chars) > 0.5


_KYOIKU_KANJI: Dict[int, str] = {
    1: ("一右雨円王音下火花貝学気九休玉金空月犬見五口校左三山子四糸字耳七車手"
        "十出女小上森人水正生青夕石赤千川先早草足村大男竹中虫町天田土二日入年"
        "白八百文木本名目立力林六"),
    2: ("引羽雲園遠何科夏家歌画回会海絵外角楽活間丸岩顔汽記帰弓牛魚京強教近"
        "兄形計元言原戸古午後語工公広交光考行高黄合谷国黒今才細作算止市矢姉思"
        "紙寺自時室社弱首秋週春書少場色食心新親図数西声星晴切雪船線前組走多太"
        "体台地池知茶昼長鳥朝直通弟店点電刀冬当東答頭同道読内南肉馬売買麦半"
        "番父風分聞米歩母方北毎妹万明鳴毛門夜野友用曜来里理話"),
    3: ("悪安暗医委意育員院飲運泳駅央横屋温化荷界開階寒感漢館岸起期客究急級宮"
        "球去橋業曲局銀区苦具君係軽血決研県庫湖向幸港号根祭皿仕死使始指歯詩次"
        "事持式実写者主守取酒受州拾終習集住重宿所暑助昭消商章勝乗植申身神真深"
        "進世整昔全相送想息速族他打対待代第題炭短談着柱注丁帳調追定届鉄転都度"
        "投豆島湯登等動童農波配倍箱畑発反坂板皮悲美鼻筆氷表秒病品負部服福物平"
        "返勉放味命面問役薬由油有遊予羊洋葉陽様落流旅両緑礼列練路和丁"),
    4: ("愛案以衣位茨印英栄媛塩岡億加果貨課芽賀改械害街各覚潟完官管関観願岐希"
        "季旗器機議求泣給挙漁共協鏡競極熊訓群軍郡径型景芸欠結建健験固功好候香"
        "佐差菜最埼材崎昨札刷察参産散残氏司史士始試児治滋鹿失借種周祝順初松笑"
        "唱焼象照城縄臣信井成省清静席積折節説浅戦選然争倉巣束側続卒孫帯隊達単"
        "置仲沖兆低底的典伝徒努灯堂働特徳栃奈梨熱念敗梅博阪飯飛必票標不夫付"
        "府副粉兵庫別辺変便包法望牧末満未民無約勇養浴利陸良料量輪類令冷例連老"
        "労録"),
    5: ("圧移因永営衛易益液演応往桜恩可仮価河過快解格確額刊幹慣眼紀基寄規技義"
        "逆久旧救居許境均禁句群経潔件券険検限現減故個護効厚耕鉱構興講混査再災"
        "妻採際在財罪雑酸賛支志枝師資飼示似識質舎謝授修述術準序招承証常条状織職"
        "制性政勢精製税責績接設舌絶祖素総造像増則測属率損退貸態団断築張提程適敵"
        "統銅導徳独任燃能破犯判版比肥非備俵評貧布婦富武復複仏編弁保墓豊防貿暴"
        "務夢迷綿輸余預容略留領歴"),
    6: ("胃異遺域宇映延沿我灰拡革閣割株干巻看簡危揮貴疑吸供胸郷勤筋系敬警劇激"
        "穴絹権憲源厳己呼誤后孝皇紅降鋼刻穀骨困砂座済裁策冊蚕至私姿視詞誌磁射"
        "捨尺若樹収宗就衆従縦縮熟純処署諸除将傷障城蒸針仁垂推寸盛聖誠宣専泉洗"
        "染善奏窓創装層操蔵臓存尊退宅担探誕段暖値宙忠著庁頂潮賃痛展討党糖届難"
        "乳認納脳派拝背肺俳班晩否批秘腹奮並陛閉片補暮宝訪亡忘棒枚幕密盟模訳優"
        "郵幼欲翌乱卵覧裏律臨朗論"),
}


def _load_kanji_whitelist(grade_level: int) -> Optional[set]:
    """Return the set of kanji allowed up to *grade_level* (1-6)."""
    if grade_level <= 0:
        return None

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "kanji_grades.txt")
    if os.path.isfile(path):
        allowed = set()
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                if i > grade_level:
                    break
                allowed.update(ch for ch in line.strip() if _KANJI_RE.match(ch))
        if allowed:
            return allowed

    allowed = set()
    for g in range(1, grade_level + 1):
        if g in _KYOIKU_KANJI:
            allowed.update(ch for ch in _KYOIKU_KANJI[g] if _KANJI_RE.match(ch))
    return allowed or None


def _word_ok_kanji(word: str, whitelist: Optional[set]) -> bool:
    """True if every kanji in *word* is in the whitelist (or no whitelist)."""
    if whitelist is None:
        return True
    return all(ch in whitelist for ch in word if _KANJI_RE.match(ch))


def _freq_bin(logfreq: float) -> int:
    """Map a log-frequency value to an integer bin index."""
    return int(logfreq / _FREQ_BIN_WIDTH)


_LOG2_1E9 = math.log2(1e9)          # ≈ 29.897
_INV_LN2  = 1.0 / math.log(2)       # ≈ 1.443


def _unigram_bits(logfreq: float) -> float:
    """Unigram surprisal of a word in bits, from wordfreq's logfreq."""
    return _LOG2_1E9 - logfreq * _INV_LN2


def _no_duplicates(lst: List) -> bool:
    """True if lst contains no duplicate elements."""
    return len(lst) == len(set(lst))


class JapaneseTokenizer:
    """Bunsetsu-level tokeniser backed by MeCab (UniDic preferred, IPADic fallback)."""

    def __init__(self):
        if not _FUGASHI_AVAILABLE:
            raise RuntimeError("fugashi required. pip install fugashi unidic-lite")
        try:
            import unidic_lite
            self._tagger = fugashi.GenericTagger("-d " + unidic_lite.DICDIR)
            print("  MeCab dictionary: unidic-lite")
        except ImportError:
            import ipadic
            self._tagger = fugashi.GenericTagger("-d " + ipadic.DICDIR)
            print("  MeCab dictionary: ipadic (install unidic-lite for better results)")


    def _pos(self, m) -> str:
        """Top-level POS tag of a morpheme."""
        f = m.feature
        if isinstance(f, tuple):
            return f[0] if f else ""
        return f.split(",")[0] if f else ""

    def _feat(self, m, idx: int) -> str:
        """Get a specific feature field (0-indexed) from a morpheme."""
        f = m.feature
        if isinstance(f, tuple):
            return f[idx] if len(f) > idx and f[idx] != "*" else ""
        parts = f.split(",") if f else []
        return parts[idx] if len(parts) > idx and parts[idx] != "*" else ""


    def tokenize(self, sentence: str) -> List[str]:
        """Split a raw Japanese sentence into bunsetsu chunks.

        Compound rules (no intervening particle): 名詞+する-verb,
        動詞+動詞 (aux), *+接尾辞, 固有名詞+普通名詞.  A case/topic/final
        particle between two content words always forces a split, avoiding
        false compounds like 昨日公園.
        """
        morphemes = self._tagger(sentence)
        if not morphemes:
            return []

        _SPLITTING_PARTICLES = (
            "格助詞", "係助詞", "副助詞", "並立助詞", "終助詞",
        )
        bunsetsu: List[str] = []
        current = ""
        last_content_pos = ""
        last_content_sub = ""
        had_splitting_particle = False
        prefix_pending = False

        for w in morphemes:
            p = self._pos(w)
            pos_sub2 = self._feat(w, 1)
            pos_sub = pos_sub2 + "/" + self._feat(w, 2)

            if p not in _CONTENT_POS:
                current += w.surface
                if p == "助詞" and pos_sub2 in _SPLITTING_PARTICLES:
                    had_splitting_particle = True
                continue

            if not current:
                current = w.surface
                last_content_pos = p
                last_content_sub = pos_sub
                had_splitting_particle = False
                prefix_pending = p in ("接頭詞", "接頭辞")
                continue

            if prefix_pending:
                current += w.surface
                last_content_pos = p
                last_content_sub = pos_sub
                prefix_pending = False
                continue

            if had_splitting_particle:
                bunsetsu.append(current)
                current = w.surface
                last_content_pos = p
                last_content_sub = pos_sub
                had_splitting_particle = False
                prefix_pending = p in ("接頭詞", "接頭辞")
                continue

            lemma = self._feat(w, 7)

            attach = (
                (p == "動詞" and last_content_pos == "名詞"
                    and lemma in _COMPOUND_VERB_LEMMAS)
                or (p == "動詞" and last_content_pos == "動詞")
                or "接尾" in pos_sub
                or (p == "名詞" and last_content_pos == "名詞"
                    and "固有名詞" in last_content_sub
                    and "普通名詞" in pos_sub)
                or (last_content_pos == "名詞"
                    and "副詞可能" in last_content_sub
                    and p in ("副詞", "形容詞"))
            )

            if attach:
                current += w.surface
            else:
                bunsetsu.append(current)
                current = w.surface
            last_content_pos = p
            last_content_sub = pos_sub

        if current:
            bunsetsu.append(current)
        return bunsetsu

    def get_bunsetsu_info(self, bunsetsu: str) -> Dict:
        """Analyse a bunsetsu and return its POS, trailing particle, and stem."""
        morphemes = self._tagger(bunsetsu)
        if not morphemes:
            return {"pos": "other", "particle": "", "stem": bunsetsu}

        first_content_pos = "other"
        _POS_MAP = {"名詞": "noun", "動詞": "verb",
                     "形容詞": "adj", "形容動詞": "adj", "副詞": "adv"}
        for m in morphemes:
            p = self._pos(m)
            if p in _CONTENT_POS:
                first_content_pos = _POS_MAP.get(p, "other")
                break

        stem_end = len(morphemes)
        for i in range(len(morphemes) - 1, -1, -1):
            p = self._pos(morphemes[i])
            if p in ("助詞", "記号"):
                stem_end = i
            else:
                break
        particle = "".join(m.surface for m in morphemes[stem_end:])
        stem = "".join(m.surface for m in morphemes[:stem_end])
        return {"pos": first_content_pos, "particle": particle, "stem": stem}


def _conjugate_past(dict_form: str, conj_type: str) -> Optional[str]:
    """Generate past-tense (た形) from a verb's dictionary form."""
    if not dict_form:
        return None

    if dict_form == "する":
        return "した"
    if dict_form in ("来る", "くる"):
        return "来た" if "来" in dict_form else "きた"
    if "サ変" in conj_type:
        if dict_form.endswith("する"):
            return dict_form[:-2] + "した"
        if dict_form.endswith("ずる"):
            return dict_form[:-2] + "じた"
        return None
    if "カ変" in conj_type:
        return dict_form[:-2] + "きた" if dict_form.endswith("くる") else None

    if "一段" in conj_type:
        return dict_form[:-1] + "た" if dict_form.endswith("る") else None

    if "五段" in conj_type:
        if dict_form == "行く":
            return "行った"
        stem = dict_form[:-1]
        _GODAN_TABLE = {
            "う": "った", "つ": "った", "る": "った",
            "く": "いた", "ぐ": "いだ", "す": "した",
            "ぬ": "んだ", "ぶ": "んだ", "む": "んだ",
        }
        suf = _GODAN_TABLE.get(dict_form[-1])
        return stem + suf if suf else None

    return None


class JapaneseGPT2LM:
    """Wrapper around a Hugging Face causal LM for surprisal computation."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        print(f"Loading Japanese LM: {model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self._model = AutoModelForCausalLM.from_pretrained(model_name)
        self._model.eval()

        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")
        self._model.to(self._device)

        self._pad_id = (
            getattr(self._tokenizer, "pad_token_id", None)
            or getattr(self._tokenizer, "eos_token_id", 0)
            or 0
        )

        if (self._device.type == "cuda"
                and hasattr(torch, "compile")
                and os.name != "nt"):
            try:
                self._model = torch.compile(self._model)
                print(f"  running on {self._device} (torch.compile enabled)")
            except Exception:
                print(f"  running on {self._device} (torch.compile unavailable)")
        else:
            print(f"  running on {self._device}")


    def empty_sentence(self) -> tuple:
        """Return an empty sentence context (no token IDs yet)."""
        return ()

    def update(self, ctx: tuple, word: str) -> tuple:
        """Append *word*'s token IDs to the running context."""
        return ctx + tuple(self._encode(word))


    def get_surprisal(self, ctx: tuple, word: str) -> float:
        """Single-word surprisal (bits) given context *ctx*."""
        return self.get_surprisals_batch(ctx, [word])[0]

    def get_surprisals_batch(self, ctx: tuple, words: List[str]) -> List[float]:
        """Compute surprisal (bits) for many candidates using KV-caching."""
        if not words:
            return []
        all_ids = [self._encode(w) for w in words]
        inv_ln2 = 1.0 / math.log(2)

        prefix_ids = list(ctx) if ctx else [getattr(self._tokenizer, "bos_token_id", None) or 0]

        try:
            prefix_inp = torch.tensor([prefix_ids], device=self._device)
            with torch.no_grad():
                prefix_out = self._model(prefix_inp, use_cache=True)
                prefix_lp = torch.log_softmax(
                    prefix_out.logits[0, -1, :], dim=-1
                ).cpu()

                past_kv_raw = prefix_out.past_key_values
                if hasattr(past_kv_raw, "to_legacy_cache"):
                    past_kv = past_kv_raw.to_legacy_cache()
                elif not isinstance(past_kv_raw, tuple):
                    past_kv = tuple(past_kv_raw)
                else:
                    past_kv = past_kv_raw
        except Exception as exc:
            logging.warning("prefix forward error: %s", exc)
            return [0.0] * len(words)

        results = [0.0] * len(words)
        for i, ids in enumerate(all_ids):
            if ids:
                results[i] = -prefix_lp[ids[0]].item() * inv_ln2

        multi = [(i, ids) for i, ids in enumerate(all_ids) if len(ids) > 1]
        if not multi:
            return results

        max_cand_len = max(len(ids) for _, ids in multi)
        batch_rows = [
            ids + [self._pad_id] * (max_cand_len - len(ids))
            for _, ids in multi
        ]

        try:
            inp = torch.tensor(batch_rows, device=self._device)
            B = len(batch_rows)
            expanded_kv = tuple(
                tuple(t.expand(B, *(-1,) * (t.dim() - 1)) for t in layer)
                for layer in past_kv
            )
            with torch.no_grad():
                out = self._model(inp, past_key_values=expanded_kv, use_cache=False)
                log_probs = torch.log_softmax(out.logits, dim=-1).cpu()
        except Exception as exc:
            logging.warning("KV-batched surprisal error: %s", exc)
            return results

        for batch_idx, (orig_idx, ids) in enumerate(multi):
            total = results[orig_idx]   # first-token from step 2
            for j in range(1, len(ids)):
                total += -log_probs[batch_idx, j - 1, ids[j]].item() * inv_ln2
            results[orig_idx] = total

        return results


    def _encode(self, text: str) -> List[int]:
        """Tokenise *text* into a list of token IDs (no special tokens)."""
        return self._tokenizer.encode(text, add_special_tokens=False)


class _BunsetsuEntry:
    """A single pre-formed distractor bunsetsu (e.g. '猫が', '走った')."""
    __slots__ = ("text", "pos", "particle", "freq", "length", "concrete")

    def __init__(self, text: str, pos: str, particle: str, freq: float,
                 concrete: bool = True):
        self.text = text
        self.pos = pos
        self.particle = particle   # 'が','を', etc. or '' for verbs/adjs
        self.freq = freq           # log-frequency from wordfreq
        self.length = len(text)
        self.concrete = concrete   # False for abstract/formal/adverbial nouns


class JapaneseDistractorDict:
    """Pool of morphologically valid bunsetsu indexed for fast lookup."""

    def __init__(self, params: Dict = {}, tokenizer: "Optional[JapaneseTokenizer]" = None):
        grade_level = int(params.get("grade_level", 0))

        cache_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f".bunsetsu_pool_v{_POOL_CACHE_VERSION}_g{grade_level}.pkl",
        )
        if os.path.isfile(cache_path):
            try:
                print(f"Loading distractor dictionary from cache: {cache_path}")
                with open(cache_path, "rb") as f:
                    state = pickle.load(f)
                self._entries = state["entries"]
                self._by_particle = state["by_particle"]
                self._by_pos = state["by_pos"]
                self._by_freq_bin = state.get("by_freq_bin", {})
                if not self._by_freq_bin:
                    self._build_freq_bins()
                print(f"  {len(self._entries):,} bunsetsu loaded")
                return
            except Exception as exc:
                logging.warning("cache load failed (%s); rebuilding", exc)

        print("Building Japanese distractor dictionary (bunsetsu pool)...")
        _ensure_freq_dict()

        freq_floor = _GRADE_FREQ_FLOOR.get(grade_level, -15.0)
        kanji_wl = _load_kanji_whitelist(grade_level) if grade_level > 0 else None
        if grade_level > 0:
            print(f"  Grade {grade_level}: freq≥{freq_floor:.1f}, "
                  f"kanji={'ON' if kanji_wl else 'OFF'}")

        tagger = tokenizer._tagger if tokenizer else None

        self._entries: List[_BunsetsuEntry] = []
        self._by_particle: Dict[str, List[_BunsetsuEntry]] = {p: [] for p in _ALL_PARTICLES}
        self._by_particle[""] = []
        self._by_pos: Dict[str, List[_BunsetsuEntry]] = {}
        self._by_freq_bin: Dict[int, List[_BunsetsuEntry]] = {}

        counts = {"noun": 0, "verb": 0, "adj": 0, "skipped": 0}

        for word, logfreq in _JA_FREQ_DICT.items():
            if logfreq < freq_floor:
                continue
            if not _JP_RE.search(word):
                continue
            if re.search(r'[a-zA-Z0-9\s]', word):
                continue
            if not _word_ok_kanji(word, kanji_wl):
                continue
            if _SMALL_KANA_START.match(word):
                continue
            if _SMALL_TSU_END.search(word):
                continue
            if word in _BLOCKED_WORDS:
                continue
            if len(word) == 1 and word not in _STANDALONE_SINGLE_CHAR:
                continue

            if tagger is None:
                continue
            morphemes = tagger(word)
            if not morphemes:
                continue

            first_pos = ""
            conj_type = ""
            first_pos_subs: List[str] = []
            content_count = 0
            is_bound = False

            for m in morphemes:
                feat = m.feature
                if isinstance(feat, tuple):
                    p = feat[0] if feat else ""
                    pos_subs = [f for f in feat[1:4] if f and f != "*"]
                    ct = feat[4] if len(feat) > 4 and feat[4] != "*" else ""
                else:
                    parts = feat.split(",") if feat else []
                    p = parts[0] if parts else ""
                    pos_subs = [f for f in parts[1:4] if f and f != "*"]
                    ct = parts[4] if len(parts) > 4 and parts[4] != "*" else ""

                if p in _CONTENT_POS:
                    content_count += 1
                    if not first_pos:
                        first_pos = p
                        conj_type = ct
                        first_pos_subs = pos_subs
                    if (p == "接頭詞"
                            or p in _EXCLUDED_DISTRACTOR_POS
                            or any(b in s for s in pos_subs for b in _BOUND_POS_SUBS)):
                        is_bound = True
                elif p not in ("助詞", "助動詞", "記号"):
                    content_count = 99
                    break

            if content_count != 1 or is_bound:
                counts["skipped"] += 1
                continue
            if first_pos in _EXCLUDED_DISTRACTOR_POS:
                counts["skipped"] += 1
                continue

            if first_pos == "名詞" and morphemes:
                feat0 = morphemes[0].feature
                if isinstance(feat0, tuple):
                    base = feat0[7] if len(feat0) > 7 else ""
                else:
                    parts0 = feat0.split(",")
                    base = parts0[7] if len(parts0) > 7 else ""
                if (all('\u3041' <= ch <= '\u3096' for ch in word)
                        and len(word) <= 3
                        and base and base[-1] in "りしびみきちいえめ"):
                    counts["skipped"] += 1
                    continue

            if len(morphemes) > 1:
                last_feat = morphemes[-1].feature
                last_pos = (last_feat[0] if isinstance(last_feat, tuple)
                            else last_feat.split(",")[0])
                if last_pos == "助詞":
                    counts["skipped"] += 1
                    continue

            is_concrete = not any(
                a in s for s in first_pos_subs for a in _ABSTRACT_NOUN_SUBS
            )

            if first_pos == "名詞":
                for particle in _CASE_PARTICLES:
                    entry = _BunsetsuEntry(
                        word + particle, "noun", particle, logfreq,
                        concrete=is_concrete,
                    )
                    self._entries.append(entry)
                    self._by_particle[particle].append(entry)
                    self._by_pos.setdefault("noun", []).append(entry)
                    counts["noun"] += 1

            elif first_pos == "動詞":
                if word and word[-1] in _VERB_DICT_TAILS:
                    entry = _BunsetsuEntry(word, "verb", "", logfreq)
                    self._entries.append(entry)
                    self._by_particle[""].append(entry)
                    self._by_pos.setdefault("verb", []).append(entry)
                    counts["verb"] += 1
                past = _conjugate_past(word, conj_type)
                if past and past[-1] in _VERB_PAST_TAILS:
                    entry = _BunsetsuEntry(past, "verb", "", logfreq)
                    self._entries.append(entry)
                    self._by_particle[""].append(entry)
                    self._by_pos.setdefault("verb", []).append(entry)
                    counts["verb"] += 1

            elif first_pos == "形容詞":
                if word and word[-1] in _ADJ_TAILS:
                    entry = _BunsetsuEntry(word, "adj", "", logfreq)
                    self._entries.append(entry)
                    self._by_particle[""].append(entry)
                    self._by_pos.setdefault("adj", []).append(entry)
                    counts["adj"] += 1
                if word and word.endswith("い"):
                    entry = _BunsetsuEntry(word[:-1] + "かった", "adj", "", logfreq)
                    self._entries.append(entry)
                    self._by_particle[""].append(entry)
                    self._by_pos.setdefault("adj", []).append(entry)
                    counts["adj"] += 1

            elif first_pos == "形容動詞":
                for suffix in ("だ", "だった"):
                    na_text = word + suffix
                    if not _SMALL_KANA_START.match(na_text):
                        entry = _BunsetsuEntry(na_text, "adj", "", logfreq)
                        self._entries.append(entry)
                        self._by_particle[""].append(entry)
                        self._by_pos.setdefault("adj", []).append(entry)
                        counts["adj"] += 1

            elif first_pos == "副詞":
                entry = _BunsetsuEntry(word, "adv", "", logfreq)
                self._entries.append(entry)
                self._by_particle[""].append(entry)
                self._by_pos.setdefault("adv", []).append(entry)

        self._build_freq_bins()

        print(f"  {len(self._entries):,} bunsetsu total  "
              f"({len(self._by_freq_bin)} freq bins)")
        print(f"    noun: {counts['noun']:,}  verb: {counts['verb']:,}"
              f"  adj: {counts['adj']:,}  skipped: {counts['skipped']:,}")
        for p in _CASE_PARTICLES:
            print(f"    particle '{p}': {len(self._by_particle[p]):,}")

        try:
            with open(cache_path, "wb") as f:
                pickle.dump({
                    "entries": self._entries,
                    "by_particle": self._by_particle,
                    "by_pos": self._by_pos,
                    "by_freq_bin": self._by_freq_bin,
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"  cached → {os.path.basename(cache_path)}")
        except Exception as exc:
            logging.warning("cache save failed: %s", exc)


    def _build_freq_bins(self):
        """Index all entries into frequency bins for O(1) lookup."""
        self._by_freq_bin = {}
        for e in self._entries:
            self._by_freq_bin.setdefault(_freq_bin(e.freq), []).append(e)

    def _freq_bin_entries(self, target_freq: float) -> List[_BunsetsuEntry]:
        """Return entries from ±RADIUS bins."""
        center = _freq_bin(target_freq)
        out: List[_BunsetsuEntry] = []
        for b in range(center - _FREQ_BIN_RADIUS, center + _FREQ_BIN_RADIUS + 1):
            out.extend(self._by_freq_bin.get(b, []))
        return out


    def get_violating_distractors(
        self,
        prior_particles: List[str],
        is_final: bool,
        target_pos: str,
        min_length: int,
        max_length: int,
        params: Dict,
        banned: Optional[List[str]] = None,
        target_freq: float = 0.0,
    ) -> List[tuple]:
        """Return (bunsetsu, tier) pairs that create grammatical violations.

        Tiers (lower = stronger, more preferred):
          1  Strong duplicate particle (が, を) — categorical
          2  Weak duplicate particle (に, で, は) / POS mismatch
        No tier-3 fallback: positions with no tier-1/2 candidate emit x-x-x.
        """
        banned_set = frozenset(banned) if banned else frozenset()
        n = params.get("num_to_test", 80)
        freq_entries = self._freq_bin_entries(target_freq)

        seen: set = set()
        tier1: List[tuple] = []
        rest:  List[tuple] = []

        def _ok(e: _BunsetsuEntry) -> bool:
            # Adverbs are too syntactically flexible to be strong distractors
            # in Japanese — they can float into almost any slot without
            # producing a clean violation.  Exclude globally.
            if e.pos == "adv":
                return False
            return (min_length <= e.length <= max_length
                    and e.text not in banned_set
                    and e.text not in seen)

        # Tier 1: strong duplicate particles only (が, を).
        used_strong = [p for p in prior_particles if p in _STRONG_DUPLICATE_PARTICLES]
        if used_strong:
            for e in freq_entries:
                if e.particle in used_strong and _ok(e):
                    tier1.append((e.text, 1))
                    seen.add(e.text)

        # Only fall through to weaker strategies if tier 1 is sparse.
        if len(tier1) < max(n // 2, 15):
            # Tier 2a: weak duplicate particles (に, で, は).
            used_weak = [p for p in prior_particles
                         if p in _CASE_PARTICLES and p not in _STRONG_DUPLICATE_PARTICLES]
            if used_weak:
                for e in freq_entries:
                    if e.particle in used_weak and _ok(e):
                        rest.append((e.text, 2))
                        seen.add(e.text)

            # Tier 2b: any POS mismatch — covers verb/noun/adj/other slots.
            for e in freq_entries:
                if e.pos != target_pos and _ok(e):
                    rest.append((e.text, 2))
                    seen.add(e.text)

        def _safe(text: str) -> bool:
            if _SMALL_KANA_START.match(text):
                return False
            stem = RepeatCounter._extract_stem(text)
            if _SMALL_TSU_END.search(stem):
                return False
            return stem not in _BLOCKED_WORDS and text not in _BLOCKED_WORDS

        random.shuffle(tier1)
        random.shuffle(rest)
        pool = [(t, r) for t, r in tier1 + rest if _safe(t)]
        return pool[:n]


_JA_FREQ_DICT: Dict[str, float] = {}


def _ensure_freq_dict():
    """Lazily load the wordfreq Japanese dictionary into _JA_FREQ_DICT."""
    global _JA_FREQ_DICT
    if not _JA_FREQ_DICT:
        raw = wordfreq.get_frequency_dict("ja")
        _JA_FREQ_DICT = {w: math.log(p * 1e9) for w, p in raw.items() if p > 0}


def get_frequency_ja(word: str) -> float:
    """Return the log-frequency of *word* (default -15.0 for unknown)."""
    _ensure_freq_dict()
    return _JA_FREQ_DICT.get(word, -15.0)


class Sentence:
    """A single stimulus sentence split into bunsetsu with integer labels."""

    def __init__(self, words: List[str], labels: List, item_id: str, tag: str):
        if not _no_duplicates(labels):
            raise ValueError(f"Duplicate labels in item {item_id}: {labels}")
        self.words = words
        self.labels = labels
        self.id = item_id
        self.tag = tag
        self.distractors: List[str] = ["x-x-x"]   # first position always x-x-x
        self.distractor_sentence: str = ""
        self.probs: Dict = {}        # label → prefix context (token IDs)
        self.surprisal: Dict = {}    # label → surprisal (bits)

    @property
    def word_sentence(self) -> str:
        return " ".join(self.words)

    @property
    def label_sentence(self) -> str:
        return " ".join(str(l) for l in self.labels)

    def do_model(self, model: JapaneseGPT2LM):
        """Build prefix contexts for each label position."""
        state = model.empty_sentence()
        for i in range(len(self.words) - 1):
            self.probs[self.labels[i + 1]] = state
            state = model.update(state, self.words[i])

    def do_surprisal(self, model: JapaneseGPT2LM):
        """Compute surprisal of each real word given its prefix."""
        for i in range(1, len(self.labels)):
            lab = self.labels[i]
            self.surprisal[lab] = model.get_surprisal(self.probs[lab], self.words[i])


class Label:
    """Aggregated data for one label position across all sentences of an item."""

    _LM_WAVE_SIZE = 16
    _HEURISTIC_ACCEPT = 3

    def __init__(self, item_id: str, lab):
        self.id = item_id
        self.lab = lab
        self.words: List[str] = []
        self.probs: List[tuple] = []
        self.surprisals: List[float] = []
        self.distractor: str = "x-x-x"

    def add_sentence(self, word: str, context_ids: tuple, surprisal: float):
        """Register a sentence's word/context/surprisal for this label."""
        self.words.append(word)
        self.probs.append(context_ids)
        self.surprisals.append(surprisal)

    def choose_distractor(
        self,
        model: JapaneseGPT2LM,
        dist_dict: JapaneseDistractorDict,
        params: Dict,
        banned: List[str],
        prior_particles: List[str],
        is_final: bool,
        target_pos: str,
    ) -> str:
        """Select the best distractor for this label position."""
        if all(not _JP_RE.search(w) for w in self.words):
            self.distractor = "x-x-x"
            return "x-x-x"

        lengths = [len(w) for w in self.words]
        min_len = max(1, min(lengths) - 1)
        max_len = max(lengths) + 3

        _stems = [RepeatCounter._extract_stem(w) for w in self.words]
        _freqs = [get_frequency_ja(s) for s in _stems if s]
        _freqs = [f for f in _freqs if f > -15.0]
        target_freq = sum(_freqs) / len(_freqs) if _freqs else 10.0

        candidates = dist_dict.get_violating_distractors(
            prior_particles, is_final, target_pos,
            min_len, max_len, params, banned,
            target_freq=target_freq,
        )

        banned_set = frozenset(banned)
        cand_pairs = [(c, tier) for c, tier in candidates if c not in banned_set]

        # Sort tier-first (stronger violations before weaker), then by
        # freq proximity with jitter and katakana bonus within each tier.
        cand_pairs.sort(key=lambda ct: (
            ct[1],
            abs(get_frequency_ja(ct[0]) - target_freq)
            + random.random() * 0.05
            + (_KATAKANA_BONUS if _is_katakana_heavy(ct[0]) else 0.0)
        ))
        cand_list = [c for c, _ in cand_pairs]
        if not cand_list:
            self.distractor = "x-x-x"
            return "x-x-x"

        target_punct = ""
        for w in self.words:
            m = _PUNCT_FINAL.search(w)
            if m:
                target_punct = m.group()
                break

        def _apply_punct(word: str) -> str:
            if target_punct and not _PUNCT_FINAL.search(word):
                return word + target_punct
            return word

        for cand in cand_list:
            if abs(get_frequency_ja(cand) - target_freq) <= 1.5:
                self.distractor = _apply_punct(cand)
                return self.distractor

        if cand_list:
            self.distractor = _apply_punct(cand_list[0])
            return self.distractor

        _TOP_K_PICK = 5
        min_pmi       = params["min_pmi"]
        min_abs_floor = params["min_abs"]
        best_word     = "x-x-x"
        best_min_pmi  = -float("inf")
        wave          = self._LM_WAVE_SIZE
        passed: List[str] = []

        for start in range(0, len(cand_list), wave):
            batch = cand_list[start : start + wave]

            penalties = [
                _BORING_PENALTY if RepeatCounter._extract_stem(c) in _BORING_WORDS
                else 0.0
                for c in batch
            ]
            uni_bits     = [_unigram_bits(get_frequency_ja(c)) for c in batch]
            min_pmi_seen = [float("inf")] * len(batch)
            alive = list(range(len(batch)))

            for ctx in self.probs:
                if not alive:
                    break
                alive_batch = [batch[i] for i in alive]
                row = model.get_surprisals_batch(ctx, alive_batch)

                next_alive: List[int] = []
                for local_idx, orig_idx in enumerate(alive):
                    s   = row[local_idx]
                    pmi = s - uni_bits[orig_idx]
                    if pmi < min_pmi_seen[orig_idx]:
                        min_pmi_seen[orig_idx] = pmi
                    if s > _SURPRISAL_CEILING:
                        continue                         # gibberish
                    if pmi < min_pmi + penalties[orig_idx]:
                        continue                         # context too weak
                    if s < min_abs_floor:
                        continue                         # absolute backstop
                    next_alive.append(orig_idx)
                alive = next_alive

            for orig_idx in alive:
                passed.append(batch[orig_idx])
                if len(passed) >= _TOP_K_PICK:
                    break

            for orig_idx, mp in enumerate(min_pmi_seen):
                if mp != float("inf") and mp > best_min_pmi:
                    best_min_pmi = mp
                    best_word = batch[orig_idx]

            if len(passed) >= _TOP_K_PICK:
                break

        if passed:
            chosen = random.choice(passed)
            self.distractor = _apply_punct(chosen)
            return self.distractor

        logging.warning(
            "PMI threshold not met for %s/%s; best='%s' (PMI=%.1f bits)",
            self.id, self.lab, best_word, best_min_pmi,
        )
        self.distractor = _apply_punct(best_word)
        return self.distractor


class SentenceSet:
    """All sentences sharing one item ID.  Owns label aggregation and"""

    def __init__(self, item_id: str):
        self.id = item_id
        self.sentences: List[Sentence] = []
        self.label_ids: set = set()
        self.first_labels: set = set()
        self.labels: Dict[object, Label] = {}

    def add(self, sentence: Sentence):
        assert sentence.id == self.id
        self.sentences.append(sentence)
        self.first_labels.add(sentence.labels[0])
        self.label_ids.update(sentence.labels[1:])
        if self.first_labels & self.label_ids:
            raise ValueError(f"Item {self.id}: overlap in first/later labels.")

    def do_model(self, model: JapaneseGPT2LM):
        """Build prefix contexts for every sentence."""
        for s in self.sentences:
            s.do_model(model)

    def do_surprisals(self, model: JapaneseGPT2LM):
        """Compute real-word surprisals for every sentence."""
        for s in self.sentences:
            s.do_surprisal(model)

    def make_labels(self):
        """Create Label objects and aggregate cross-sentence data."""
        for lab in self.label_ids:
            self.labels[lab] = Label(self.id, lab)
        for s in self.sentences:
            for i in range(1, len(s.labels)):
                lab = s.labels[i]
                self.labels[lab].add_sentence(
                    s.words[i], s.probs[lab], s.surprisal[lab]
                )

    def do_distractors(
        self,
        model: JapaneseGPT2LM,
        dist_dict: JapaneseDistractorDict,
        tokenizer: "Optional[JapaneseTokenizer]",
        params: Dict,
        repeats: "RepeatCounter",
    ):
        """Select distractors for every label, clause-aware."""
        banned = repeats.banned[:]

        first = self.sentences[0] if self.sentences else None
        label_ctx: Dict = {}
        if first and tokenizer:
            running_particles: List[str] = []
            for i, w in enumerate(first.words):
                lab = first.labels[i]
                info = tokenizer.get_bunsetsu_info(w)

                if info["pos"] in ("verb", "adj") and i < len(first.words) - 1:
                    running_particles = []

                if lab in self.label_ids:
                    label_ctx[lab] = {
                        "prior": list(running_particles),
                        "is_final": (i == len(first.words) - 1),
                        "pos": info["pos"],
                    }
                p = info["particle"]
                for cp in sorted(_ALL_PARTICLES, key=len, reverse=True):
                    if p.startswith(cp):
                        running_particles.append(cp)
                        break

        sentence_banned: set = set()

        for label in self.labels.values():
            ctx = label_ctx.get(label.lab, {})
            dist = label.choose_distractor(
                model, dist_dict, params,
                banned + list(sentence_banned),
                prior_particles=ctx.get("prior", []),
                is_final=ctx.get("is_final", False),
                target_pos=ctx.get("pos", "other"),
            )
            stem = RepeatCounter._extract_stem(dist)
            sentence_banned.add(dist)
            sentence_banned.add(stem)
            for p in _CASE_PARTICLES:
                sentence_banned.add(stem + p)

            banned.append(dist)
            repeats.increment(dist)

        for s in self.sentences:
            for i in range(1, len(s.labels)):
                lab = s.labels[i]
                s.distractors.append(self.labels[lab].distractor)
            s.distractor_sentence = " ".join(s.distractors)

    def clean_up(self):
        """Free memory after distractor selection is complete."""
        self.labels = {}
        for s in self.sentences:
            s.probs = {}


class RepeatCounter:
    """Track distractor usage across the full run for variety enforcement."""

    def __init__(self, max_repeats: int = 0):
        self.max = max_repeats
        self.limit = max_repeats > 0
        self._counts: Dict[str, int] = {}
        self._stem_counts: Dict[str, int] = {}
        self.banned: List[str] = []

    @staticmethod
    def _extract_stem(word: str) -> str:
        """Strip trailing punctuation and case particle to get the content stem."""
        w = word
        while w and w[-1] in "。！？":
            w = w[:-1]
        for p in sorted(_CASE_PARTICLES, key=len, reverse=True):
            if w.endswith(p) and len(w) > len(p):
                return w[:-len(p)]
        return w

    def _ban_stem_variants(self, stem: str):
        """Ban the stem itself and all its particle variants."""
        for form in [stem] + [stem + p for p in _CASE_PARTICLES]:
            if form not in self.banned:
                self.banned.append(form)

    def increment(self, word: str):
        """Record one use of *word* and ban if cap is reached."""
        self._counts[word] = self._counts.get(word, 0) + 1
        if self.limit and self._counts[word] >= self.max:
            if word not in self.banned:
                self.banned.append(word)

        stem = self._extract_stem(word)
        self._stem_counts[stem] = self._stem_counts.get(stem, 0) + 1

        cap = _FORMAL_NOUN_CAP if stem in _FORMAL_NOUNS else _STEM_VARIETY_CAP
        if self._stem_counts[stem] >= cap:
            self._ban_stem_variants(stem)


def read_input(
    filename: str,
    tokenizer: Optional[JapaneseTokenizer] = None,
) -> Dict[str, SentenceSet]:
    """Read semicolon-delimited input file into SentenceSet objects."""
    all_sets: Dict[str, SentenceSet] = {}
    with open(filename, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";", quotechar='"')
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            tag     = row[0].strip()
            item_id = row[1].strip()
            sent    = row[2].strip()

            if " " in sent:
                words = sent.split()
            else:
                if tokenizer is None:
                    tokenizer = JapaneseTokenizer()
                words = tokenizer.tokenize(sent)

            if len(row) > 3 and row[3].strip():
                labels = row[3].strip().split()
                if len(labels) != len(words):
                    raise ValueError(
                        f"Label count ({len(labels)}) != word count ({len(words)}) "
                        f"for item {item_id}"
                    )
            else:
                labels = list(range(len(words)))

            if item_id not in all_sets:
                all_sets[item_id] = SentenceSet(item_id)
            all_sets[item_id].add(Sentence(words, labels, item_id, tag))
    return all_sets


def save_delim(outfile: str, all_sets: Dict[str, SentenceSet]):
    """Write semicolon-delimited output: tag;id;words;distractors;labels."""
    with open(outfile, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        for ss in all_sets.values():
            for s in ss.sentences:
                writer.writerow([
                    s.tag, s.id, s.word_sentence,
                    s.distractor_sentence, s.label_sentence,
                ])


def save_ibex(outfile: str, all_sets: Dict[str, SentenceSet]):
    """Write Ibex-format output for online Maze experiments."""
    with open(outfile, "w", encoding="utf-8") as f:
        for ss in all_sets.values():
            for s in ss.sentences:
                f.write(
                    f'[["{s.tag}", {repr(s.id)}], "Maze", '
                    f'{{s:"{s.word_sentence}", a:"{s.distractor_sentence}"}}],\n'
                )


def load_params(params_file: Optional[str]) -> Dict:
    """Load runtime parameters from a key:value text file."""
    params = dict(DEFAULT_PARAMS)
    if params_file is None:
        return params
    with open(params_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, _, val = line.partition(":")
                try:
                    params[key.strip()] = ast.literal_eval(val.strip())
                except Exception:
                    params[key.strip()] = val.strip()
    return params


def run_maze_japanese(
    infile: str,
    outfile: str,
    params_file: Optional[str] = None,
    outformat: str = "delim",
    seed: Optional[int] = None,
):
    """End-to-end pipeline: load → model → distractors → write → report."""
    if outformat not in ("delim", "ibex"):
        raise ValueError(f"Unknown format: {outformat!r}")

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"Random seed set to {seed}")

    params    = load_params(params_file)
    tokenizer = JapaneseTokenizer() if _FUGASHI_AVAILABLE else None
    lm        = JapaneseGPT2LM(params.get("model_name", DEFAULT_MODEL_NAME))
    dist_dict = JapaneseDistractorDict(params, tokenizer=tokenizer)
    repeats   = RepeatCounter(params.get("max_repeat", 0))

    print("Reading input...")
    all_sets = read_input(infile, tokenizer)
    n_items = len(all_sets)
    n_sents = sum(len(ss.sentences) for ss in all_sets.values())
    print(f"  {n_sents} sentences across {n_items} items")

    t_start = time.time()
    for idx, (item_id, ss) in enumerate(all_sets.items(), 1):
        print(f"  [{idx}/{n_items}] item {item_id}")
        ss.do_model(lm)
        ss.do_surprisals(lm)
        ss.make_labels()
        ss.do_distractors(lm, dist_dict, tokenizer, params, repeats)
        ss.clean_up()
    t_elapsed = time.time() - t_start

    print(f"Writing {outformat} output to {outfile!r}...")
    if outformat == "ibex":
        save_ibex(outfile, all_sets)
    else:
        save_delim(outfile, all_sets)

    all_distractors = []
    for ss in all_sets.values():
        for lab in ss.labels.values():
            if lab.distractor and lab.distractor != "x-x-x":
                all_distractors.append(lab.distractor)
    n_dist = len(all_distractors)
    n_unique = len(set(all_distractors))
    stems = [RepeatCounter._extract_stem(d) for d in all_distractors]
    n_unique_stems = len(set(stems))
    katakana_count = sum(1 for d in all_distractors if _is_katakana_heavy(d))

    print(f"\n{'─' * 50}")
    print(f"  Diagnostics")
    print(f"{'─' * 50}")
    print(f"  Elapsed time:       {t_elapsed:.1f}s ({t_elapsed/max(n_sents,1):.2f}s/sentence)")
    print(f"  Total distractors:  {n_dist}")
    print(f"  Unique distractors: {n_unique} ({100*n_unique/max(n_dist,1):.0f}%)")
    print(f"  Unique stems:       {n_unique_stems} ({100*n_unique_stems/max(n_dist,1):.0f}%)")
    print(f"  Katakana words:     {katakana_count} ({100*katakana_count/max(n_dist,1):.0f}%)")
    print(f"  Pool size:          {len(dist_dict._entries):,} bunsetsu")
    print(f"{'─' * 50}")
    print("Done.")


def _val_particle(word: str) -> str:
    w = word
    while w and w[-1] in "。！？":
        w = w[:-1]
    for p in sorted(_CASE_PARTICLES, key=len, reverse=True):
        if w.endswith(p) and len(w) > len(p):
            return p
    return ""


def _val_pos(word: str) -> str:
    stem = RepeatCounter._extract_stem(word)
    if _val_particle(word):
        return "noun_particle"
    if not stem:
        return "other"
    last = stem[-1]
    if last in "ただ" and len(stem) >= 2:
        return "verb"
    if last in "うくぐすつぬぶむる":
        return "verb"
    if last == "い" and len(stem) >= 2:
        return "adj"
    return "other"


def _val_classify(word: str, distractor: str, prior_words: List[str]) -> str:
    d_p = _val_particle(distractor)
    w_pos, d_pos = _val_pos(word), _val_pos(distractor)
    priors = {_val_particle(w) for w in prior_words} - {""}
    if d_p and d_p in priors:
        return "STRONG" if d_p in _STRONG_DUPLICATE_PARTICLES else "MEDIUM"
    if w_pos != d_pos and w_pos != "other" and d_pos != "other":
        return "MEDIUM"
    if w_pos == d_pos:
        return "WEAK"
    return "UNKNOWN"


def validate_output(output_file: str, verbose: bool = False) -> int:
    """Validate G-Maze distractor output quality. Returns 0 on pass, 1 on issues."""
    rows = []
    with open(output_file, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter=";", quotechar='"'):
            if row and not row[0].startswith("#"):
                rows.append(row)
    if not rows:
        print("ERROR: No data rows found.")
        return 1

    all_d, issues, weak_samples = [], [], []
    bound_leaks, nsfw_leaks, sokuon_leaks, singlechar = [], [], [], []
    punct_fails = 0
    strength = Counter()

    for row in rows:
        if len(row) < 4:
            issues.append(f"Malformed row: {row}")
            continue
        tag, iid = row[0].strip(), row[1].strip()
        words = row[2].strip().split()
        dists = row[3].strip().split()
        if len(words) != len(dists):
            issues.append(f"  {tag};{iid}: word/distractor count mismatch")
        for i, (w, d) in enumerate(zip(words, dists)):
            if len(w) == 1 and _JP_RE.match(w) and not _KATAKANA_RE.match(w):
                singlechar.append(f"{tag};{iid} pos {i}: word='{w}'")
            if d == "x-x-x":
                continue
            all_d.append(d)
            stem = RepeatCounter._extract_stem(d)
            if bool(_PUNCT_FINAL.search(w)) != bool(_PUNCT_FINAL.search(d)):
                punct_fails += 1
                if verbose:
                    issues.append(f"  {tag};{iid} pos {i}: punct mismatch '{w}' vs '{d}'")
            if stem in _MORPHOLOGICAL_BLOCKED:
                bound_leaks.append(f"{tag};{iid} pos {i}: '{d}' (stem='{stem}')")
            if _SMALL_TSU_END.search(stem):
                sokuon_leaks.append(f"{tag};{iid} pos {i}: '{d}' (stem='{stem}')")
            if stem in _BLOCKED_WORDS and stem not in _MORPHOLOGICAL_BLOCKED:
                nsfw_leaks.append(f"{tag};{iid} pos {i}: '{d}' (stem='{stem}')")
            s = _val_classify(w, d, words[:i])
            strength[s] += 1
            if s == "WEAK" and len(weak_samples) < 10:
                weak_samples.append(f"  {tag};{iid} pos {i}: '{w}' → '{d}'")

    n = len(all_d)
    if n == 0:
        print("WARNING: No non-x-x-x distractors.")
        return 1

    stems = [RepeatCounter._extract_stem(d) for d in all_d]
    kata = sum(1 for d in all_d if _is_katakana_heavy(d))
    top_stems = Counter(stems).most_common(10)
    strong, medium = strength.get("STRONG", 0), strength.get("MEDIUM", 0)
    weak, unknown = strength.get("WEAK", 0), strength.get("UNKNOWN", 0)

    print(f"\n{'═' * 60}")
    print(f"  G-Maze Output Validation: {output_file}")
    print(f"{'═' * 60}")
    print(f"  Sentences:          {len(rows)}")
    print(f"  Distractors:        {n}  (excluding x-x-x)")
    print(f"  Unique distractors: {len(set(all_d))}  ({100*len(set(all_d))/n:.0f}%)")
    print(f"  Unique stems:       {len(set(stems))}  ({100*len(set(stems))/n:.0f}%)")
    print(f"  Katakana words:     {kata}  ({100*kata/n:.0f}%)\n")
    print(f"  Violation strength:")
    print(f"      STRONG  (double-が/を)    {strong:4d}  ({100*strong/n:.0f}%)")
    print(f"      MEDIUM  (weak-dup / POS)  {medium:4d}  ({100*medium/n:.0f}%)")
    print(f"      WEAK    (likely paraphrase){weak:4d}  ({100*weak/n:.0f}%)")
    print(f"      UNKNOWN                    {unknown:4d}  ({100*unknown/n:.0f}%)\n")

    ok = True
    for label, items, sym in [
        ("Punctuation sync", [f"fails: {punct_fails}"] if punct_fails else [], "⚠"),
        ("Sokuon fragments", sokuon_leaks, "✗"),
        ("Bound morphemes", bound_leaks, "⚠"),
        ("Content safety", nsfw_leaks, "✗"),
        ("Tokenization (single-char content)", singlechar, "⚠"),
    ]:
        if items:
            print(f"  {sym} {label}: {len(items)}")
            for x in items[:5]:
                print(f"      {x}")
            if len(items) > 5:
                print(f"      ... and {len(items)-5} more")
            ok = False
        else:
            print(f"  ✓ {label}: clean")

    if weak + unknown > n * 0.3:
        print(f"  ⚠ High weak-violation ratio: {100*(weak+unknown)/n:.0f}%")
        for ws in weak_samples[:5]:
            print(ws)
        ok = False
    else:
        print(f"  ✓ Violation strength: acceptable distribution")

    print(f"\n  Top 10 repeated stems:")
    for s, c in top_stems:
        print(f"      {s:12s}  ×{c}")
    if issues:
        print(f"\n  Other issues ({len(issues)}):")
        for iss in issues[:20]:
            print(f"    {iss}")
    print(f"\n{'═' * 60}")
    print("  PASS — no quality issues detected" if ok else "  WARN — issues found")
    print(f"{'═' * 60}\n")
    return 0 if ok else 1


def _cli():
    ap = argparse.ArgumentParser(
        description="Generate or validate G-Maze distractors for Japanese stimuli")
    ap.add_argument("input",  nargs="?", default="input.txt",
                    help="Input file (default: input.txt). Ignored with --validate.")
    ap.add_argument("output", nargs="?",
                    help="Output file path (for generate); or file to check with --validate")
    ap.add_argument("--validate", action="store_true",
                    help="Validate an existing output file instead of generating")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Show per-position detail (validate mode)")
    ap.add_argument("-p", "--params", default=None,
                    help="Parameters file (key:value format)")
    ap.add_argument("--format", choices=["delim", "ibex"], default="delim",
                    help="Output format (default: delim)")
    ap.add_argument("--log", default="WARNING",
                    help="Logging level (DEBUG, INFO, WARNING)")
    ap.add_argument("--seed", type=int, default=None,
                    help="Random seed for reproducible runs")
    args = ap.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log.upper(), logging.WARNING),
        format="%(levelname)s: %(message)s",
    )
    if args.validate:
        target = args.output or args.input
        if not target:
            ap.error("--validate requires a file argument")
        import sys
        sys.exit(validate_output(target, verbose=args.verbose))
    if not args.output:
        ap.error("output file required")
    run_maze_japanese(
        args.input, args.output,
        params_file=args.params,
        outformat=args.format,
        seed=args.seed,
    )


if __name__ == "__main__":
    _cli()
