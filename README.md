# Japanese A-Maze

A surprisal-based distractor generator for the G-Maze psycholinguistic task in Japanese.
Inspired by the English/French A-Maze of Boyce, Futrell & Levy (2020); rebuilt around Japanese-specific structural constraints (particle clashes, POS-mismatch by slot type, bunsetsu-level segmentation) and an autoregressive language model (`rinna/japanese-gpt2-medium`) for surprisal scoring.

## What's new (2026-11): factorial design support

The distractor strategy is unchanged; the changes are in input/output handling, cross-condition sharing, and multiple candidates:

1. **Input is now `input.csv` with `item`, `condition` and `sentence` columns.**
   Stimuli from a factorial design can be written as they are. Rows with the same `item` are the conditions of one item.
2. **Distractors are shared across conditions word by word.**
   The same bunsetsu gets the same distractor in every condition of an item, even when the word order is scrambled. A shared distractor is chosen only from candidates that are ungrammatical in the context of *every* condition (the original only looked at the first sentence of an item).
3. **2nd and 3rd candidates.**
   For each sentence the output lists the 1st, 2nd and 3rd choice distractor lines, so distractors can be swapped by hand. The number of lines is set with `n_candidates`.
4. **Output format designed for manual inspection** (the new default). The original format is still available with `--format delim`.
5. **`params.txt` next to the script is read automatically**, so settings can be changed without command-line options.
6. **`grade_level` now defaults to 0** (no kanji or frequency filter). See [About `grade_level`](#about-grade_level).

> **Note:** The input and output formats changed with this revision, so input files written for the old version will not work as they are.

## Quick start
```bash
# 1. Install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install sentencepiece protobuf

# 2. Generate distractors
python maze_japanese.py input.csv output.txt --seed 42

# 3. Validate the output
python maze_japanese.py --validate output.txt
```
The first run takes a few minutes: it builds and caches the bunsetsu pool (`bunsetsu_pool_v21_g<grade_level>.pkl`, one file per `grade_level`) and downloads the LM (~1.2 GB). Later runs load the cache and start faster.

---

## Input format (input.csv)

Comma-separated, with the columns `item, condition, sentence` and an optional `labels` column. The header row is optional (without it, columns are read in that order). UTF-8 (including Excel's "CSV UTF-8") and Shift-JIS are both accepted.

```
item,condition,sentence
1,a,学生は 図書館で 本を 読んだ。
1,b,学生は 公園で 本を 読んだ。
2,c,派遣社員が 昨日 ゆっくり おこなった 業務は それほど 難しくない。
2,d,昨日 ゆっくり 派遣社員が おこなった 業務は それほど 難しくない。
```

- Bunsetsu in the sentence are separated by half-width spaces; you can pre-segment or let MeCab (UniDic) do it.
- Within an item, bunsetsu with the same text get the same distractor, wherever they appear. Bunsetsu that differ between conditions (図書館で / 公園で above) get their own distractors.
- The sentence-initial bunsetsu is always `x-x-x`. If a word is sentence-initial in one condition but not in another (派遣社員が above), it gets a real distractor in the condition where it is not initial.
- To control sharing yourself, fill in the `labels` column for every condition of the item: one label per bunsetsu, separated by spaces. Bunsetsu with the same label share a distractor (e.g. `0 9 1 2`).

`input.csv` in this repository is a sample with 4 items × 4 conditions (genitive の vs. nominative が subject, canonical vs. scrambled word order), written without a header row.

---

## Output format

Default (`--format inspect`): a header line, the sentence, and `n_candidates` distractor lines (1st, 2nd, 3rd choice). All lines start at the beginning of the line, so each distractor sits under its bunsetsu.

```
# item=1 condition=a
学生は 図書館で 本を 読んだ。
x-x-x 解消が 以外が 上回る。;0 1 2 3
x-x-x 改善が 予定が 下回る。;0 1 2 3
x-x-x 処理が 内部が 立ち去る。;0 1 2 3
```
(The distractors above are illustrative.)

- The first distractor line is the 1st choice. `--validate`, `--format delim` and `--format ibex` use this line.
- 2nd and 3rd choices are the next best candidates in the same ranking. Each differs in stem from every 1st choice of the item and from the other words on its own line. They are not added to the run-wide repeat list, so they do not change the 1st choices.
- `x-x-x` means no distractor was emitted at that position (position 0 is always `x-x-x`). In a 2nd/3rd line it means no further candidate was available.
- `*word` (asterisk-prefixed) means the distractor is a **fallback**: it bypassed the strict rules and should be reviewed manually before use.

`--format delim` writes the original one-line format, `condition ; item ; sentence ; distractors ; label_indices`, and `--format ibex` writes Ibex experiment items. Both contain the 1st choice only.

---

## Distractor strategy

| Target slot | Distractor type | Reason |
|---|---|---|
| Nominal (noun + particle) | Noun stem + structurally improbable particle | Duplicate of a prior particle, or `が`/`は` after any prior |
| Non-final verb | Same — noun + improbable particle | Substitutes a non-predicate for a predicate, plus particle clash |
| Sentence-final | Verb or adjective | Sentence ends with a predicate, but contextually wrong / wrong argument structure |
| Adjective / adverb (not after が/は) | Verb intrusion | Finite verb breaks the modification structure |
| Adjective / adverb after が/は | (banned) — falls back to noun + improbable particle | Prevents distractor reading as a relative clause or attributive modifier |

Each candidate is then scored for **LM surprisal** in the actual sentence prefix; only candidates whose surprisal is between `min_surprisal` and `surprisal_ceiling` are kept; the most surprising one wins.

When a distractor is shared by several conditions, the particle rules are checked against the context of every condition (only particles improbable in all of them are used), and the LM surprisal is the minimum over all the conditions' prefixes.

If the strict rule yields no candidates (or if no candidate passes the surprisal gate), the **fallback** strategy emits the most plausible-implausible candidate from a broader pool, marked with `*`.

---

## CLI

```bash
python maze_japanese.py [-h] [--version] [--validate] [-v] [-p PARAMS]
                       [--format {inspect,delim,ibex}] [--log LEVEL] [--seed N]
                       [input] [output]
```

| Flag | Description |
|---|---|
| `input` | Input CSV (default `input.csv`). Ignored with `--validate`. |
| `output` | Output file path (generate mode); file to check (validate mode). |
| `--validate` | Run the validator instead of generating. Reads both the inspect and delim formats. |
| `-v` / `--verbose` | More detail in validator output. |
| `-p / --params` | Use this parameters file instead of `params.txt` next to the script. |
| `--format` | `inspect` (default), `delim` (original format) or `ibex` (Ibex experiment format). |
| `--seed` | Random seed for reproducibility. |
| `--log` | `DEBUG`, `INFO`, `WARNING`. |
| `--version` | Print version. |

---

## Parameters (params.txt)

Settings are read from `params.txt`, which must be in the **same folder as `maze_japanese.py`**. The script prints which file it used. If `params.txt` is missing, the defaults below are used. To use another file for a particular run, pass it with `-p`.

The format is a list of `key: value` pairs; keys left out keep their defaults:

```
num_to_test:        80      # max candidates per slot considered for LM scoring
min_surprisal:      14.0    # bits — min LM surprisal for distractor acceptance
surprisal_ceiling:  35.0    # bits — above this, candidate is LM-OOV (skip)
freq_match_band:    2.0     # log-freq tolerance for matching target word
max_repeat:         0       # max reuses of the same exact form (0 = stem-cap only)
model_name:         "rinna/japanese-gpt2-medium"
grade_level:        0       # 0 = off; 1–6 = elementary kanji + frequency filter
use_lm:             True    # turn off to use frequency-matching only (no LM)
n_candidates:       3       # distractor lines per sentence in the inspect output
```

To reduce `x-x-x` count: lower `min_surprisal`, raise `freq_match_band`, or increase `num_to_test`.

### About `grade_level`

`grade_level` combines two filters that are applied when the bunsetsu pool is built.

**1. Kanji (based on the official grade-level kanji list).**
Words containing a kanji not taught by the given elementary school grade are excluded. The built-in table (`_KYOIKU_KANJI` in the script) follows the *Gakunen-betsu Kanji Haitōhyō* of the Course of Study, apparently the 2017 revision in force since 2020 (it includes the prefecture-name kanji such as 茨, 媛 and 岡 in grade 4).
The built-in table does **not** match the official list exactly:

| Grade | 1 | 2 | 3 | 4 | 5 | 6 | Total |
|---|---|---|---|---|---|---|---|
| Built-in table | 80 | 160 | 199 | 205 | 184 | 182 | 1010 |
| Official (2020) | 80 | 160 | 200 | 202 | 193 | 191 | 1026 |

To use an exact list, place a file named `kanji_grades.txt` in the same folder as the script, with the kanji for grade 1 on line 1, grade 2 on line 2, and so on. It is used instead of the built-in table.

**2. Word frequency (a heuristic of this script, not from the Course of Study).**
Words below a minimum frequency are excluded. The thresholds are the author's choice; no source is given in the code.

| `grade_level` | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| Minimum log-frequency | none | 10.0 | 9.0 | 8.0 | 7.0 | 6.0 | 5.0 |
| ≈ occurrences per million words | none | 22 | 8.1 | 3.0 | 1.1 | 0.40 | 0.15 |

Words written only in kana are not affected by the kanji filter, so at low grade levels the frequency threshold is what actually limits their difficulty.

With `grade_level: 0` (the default) both filters are off: any word in the wordfreq Japanese list can enter the pool, so the first pool build takes longer. Distractors are still matched to the frequency of the target word (within ±1.0 log-frequency), so a common target word will not get an obscure distractor.

---

## Validation report

`python maze_japanese.py --validate output.txt` produces:

```
════════════════════════════════════════════════════════════
  G-Maze Output Validation: output.txt
════════════════════════════════════════════════════════════
  Sentences:          75
  Distractors:        179  (excluding x-x-x)
  Unique distractors: 172  (96%)
  Unique stems:       169  (94%)
  Katakana words:     18  (10%)
  Fallback (*-marked):  3  (2%)  [manual review recommended]

  Violation strength:
      STRONG  (double-が/を)     22  (12%)
      MEDIUM  (weak-dup / POS)  134  (75%)
      WEAK    (likely paraphrase)22  (12%)
      UNKNOWN                     1  (1%)

  ✓ Punctuation sync: clean
  ✓ Sokuon fragments: clean
  ✓ Bound morphemes:  clean
  ✓ Content safety:   clean
  ✓ Tokenization:     clean
  ✓ Violation strength: acceptable distribution
```
(Example from the old version.)

Violation strength classes:
- **STRONG** — categorical double-が/を particle clash.
- **MEDIUM** — weak particle duplication (に/で/は), or POS mismatch.
- **WEAK** — same-POS substitution (often grammatical, just implausible).
- **UNKNOWN** — heuristic could not classify.

---

## Tools

`tools/dump_pool.py` writes the cached bunsetsu pool to a CSV file for inspection:

```bash
python tools/dump_pool.py bunsetsu_pool_v21_g0.pkl pool_g0.csv
```

---

## Citation & acknowledgements

If you use this code, please cite:

> Noda, Shotaro, & Nakatani, Kentaro. (2026). *A Japanese Implementation of the A-Maze Task: Method and Validation*. IEICE Technical Report TL2026-4(2026-06): 19-23.

This work adapts the A-Maze framework of:

> Boyce, V., Futrell, R., & Levy, R. P. (2020). *Maze Made Easy: Better and easier measurement of incremental processing difficulty*. Journal of Memory and Language, 111, 104082.

Original English/French implementation: <https://github.com/vboyce/Maze>.

For G-Maze paradigm:

> Forster, K. I., Guerrera, C., & Elliot, L. (2009). *The maze task: Measuring forced incremental sentence processing time*. Behavior Research Methods, 41(1), 163–171.

---

## License

MIT License - see [LICENSE](LICENSE).
