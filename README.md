# Japanese A-Maze
A surprisal-based distractor generator for the G-Maze psycholinguistic task in Japanese.
Inspired by the English/French A-Maze of Boyce, Futrell & Levy (2020); rebuilt around Japanese-specific structural constraints (particle clashes, POS-mismatch by slot type, bunsetsu-level segmentation) and an autoregressive language model (`rinna/japanese-gpt2-medium`) for surprisal scoring.

## Quick start
```bash
# 1. Install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Generate distractors
python maze_japanese.py input.txt output.txt --seed 42

# 3. Validate the output
python maze_japanese.py --validate output.txt
```
First run will take a few minutes as it builds and caches the ~157k-bunsetsu pool (`.bunsetsu_pool_*.pkl`) and downloads the LM (~1.2 GB) on first use.

## Input format

One sentence per line, semicolon-delimited:

```
sample;1;猫が 走った。
sample;2;学生は 図書館で 本を 読んだ。
```

Fields: `tag ; item_id ; sentence`. Bunsetsu in the sentence are separated by spaces; you can pre-segment or let MeCab (UniDic) do it.

---

## Output format

```
sample;1;猫が 走った。;x-x-x 授業が。;0 1
sample;2;学生は 図書館で 本を 読んだ。;x-x-x 正常が 実際で メガネで。;0 1 2 3
```

Fields: `tag ; item_id ; sentence ; distractors ; label_indices`.

- Distractor at each position is the maze alternative to the corresponding bunsetsu of the real sentence.
- `x-x-x` means no distractor was emitted at that position (position 0 is always `x-x-x` by convention).
- `*word` (asterisk-prefixed) means the distractor is a **fallback** — it bypassed the strict rules and should be reviewed manually before use.

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

If the strict rule yields no candidates (or if no candidate passes the surprisal gate), the **fallback** strategy emits the most plausible-implausible candidate from a broader pool, marked with `*`. Position 0 (sentence-initial) is left as `x-x-x` by convention.

---

## CLI

```bash
python maze_japanese.py [-h] [--version] [--validate] [-v] [-p PARAMS]
                       [--format {delim,ibex}] [--log LEVEL] [--seed N]
                       [input] [output]
```

| Flag | Description |
|---|---|
| `input` | Input file (default `input.txt`). Ignored with `--validate`. |
| `output` | Output file path (generate mode); file to check (validate mode). |
| `--validate` | Run the validator instead of generating. |
| `-v` / `--verbose` | More detail in validator output. |
| `-p / --params` | Path to a key:value parameters file (see below). |
| `--format` | `delim` (default) or `ibex` (Ibex experiment format). |
| `--seed` | Random seed for reproducibility. |
| `--log` | `DEBUG`, `INFO`, `WARNING`. |
| `--version` | Print version. |

---

## Parameters

A parameters file (`-p params.txt`) is a list of `key: value` pairs. Defaults shown:

```
num_to_test:        80      # max candidates per slot considered for LM scoring
min_surprisal:      14.0    # bits — min LM surprisal for distractor acceptance
surprisal_ceiling:  35.0    # bits — above this, candidate is LM-OOV (skip)
freq_match_band:    2.0     # log-freq tolerance for matching target word
max_repeat:         0       # max reuses of the same exact form (0 = stem-cap only)
model_name:         "rinna/japanese-gpt2-medium"
grade_level:        6       # 1–6 = elementary kanji filter; 0 = off
use_lm:             True    # turn off to use frequency-matching only (no LM)
```

To reduce `x-x-x` count: lower `min_surprisal`, raise `freq_match_band`, or increase `num_to_test`.

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

Violation strength classes:
- **STRONG** — categorical double-が/を particle clash.
- **MEDIUM** — weak particle duplication (に/で/は), or POS mismatch.
- **WEAK** — same-POS substitution (often grammatical, just implausible).
- **UNKNOWN** — heuristic could not classify.

---

## Citation & acknowledgements

This work adapts the A-Maze framework of:

> Boyce, V., Futrell, R., & Levy, R. P. (2020). *Maze Made Easy: Better and easier measurement of incremental processing difficulty*. Journal of Memory and Language, 111, 104082.

Original English/French implementation: <https://github.com/vboyce/Maze>.

For G-Maze paradigm:

> Forster, K. I., Guerrera, C., & Elliot, L. (2009). *The maze task: Measuring forced incremental sentence processing time*. Behavior Research Methods, 41(1), 163–171.

---

## License

(Add a license here — e.g., MIT, CC-BY-NC. Currently unspecified.)
