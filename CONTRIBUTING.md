# Contributing to Social Sentiment Tracker

Thank you for your interest in contributing! This document defines the standards that keep the project's code quality and **bilingual documentation** consistent and professional.

> This is a portfolio project. Contributions are welcome as GitHub Issues, Pull Requests, or documentation improvements.

---

## Table of Contents

- [Documentation Standards](#documentation-standards)
  - [Bilingual Sync Policy](#bilingual-sync-policy)
  - [When to Update Documentation](#when-to-update-documentation)
- [Docstring Standards](#docstring-standards)
  - [Format — Google Style](#format--google-style)
  - [Class Docstrings](#class-docstrings)
  - [Module-Level Docstrings](#module-level-docstrings)
- [Commit Message Format](#commit-message-format)
  - [Conventional Commits](#conventional-commits)
  - [Types and Scopes](#types-and-scopes)
  - [Examples](#commit-examples)
- [Branch Naming](#branch-naming)
- [Pull Request Checklist](#pull-request-checklist)
- [Code Style](#code-style)
- [Directory Structure](#directory-structure)
- [.gitignore Supplement Rules](#gitignore-supplement-rules)
- [Forbidden Items](#forbidden-items)
- [Getting Help](#getting-help)

---

## Documentation Standards

### Bilingual Sync Policy

> **Core rule:** `README.md` (English) and `README_CN.md` (Chinese) are **primary documentation**.
> They **must be updated in the same commit** for every change that affects user-visible behaviour.

This rule applies to any change involving:

- Public API additions or signature changes
- New or removed modules, files, or classes
- Installation steps or dependency changes
- Project structure changes
- Hyperparameter or configuration changes
- New Streamlit pages or notebook additions

**Workflow for every code change:**

```
1. Make your code change
2. Update README.md — add/modify the relevant section(s)
3. Mirror the change in README_CN.md — translate prose; keep code blocks,
   file names, and function names in English
4. Update CHANGELOG.md under the appropriate version heading
5. Verify both files render correctly on GitHub before committing
6. Stage all three files in the same commit
```

**Forbidden pattern:**

```bash
# ❌ WRONG — code change without documentation update
git commit -m "feat(bert_model): add batch inference"
# README.md and README_CN.md not updated → violates bilingual sync policy

# ✅ CORRECT — all three files in one commit
git add src/bert_model.py README.md README_CN.md CHANGELOG.md
git commit -m "feat(bert_model): add predict_bert() batch inference function"
```

---

### When to Update Documentation

| Change Type | `README.md` | `README_CN.md` | `CHANGELOG.md` | Docstring |
|-------------|:-----------:|:--------------:|:--------------:|:---------:|
| New public function | ✅ API Overview | ✅ Mirror | ✅ Added | Required |
| Changed function signature | ✅ API Overview | ✅ Mirror | ✅ Changed | Required |
| Removed function | ✅ API Overview | ✅ Mirror | ✅ Removed | N/A |
| New dependency added | ✅ Tech Stack + Install | ✅ Mirror | ✅ Added | N/A |
| Bug fix (no API change) | — | — | ✅ Fixed | If applicable |
| New Streamlit page | ✅ Usage section | ✅ Mirror | ✅ Added | N/A |
| New notebook | ✅ Notebooks table | ✅ Mirror | ✅ Added | N/A |
| Config constant added | ✅ API Overview | ✅ Mirror | ✅ Added | Required |
| Hyperparameter default changed | ✅ Installation table | ✅ Mirror | ✅ Changed | Required |
| Documentation-only fix | ✅ | ✅ | ✅ Changed | If applicable |
| Refactor (no behaviour change) | — | — | ✅ Changed | If applicable |

---

## Docstring Standards

### Format — Google Style

All public functions and methods must have **Google-style docstrings**. This is the canonical format used throughout this codebase.

**Mandatory sections:**

| Section | Required | Notes |
|---------|----------|-------|
| One-line summary | ✅ | Imperative mood: "Compute…", "Load…", "Return…" |
| `Args:` | ✅ | One entry per parameter; include type and default |
| `Returns:` | ✅ | Type and description of return value(s) |
| `Raises:` | If applicable | List exceptions by class name only when they can actually be raised |
| `Example:` | ✅ | At least one `>>>` doctest showing a realistic call and expected output |

**Template — copy this for new functions:**

```python
def function_name(
    param1: type,
    param2: type = default,
) -> return_type:
    """One-line imperative summary of what this function does.

    Optional longer description paragraph explaining the approach, any
    caveats, or important implementation details.

    Args:
        param1: Description of the first parameter.
        param2: Description of the second parameter. Defaults to ``default``.

    Returns:
        Description of what is returned — include type, shape for arrays,
        or key/value description for dicts.

    Raises:
        ValueError: When ``param1`` does not meet the required condition.
        FileNotFoundError: When a required file path does not exist on disk.

    Example:
        >>> result = function_name("hello world", param2=True)
        >>> result
        'expected_output'
    """
```

**Real example from this codebase** (`src/data_loader.py`):

```python
def clean_text(text: str) -> str:
    """Clean a raw social-media string.

    Steps applied in order:
    1. Lower-case
    2. Remove URLs (http/https/www)
    3. Remove @mentions
    4. Strip ``#`` from hashtags (keep the word)
    5. Remove remaining non-alphanumeric characters
    6. Collapse multiple whitespace to a single space

    Args:
        text: Raw input string.

    Returns:
        Cleaned string (may be empty if all tokens were noise).

    Example:
        >>> clean_text("Hello @user! Check https://example.com #NLP :)")
        'hello check nlp'
    """
```

**Rules:**
- Function summary line: max ~79 characters, no trailing period
- Use backtick pairs (` `` `) around inline code references within docstrings
- For optional parameters, always state the default: `Defaults to ``42```.
- If a function never raises, omit the `Raises:` section entirely
- `Example:` blocks must be executable as doctests where possible

---

### Class Docstrings

For classes, the class-level docstring describes purpose and documents `__init__` arguments. `forward()` / `__call__` / `__getitem__` methods get their own docstrings.

```python
class SentimentClassifier(nn.Module):
    """BERT-based binary / multi-class sentiment classifier.

    Architecture:
        BERT encoder → Dropout(p) → Linear(hidden_size, num_labels)

    Args:
        num_labels: Number of output classes (default 2 for binary).
        model_name: HuggingFace model identifier.
        dropout: Dropout probability applied before the classification head.

    Example:
        >>> model = SentimentClassifier(num_labels=2)
        >>> output = model(input_ids, attention_mask)
    """
```

---

### Module-Level Docstrings

Every `src/` file must begin with a module-level docstring that includes:

1. **One-line summary** — what the module provides
2. **Brief description** — the role of this module in the pipeline
3. **Key public symbols** — a short list of the main exports (optional but recommended)

```python
"""
src/evaluate.py — Model evaluation and visualisation utilities.

Provides functions for computing metrics, plotting confusion matrices,
ROC curves, and comparing multiple models side-by-side.
"""
```

---

## Commit Message Format

### Conventional Commits

All commits must follow [Conventional Commits v1.0.0](https://www.conventionalcommits.org/).

```
<type>(<scope>): <short description>

[optional body — explain WHY, not WHAT]

[optional footer — BREAKING CHANGE: description]
```

**Rules for the subject line:**
- Max **72 characters**
- Lowercase after `type(scope):`
- No trailing period
- Use imperative mood: "add", "fix", "update" (not "added", "fixed", "updated")

**Body rules:**
- Explain the motivation or context, not the implementation
- Wrap lines at 100 characters
- Separate from subject with a blank line

---

### Types and Scopes

**Types:**

| Type | When to Use |
|------|-------------|
| `feat` | New feature or public function |
| `fix` | Bug fix |
| `docs` | Documentation-only change (README, CHANGELOG, docstrings) |
| `refactor` | Code restructuring without behaviour change |
| `test` | Adding or updating tests |
| `chore` | Build scripts, dependency updates, environment config |
| `perf` | Performance improvement (e.g. vectorisation, caching) |
| `style` | Formatting only (whitespace, blank lines — no logic change) |

**Scopes** — use the module or file name:

| Scope | File |
|-------|------|
| `config` | `config.py` |
| `data_loader` | `src/data_loader.py` |
| `preprocess` | `src/preprocess.py` |
| `baseline_model` | `src/baseline_model.py` |
| `bert_model` | `src/bert_model.py` |
| `evaluate` | `src/evaluate.py` |
| `visualize` | `src/visualize.py` |
| `streamlit` | `app/streamlit_app.py` |
| `notebooks` | `notebooks/` |
| `docs` | `README.md`, `README_CN.md`, `CHANGELOG.md`, `CONTRIBUTING.md` |
| `deps` | `requirements.txt`, `environment.yml` |
| `ci` | GitHub Actions workflows |
| `rag` | `agentic_rag/` — any file in the RAG subproject |
| `rag_embed` | `agentic_rag/embedder.py` |
| `rag_retriever` | `agentic_rag/retriever.py` |
| `rag_agent` | `agentic_rag/agent.py` |
| `rag_api` | `agentic_rag/api.py` |
| `rag_config` | `agentic_rag/config.py` |

---

### Commit Examples

```bash
# New feature
feat(bert_model): add predict_bert() batch inference function

# Bug fix
fix(data_loader): handle None input in clean_text() without crashing

# Documentation update (bilingual)
docs(readme): sync README_CN.md with architecture diagram expansion

# Dependency update
chore(deps): pin transformers>=4.35.0 in requirements.txt

# Refactor without behaviour change
refactor(evaluate): extract ROC curve logic into standalone plot_roc_curve()

# Performance improvement
perf(visualize): cache TF-IDF vectorizer in plot_top_keywords() to avoid refit

# Breaking change (use footer)
feat(baseline_model): change predict() return type from list to numpy array

BREAKING CHANGE: predict() now returns (np.ndarray, np.ndarray) instead of
(list, list). Update any downstream code that calls list-specific methods.
```

---

## Branch Naming

Format: `<type>/<short-kebab-description>`

| Pattern | Example | Purpose |
|---------|---------|---------|
| `feat/<description>` | `feat/add-aspect-based-sentiment` | New feature |
| `fix/<description>` | `fix/wordcloud-empty-class-crash` | Bug fix |
| `docs/<description>` | `docs/update-readme-cn-api-overview` | Documentation only |
| `refactor/<description>` | `refactor/evaluate-module-cleanup` | Refactoring |
| `chore/<description>` | `chore/update-torch-version-pin` | Maintenance |
| `perf/<description>` | `perf/bert-inference-int8-quantise` | Performance |

**Rules:**
- Always branch from `main`
- Lowercase only; use hyphens (not underscores or spaces)
- Max ~40 characters in the description part
- Delete the branch after the PR is merged

---

## Pull Request Checklist

Copy this template into every PR description:

```markdown
## Description
<!-- What does this PR do and why is this change needed? -->

## Type of Change
- [ ] New feature (`feat`)
- [ ] Bug fix (`fix`)
- [ ] Documentation update (`docs`)
- [ ] Refactoring (`refactor`)
- [ ] Performance improvement (`perf`)
- [ ] Dependency / config update (`chore`)

## Checklist

### Code Quality
- [ ] Code follows PEP 8 style conventions
- [ ] No unused imports or variables introduced
- [ ] No hardcoded paths — all paths go through `config.py` using `pathlib.Path`
- [ ] Random seeds set via `config.set_seed()` where applicable

### Documentation — REQUIRED for every user-visible change
- [ ] New / changed public functions have complete Google-style docstrings
      (`Args`, `Returns`, `Raises`, `Example`)
- [ ] Module-level docstring updated if new exports were added
- [ ] `README.md` updated for any user-visible change (API, install, structure)
- [ ] `README_CN.md` updated to mirror `README.md` **in the same commit**
- [ ] `CHANGELOG.md` updated under the correct version heading
      (`Added` / `Changed` / `Fixed` / `Removed`)

### Safety
- [ ] No secrets, API keys, or credentials included
- [ ] Model artefacts (`*.pkl`, `*.pt`) are **not** committed (git-ignored)
- [ ] Raw dataset files are **not** committed (`data/raw/` is git-ignored)
- [ ] No large binary files added without discussion

### Git Hygiene
- [ ] Commit messages follow Conventional Commits format
- [ ] Branch name follows `<type>/<description>` convention
- [ ] All commits are logically atomic (one concern per commit)
```

---

## Code Style

| Rule | Detail |
|------|--------|
| Style guide | PEP 8 |
| Max line length | 100 characters |
| Imports order | stdlib → third-party → local (`config`, `src.*`) |
| Type hints | Required on all public function signatures |
| Logging | Use `config.get_logger(__name__)` — never use bare `print()` in `src/` |
| Paths | Always use `pathlib.Path` — never raw string paths |
| Constants | Define in `config.py` — never hardcode numbers or paths in module files |
| Exception handling | Only catch specific exceptions; never bare `except:` |

---

## Directory Structure

```
DS/
├── social-sentiment-tracker/       # existing project (BERT sentiment, FastAPI, Streamlit)
│   ├── src/                        # core library modules — one concern per file
│   │   ├── bert_model.py           #   BERT fine-tuning + inference
│   │   ├── baseline_model.py       #   TF-IDF + LR baseline
│   │   ├── data_loader.py          #   dataset loading + text cleaning
│   │   ├── preprocess.py           #   tokenisation + split helpers
│   │   ├── evaluate.py             #   metrics + confusion matrix
│   │   ├── explain.py              #   SHAP explanations
│   │   └── visualize.py            #   all matplotlib/plotly figures
│   ├── api/
│   │   ├── serve.py                #   FastAPI app (health, /predict, /predict/batch)
│   │   └── requirements.txt        #   API-only slim deps
│   ├── app/
│   │   └── streamlit_app.py        #   Streamlit frontend
│   ├── scripts/                    # one-off training / tuning entry-points
│   ├── notebooks/                  # numbered EDA + modelling notebooks
│   ├── tests/                      # pytest — mirrors src/ structure
│   ├── data/raw/                   # (git-ignored) original datasets
│   ├── data/processed/             # (git-ignored) train/val/test splits
│   ├── models/                     # (git-ignored) serialised weights
│   ├── reports/figures/            # committed PNG charts for README
│   ├── config.py                   # single source of truth for paths + hyperparams
│   ├── environment.yml             # reproducible conda env spec
│   └── requirements.txt            # pinned pip deps
│
└── agentic_rag/                    # NEW — Agentic RAG subproject
    ├── src/
    │   ├── embedder.py             #   BERT embedding wrapper (get_embedding / encode_batch)
    │   ├── retriever.py            #   FAISS index build + similarity search
    │   ├── agent.py                #   LangChain agent + tool definitions
    │   └── utils.py                #   shared helpers (chunking, text normalisation)
    ├── api/
    │   └── serve.py                #   FastAPI endpoints (/query, /index, /health)
    ├── tests/                      # pytest for every src/ module
    ├── vector_store/               # (git-ignored) persisted FAISS index files
    ├── cache/                      # (git-ignored) LLM response cache
    ├── config.py                   # paths + env-var loading (API keys via os.getenv)
    ├── requirements.txt            # agentic_rag-specific deps
    └── README.md                   # subproject README (English)
```

**Rules:**

- Every `src/` module has **one responsibility** — do not put retrieval logic inside the agent file.
- New files go into the appropriate `src/` directory — do not create top-level files unless they are entry-points (`run_*.py`, `main.py`).
- Test files mirror `src/` names exactly: `src/embedder.py` → `tests/test_embedder.py`.
- `config.py` is the **only** place that reads environment variables and defines file paths — no `os.getenv()` calls scattered in other modules.
- Cross-subproject imports are **not allowed** — `agentic_rag` must not import from `social-sentiment-tracker/src` directly; copy or package shared utilities.

---

## .gitignore Supplement Rules

The following patterns are **not yet in `.gitignore`** but must be respected for `agentic_rag/`. Add them to the root `.gitignore` once the subproject directory exists:

```gitignore
# ── agentic_rag — vector store index files (can be gigabytes) ────────────────
agentic_rag/vector_store/
agentic_rag/*.faiss
agentic_rag/*.index

# ── agentic_rag — LLM response cache ─────────────────────────────────────────
agentic_rag/cache/

# ── agentic_rag — any .env file with API keys ────────────────────────────────
agentic_rag/.env
agentic_rag/.env.*

# ── agentic_rag — model/tokenizer downloads (HuggingFace cache) ──────────────
agentic_rag/model_cache/

# ── agentic_rag — serialised embeddings (numpy arrays, pickle) ───────────────
agentic_rag/embeddings/
agentic_rag/*.npy
agentic_rag/*.npz
```

**General rules that already apply and must be observed:**

| Pattern | Reason |
|---------|--------|
| `*.pt`, `*.pkl`, `*.bin` | Model weights — too large; regenerate from training scripts |
| `data/raw/` | Datasets may have licence restrictions; regenerate with `download_data.py` |
| `.env`, `secrets.yaml` | Contains API keys — **never** commit under any circumstances |
| `logs/` | Runtime artefacts — no diagnostic value in version history |
| `*_executed.ipynb` | Executed notebooks contain absolute local paths in cell outputs |
| `reports/metrics.json` | Generated at evaluation time; not a source file |

---

## Forbidden Items

The following actions are **strictly prohibited** in all commits and PRs:

### Secrets and credentials
- **Never** commit any file containing API keys, tokens, passwords, or connection strings.
  This includes `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, database URIs, and OAuth tokens.
- All secrets must be loaded via `os.getenv("KEY_NAME")` and provided through a `.env`
  file that is git-ignored, or through the deployment environment.
- If a secret is accidentally committed, treat it as compromised immediately — rotate it
  before pushing any fix.

### Large binary files
- Do not commit model weights (`*.pt`, `*.pkl`, `*.bin`, `*.onnx`).
- Do not commit FAISS indexes (`*.faiss`, `*.index`) or embedding arrays (`*.npy`, `*.npz`).
- Do not commit raw datasets (`data/raw/`).

### Code quality violations
- **No bare `except:`** — always catch a specific exception class.
- **No `print()` in `src/` or `agentic_rag/src/`** — use `config.get_logger(__name__)`.
- **No hardcoded paths** — all file paths go through `config.py` using `pathlib.Path`.
- **No hardcoded API keys or model names** as string literals — use constants in `config.py`
  and read secrets from environment variables.
- **No `import *`** — explicit imports only.

### Workflow violations
- Do not push directly to `main` — all changes go through a feature branch and PR.
- Do not skip the PR checklist (copy it into every PR description).
- Do not commit without updating `CHANGELOG.md` for any user-visible change.
- Do not use `--no-verify` to bypass pre-commit hooks.
- Do not batch unrelated changes into a single commit — one logical concern per commit.
- Do not `force-push` to `main` or any shared branch.

### Documentation violations
- Do not add a new public function without a Google-style docstring
  (`Args`, `Returns`, `Raises`, `Example`).
- Do not change `README.md` without mirroring the change in `README_CN.md` in the same commit.

---

## Getting Help

- **Bug reports / feature requests:** Open a [GitHub Issue](https://github.com/lvzhuojun/social-sentiment-tracker/issues)
- **Documentation discrepancies** between `README.md` and `README_CN.md`: open an issue with label `documentation`
- **BERT training issues:** label the issue `model:bert`
- **Baseline model issues:** label the issue `model:baseline`
- **Streamlit demo issues:** label the issue `streamlit`
