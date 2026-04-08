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

## Getting Help

- **Bug reports / feature requests:** Open a [GitHub Issue](https://github.com/lvzhuojun/Social-Sentiment-Tracker-/issues)
- **Documentation discrepancies** between `README.md` and `README_CN.md`: open an issue with label `documentation`
- **BERT training issues:** label the issue `model:bert`
- **Baseline model issues:** label the issue `model:baseline`
- **Streamlit demo issues:** label the issue `streamlit`
