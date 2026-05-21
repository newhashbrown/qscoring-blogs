# qscoring-blogs

Analysis pipeline for QScoring's data-driven blog posts. Each post is one
self-contained Python script in `analysis/` that pulls a public dataset, runs
the analysis, writes charts to `assets/`, and emits a metrics JSON the website
template renders against.

The rendered post lives in the [qscoring.com](https://qscoring.com) repository
under `app/blog/`. This repo is the *generation* side of the hybrid setup.

## Layout

| Folder      | Contents                                                     |
| ----------- | ------------------------------------------------------------ |
| `analysis/` | One Python script per post                                   |
| `datasets/` | Raw Kaggle CSVs (gitignored; re-download via `kaggle` CLI)   |
| `assets/`   | Generated chart PNGs                                         |
| `output/`   | Generated standalone HTML and `metrics.json` per post        |

## Setup

```bash
# 1. Create a venv and install deps
py -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt

# 2. Configure Kaggle credentials (write your token to ~/.kaggle/access_token)
mkdir -p ~/.kaggle
# add your token to ~/.kaggle/access_token (single-line file)

# 3. Re-download the datasets used by current posts
./.venv/Scripts/kaggle.exe datasets download -d laotse/credit-risk-dataset -p ./datasets --unzip
```

## Running a post analysis

```bash
./.venv/Scripts/python.exe analysis/credit_scoring_breakdown.py
```

Outputs:
- Charts → `assets/*.png`
- Metrics → `output/metrics.json`
- Standalone HTML preview → `output/*.html`

## Posts

| Slug                                         | Dataset                              | Script                                |
| -------------------------------------------- | ------------------------------------ | ------------------------------------- |
| `how-credit-scoring-models-actually-work`    | `laotse/credit-risk-dataset`         | `credit_scoring_breakdown.py`         |
