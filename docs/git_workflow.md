# Git Workflow for MentalHealthSympAI

This document explains how to work with Git and GitHub for this project.

## 1. Go to the project directory

In Colab:

```python
%cd /content/drive/MyDrive/master/courses/AI_in_medicine/AI_MD_Project
```

## 2. Check Git status

```bash
git status
git branch
git remote -v
```

Expected remote:

```bash
origin  https://github.com/YanPev/MentalHealthSympAI.git
```

## 3. Work on the correct branch

For Week 1 data exploration:

```bash
git checkout data/exploration
git pull origin data/exploration
```

For Week 2 item-level dataset work:

```bash
git checkout data/item-dataset
git pull origin data/item-dataset
```

## 4. Do not commit raw data or generated outputs

Do not commit:

```text
data/
outputs/
*.csv
*.tar.gz
*.wav
*.mat
```

Before every commit:

```bash
git status
```

Only code and documentation should be staged.

## 5. Add specific files only

Do not use:

```bash
git add .
```

Instead, add only files relevant to the task.

Example for Week 1:

```bash
git add .gitignore README.md requirements.txt \
  src/data/download_edaic_data.py \
  src/data/inspect_edaic_week1.py \
  docs/data_availability_report.md \
  docs/git_workflow.md \
  notebooks/01_run_data_download.ipynb
```

## 6. Commit

```bash
git commit -m "Add E-DAIC download and week 1 inspection scripts"
```

## 7. Push to GitHub

Try regular push first:

```bash
git push origin data/exploration
```

If authentication fails in Colab, use a GitHub Personal Access Token interactively:

```python
from getpass import getpass

username = "YOUR_GITHUB_USERNAME"
repo_owner = "YanPev"
repo = "MentalHealthSympAI"

token = getpass("GitHub token: ")

!git remote set-url origin https://{username}:{token}@github.com/{repo_owner}/{repo}.git
!git push origin data/exploration
```

After a successful push, reset the remote URL:

```bash
git remote set-url origin https://github.com/YanPev/MentalHealthSympAI.git
git remote -v
```

The remote URL should not contain the token.

## 8. Open a Pull Request

After pushing, open a Pull Request on GitHub:

```text
base: main
compare: data/exploration
```

Suggested PR title:

```text
Add E-DAIC download and Week 1 data inspection
```

## 9. Important rules

- Do not commit tokens.
- Do not commit raw data.
- Do not commit generated outputs.
- Do not use `git add .`.
- Always check `git status` before committing.
