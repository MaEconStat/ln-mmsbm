# Publishing checklist / 创建 GitHub 新仓库

GitHub account identified through the connected account: `MaEconStat`. The available GitHub integration can read existing repositories and create files in an **existing** repository but currently offers **no tool for creating a new repository**. No existing repo has been modified.

Suggested new repo: `ln-mmsbm-python` (public). To publish after local acceptance:

1. Open https://github.com/new, set Repository name `ln-mmsbm-python`, Public, do **not** initialize README/license (both already included here).
2. From your extracted directory:

```bash
git init
git add .
git commit -m "Initial research prototype: TWIST baseline and lognormal extension"
git branch -M main
git remote add origin https://github.com/MaEconStat/ln-mmsbm-python.git
git push -u origin main
```

3. Add GitHub Actions CI if desired; run `pytest` and compare with the original R baseline before referencing the repository from your homepage as a validated implementation.
4. Add your own authorship and contact metadata in `pyproject.toml` / `README.md` only after verifying the desired public-facing identity. Avoid publishing private server paths, unreleased empirical datasets, student credentials or manuscript drafts.
