## Project

DAS Seismology Workshop — SSA 2026, April 14
Website: https://ai4eps.github.io/DAS_Seismology_Workshop/
Repo: https://github.com/AI4EPS/DAS_Seismology_Workshop

## Build

```bash
pip install mkdocs-material mkdocs-jupyter pymdown-extensions
mkdocs serve        # local preview at http://127.0.0.1:8000
mkdocs gh-deploy    # deploy to GitHub Pages
```

## Slides

Slides are stored as GitHub Releases (tag: `Slides`) and downloaded during CI build.
Local slide sources on OneDrive:
- /Users/weiqiang/Library/CloudStorage/OneDrive-Personal/Documents/DAS Seismology Workshop/Deep Learning DAS.pptx

### How to upload slides to GitHub Releases

1. Convert .pptx to .pdf
2. Go to https://github.com/AI4EPS/DAS_Seismology_Workshop/releases
3. Create/edit the "Slides" release, attach PDF files
4. Update `.github/workflows/docs.yml` to uncomment the wget lines with correct filenames
5. Uncomment the iframe in the corresponding `docs/*.md` file

## Structure

- `docs/` — MkDocs content (README, slide pages, KaTeX)
- `docs/notebooks/` — symlink to `../notebooks/`
- `notebooks/` — Jupyter lab notebooks
- `SSA.md` — workshop description
- `schedule.md` — detailed schedule
