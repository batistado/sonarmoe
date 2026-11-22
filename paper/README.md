# SonarMoE — *IEEE Transactions on Dependable and Secure Computing (TDSC)* Manuscript

Contains the two-column **IEEE Computer Society (`IEEEtran`, `10pt,journal,compsoc`)** manuscript, BibTeX bibliography, 5 vector PDF figures generated directly from Apple M5 Pro hardware JSON telemetry, static LaTeX validator, and one-click Overleaf archive (`sonarmoe_tdsc_overleaf.zip`).

## Quick Commands

### 1. Regenerate All 5 Vector PDF Figures from Hardware Telemetry
```bash
python3 ../scripts/generate_paper_figures.py
```

### 2. Run Static LaTeX, Citation, Cross-Reference & Abstract Gate Validator
```bash
python3 validate_latex.py
```

### 3. Compile PDF
* **Overleaf (1-click):** Upload `sonarmoe_tdsc_overleaf.zip` directly as a new project on Overleaf (`pdfLaTeX` compiler, main document `main.tex`).
* **Local TeX Live / MacTeX:**
  ```bash
  pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
  ```

## Author Roster (IEEE Author Portal)

| Order | Name | Email | Affiliation | Role / ORCID |
| :---: | :--- | :--- | :--- | :--- |
| 1 | Mohammed Kamran Syed | `mohammedkamran.syed@ieee.org` | IEEE Independent Researcher, Mountain House, CA, USA | Corresponding & Submitting Author; ORCID: `https://orcid.org/0009-0009-9058-5375` |
| 2 | Anudeep Rentala | `anudeep.rentala@ieee.org` | IEEE Independent Researcher, Fremont, CA, USA | Co-Author |
| 3 | Somesh Rahul | `someshrahul@ieee.org` | IEEE Independent Researcher, Fremont, CA, USA | Co-Author |
| 4 | Raghuram Malpe Pai | `raghuram.pai@ieee.org` | IEEE Independent Researcher, Sunnyvale, CA, USA | Co-Author |
| 5 | Krishna Suman Dara | `krishna-suman.dara@ieee.org` | IEEE Independent Researcher, Sunnyvale, CA, USA | Co-Author |
