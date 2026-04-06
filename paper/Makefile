# ============================================================
# Makefile — SynthPID paper
# Usage:
#   make        — full build (latex + bibtex + latex x2)
#   make view   — open PDF
#   make clean  — remove build artifacts
#   make wc     — word count of main text
# ============================================================

MAIN    = main
LATEX   = pdflatex
BIBTEX  = bibtex
FLAGS   = -interaction=nonstopmode -halt-on-error

.PHONY: all view clean wc

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib sections/*.tex
	$(LATEX) $(FLAGS) $(MAIN)
	$(BIBTEX) $(MAIN)
	$(LATEX) $(FLAGS) $(MAIN)
	$(LATEX) $(FLAGS) $(MAIN)

view: $(MAIN).pdf
	xdg-open $(MAIN).pdf 2>/dev/null || open $(MAIN).pdf

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc \
	      *.fls *.fdb_latexmk *.synctex.gz \
	      sections/*.aux

wc:
	@detex sections/*.tex | wc -w
	@echo "words (approx)"
