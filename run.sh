#!/bin/bash

echo "Run handin 2 Liz van der Kamp (s2135752)"


# Script that returns a txt file
echo "Run the first script ..."
python3 NURHW2LizQ1.py 

# Script that pipes output to multiple files
echo "Run the second script ..."
python3 NURHW2LizQ2.py


echo "Generating the pdf"

pdflatex SolutionsHW2.tex
bibtex SolutionsHW2.aux
pdflatex SolutionsHW2.tex
pdflatex SolutionsHW2.tex
