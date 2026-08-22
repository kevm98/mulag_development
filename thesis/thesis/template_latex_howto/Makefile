#
# Robotics Research Lab - University of Kaiserslautern (2005)
#
# Autor: Bernd Helge Schäfer
# Autor: Martin Proetzsch
#

SRC = howto

pdf:
	pdflatex $(SRC)
	bibtex $(SRC)
	makeindex $(SRC)
	pdflatex $(SRC)
	pdflatex $(SRC)

all: pdf

clean:
	rm -fr *.aux
	rm -fr template/de/*.aux
	rm -fr template/en/*.aux
	rm -fr template/diss/*aux
	rm -fr *.bbl
	rm -fr *.toc
	rm -fr *.idx
	rm -fr *.lof
	rm -fr *.blg
	rm -fr *.log
	rm -fr *~
	rm -fr *.backup
	rm -fr *.pdf
	rm -fr *.ilg
	rm -fr *.ind
	rm -fr *.out
