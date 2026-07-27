The Documents in this repository should give you an overview, of how to use our Latex templates. It includes different document classes for your Thesis, Seminar, Project or Dissertation. The content includes the common structure, Latex in general, tables and images, and more.

The first thing you need to know is how to compile the Latex Document to read the pdf file.
In **Linux**, you can simply run:
```sh
make
```
Then, you can then open the created howto.pdf with a pdf viewer of your preference, e.g.:
```sh
okular howto.pdf
```
If you want to remove the generated files run
```sh
make clean
```
This might be needed to resolve some compilation errors.

In **Windows**, you need to open howto.tex with a Latex-IDEs of your choice, e.g. TeXnicCenter and compile it with the following commands
```sh
pdflatex howto.tex
bibtex howto.aux
pdflatex howto.tex
```
This should be the standard for most Latex-IDEs


If you have any problems please ask your supervisor, or ask a senior PhD-Student