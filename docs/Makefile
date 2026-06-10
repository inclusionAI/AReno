.PHONY: html clean

SPHINXBUILD ?= sphinx-build
SOURCEDIR = .
BUILDDIR = ../dist/docs

html:
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)

clean:
	rm -rf $(BUILDDIR)
