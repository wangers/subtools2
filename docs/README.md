## Documentation

**Docments is under construnction now**

This documentation project employs [Sphinx](https://www.sphinx-doc.org/en/master/index.html) with [Napoleon](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) extension, and [autodoc](https://www.sphinx-doc.org/zh-cn/master/usage/extensions/autodoc.html#module-sphinx.ext.autodoc) is utilized to extract docstring from the code.

## Quickstart

To build the docs as HTML, first navigate to the docs folder:
```shell
cd docs
# install reqs if needed
pip install -r requirements.txt
```
on Linux, you can then use make to build the html:
```shell
make clean
make html

# or
bash .build_docs.sh
```
The built HTML docs will then be available in the `build/html` folder, and can be opened with any browser.

## Development

Docs is written in [MyST](https://myst-parser.readthedocs.io/en/latest/index.html) (enhanced markdown) format, and

### Auto-building the docs

When working on documentation, to avoid having to rebuild the docs each time a change is made, `sphinx-autobuild` can
be used. This can be installed using:
```shell
pip install sphinx-autobuild
```
and used by running the following command:
```shell
sphinx-autobuild . build/html
```

### Write & Check docs

Make changes to the documentation in `source`, do `make html` again, and check your changes by checking the html files in local dir `build/html`.
```shell
cd build/html
python3 -m http.server 8000
```
Then open browser and go to http://0.0.0.0:8000 to view the generated documentation. Alternatively, If you are [VSCode](https://code.visualstudio.com/) user, install [Live Server Extension](https://marketplace.visualstudio.com/items?itemName=ritwickdey.LiveServer) to open html directly.

If everything looks good, make a Pull Request and the document [Github Action](https://docs.github.com/zh/actions) will update automatically once the change has landed.
