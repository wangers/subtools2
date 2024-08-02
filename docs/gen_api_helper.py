"""
Generates the myst docs for a package in max 2 level.
"""

import os
import shutil
from egrecho.utils.io import resolve_patterns
from pathlib import Path
import egrecho
from collections import defaultdict

eval_rst_ = r"{eval-rst}"
toctree_ = r"{toctree}"


def get_module_md(*moudle):
    md = ""

    for m in moudle:
        md += f"# {m}"
        md += f"""
```{eval_rst_}
.. automodule:: {m}
    :members:

```

"""
    return md


base_package = "egrecho.pipeline"
DIR = "../"
DIR = Path(DIR).resolve()
rels = []
patterns = base_package.replace(".", os.sep) + "/**.py"
files = resolve_patterns(patterns=patterns, base_path=DIR)

for f in files:
    if os.path.basename(f) == "__init__.py":
        continue
    r = os.path.relpath(f, start=DIR)

    rels.append(r)

pkgs = defaultdict(list)
for rel in rels:
    pkg = os.path.dirname(rel)
    pkg = pkg.replace(os.sep, ".")
    pkgs[pkg].append(str(rel).replace(".py", "").replace(os.sep, "."))
# pkgs = {pkg: sorted(pkgs[pkg]) for pkg in sorted(pkgs)}
lvl_pkgs = defaultdict(list)
for pkg in pkgs:
    lvl_pkgs[len(pkg.split(".")) - len(base_package.split("."))].append(pkg)

temp_gen_api = "./_gen_api"
if os.path.exists(temp_gen_api):
    shutil.rmtree(temp_gen_api)
os.makedirs(temp_gen_api)

for lvl in range(len(lvl_pkgs)):
    if lvl == len(lvl_pkgs) - 1:
        for pkg in lvl_pkgs[lvl]:
            this_md = get_module_md(*pkgs[pkg])

            with open(f"./_gen_api/{pkg}.md", "w") as f:
                f.write(this_md)
    else:
        for pkg in lvl_pkgs[lvl]:
            this_md = get_module_md(*pkgs[pkg])
            for nxt_pkg in lvl_pkgs[lvl + 1]:
                if nxt_pkg.startswith(pkg):
                    this_md += f"""
# {nxt_pkg}
```{toctree_}
:maxdepth: 1

{nxt_pkg}.md
```

"""
            with open(f"./_gen_api/{pkg}.md", "w") as f:
                f.write(this_md)

base_md = f"./_gen_api/{lvl_pkgs[0][0]}.md"
index_md = "./_gen_api/index.md"
os.rename(base_md, index_md)
# os.makedirs("./source/api/_gen", exist_ok=True)
# with open(f"./source/api/_gen/utils.md", "w") as f:
#     f.write(md)
