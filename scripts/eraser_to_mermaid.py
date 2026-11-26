"""Convert docs/architecture/*.eraser flow charts to Mermaid.

Writes one .mmd file per diagram, refreshes the Mermaid blocks in
docs/architecture/README.md (GitHub renders them) and the data embedded in
docs/architecture/viewer.html.

Usage: python3 scripts/eraser_to_mermaid.py
"""

import json
import re
import sys
from pathlib import Path

ARCH = Path(__file__).resolve().parent.parent / "docs" / "architecture"
CONN = re.compile(r"\s+(<>|<-->|-->|->|>|<|--|-)\s+")
META = re.compile(r"^(title|direction|colorMode|styleMode|typeface)\s+(.+)$")
ITEM = re.compile(r"^([^\[\]{}]+?)\s*(?:\[([^\]]*)\])?\s*(\{)?$")
DIRECTIONS = {"right": "LR", "left": "RL", "down": "TB", "up": "BT"}
COLORS = {
    "gray": "#6b7280", "green": "#2b8352", "blue": "#2c66c4",
    "purple": "#7148c0", "orange": "#bf6716", "red": "#c23434",
}
BEGIN, END = "<!-- mermaid:begin -->", "<!-- mermaid:end -->"


def parse(src: str) -> dict:
    meta, items, edges, stack = {}, {}, [], []

    def ensure(name):
        items.setdefault(name, {"name": name, "attrs": {}, "parent": None, "group": False})

    for n, raw in enumerate(src.splitlines(), 1):
        line = re.sub(r"//.*$", "", raw).strip()
        if not line:
            continue
        if m := META.match(line):
            meta[m[1]] = m[2]
        elif line == "}":
            if not stack:
                raise ValueError(f"line {n}: '}}' closes nothing")
            stack.pop()
        elif CONN.search(line) and "[" not in line and not line.endswith("{"):
            parts = CONN.split(line)
            label = ""
            if ":" in parts[-1]:
                parts[-1], label = (s.strip() for s in parts[-1].split(":", 1))
            for i in range(0, len(parts) - 2, 2):
                a, op, b = parts[i].strip(), parts[i + 1], parts[i + 2].strip()
                ensure(a)
                ensure(b)
                if op == "<":
                    a, b = b, a
                edges.append({"a": a, "b": b, "op": op, "label": label if i + 3 >= len(parts) else ""})
        elif m := ITEM.match(line):
            name = m[1].strip()
            attrs = dict(
                (k.strip(), v.strip())
                for k, v in (p.split(":", 1) for p in (m[2] or "").split(",") if ":" in p)
            )
            ensure(name)
            items[name].update(attrs=attrs, parent=stack[-1] if stack else None, group=bool(m[3]))
            if m[3]:
                stack.append(name)
        else:
            raise ValueError(f"line {n}: can't read {line!r}")
    if stack:
        raise ValueError(f"group {stack[-1]!r} is missing its closing }}")
    return {"meta": meta, "items": items, "edges": edges}


def color_of(items: dict, name: str) -> str:
    while name:
        c = items[name]["attrs"].get("color")
        if c in COLORS:
            return c
        name = items[name]["parent"]
    return "gray"


def q(text: str) -> str:
    return '"' + text.replace('"', "#quot;") + '"'


def to_mermaid(src: str) -> str:
    d = parse(src)
    items, ids = d["items"], {}
    for i, name in enumerate(items):
        ids[name] = ("g" if items[name]["group"] else "n") + str(i)

    out = []
    if title := d["meta"].get("title"):
        out += ["---", f"title: {title}", "---"]
    out.append(f"flowchart {DIRECTIONS.get(d['meta'].get('direction', 'right'), 'LR')}")

    def shape(name):
        s = items[name]["attrs"].get("shape")
        return f"[({q(name)})]" if s == "cylinder" else f"{{{q(name)}}}" if s == "diamond" else f"[{q(name)}]"

    def emit(parent, indent):
        for name, it in items.items():
            if it["parent"] != parent:
                continue
            if it["group"]:
                out.append(f"{indent}subgraph {ids[name]}[{q(name)}]")
                emit(name, indent + "  ")
                out.append(f"{indent}end")
            else:
                out.append(f"{indent}{ids[name]}{shape(name)}")

    emit(None, "  ")
    for e in d["edges"]:
        arrow = {"<>": "<-->", "<-->": "<-->", "--": "-.-", "-->": "-.->", "-": "---"}.get(e["op"], "-->")
        label = f"|{q(e['label'])}|" if e["label"] else ""
        out.append(f"  {ids[e['a']]} {arrow}{label} {ids[e['b']]}")

    for c, hexa in COLORS.items():
        out.append(f"  classDef {c} fill:{hexa}22,stroke:{hexa},stroke-width:1.5px")
    by_color = {}
    for name, it in items.items():
        if not it["group"]:
            by_color.setdefault(color_of(items, name), []).append(ids[name])
    for c, members in by_color.items():
        out.append(f"  class {','.join(members)} {c}")
    for name, it in items.items():
        if it["group"]:
            hexa = COLORS[color_of(items, name)]
            out.append(f"  style {ids[name]} fill:{hexa}0f,stroke:{hexa},stroke-dasharray:5 4")
    return "\n".join(out) + "\n"


def main() -> int:
    blocks = []
    for path in sorted(ARCH.glob("*.eraser")):
        mmd = to_mermaid(path.read_text())
        path.with_suffix(".mmd").write_text(mmd)
        title = parse(path.read_text())["meta"].get("title", path.stem)
        blocks.append(f"### {title}\n\n```mermaid\n{mmd}```\n")
        print(f"wrote {path.with_suffix('.mmd').name}")

    readme = ARCH / "README.md"
    text = readme.read_text()
    section = f"{BEGIN}\n" + "\n".join(blocks) + f"{END}"
    if BEGIN in text:
        text = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END), lambda _: section, text, flags=re.S)
    else:
        text = text.rstrip() + "\n\n## Diagrams\n\n" + section + "\n"
    readme.write_text(text)
    print(f"updated {readme.name}")

    viewer = ARCH / "viewer.html"
    if viewer.exists():
        sources = {p.name: p.read_text() for p in sorted([*ARCH.glob("*.eraser"), *ARCH.glob("*.mmd")])}
        data = json.dumps(sources).replace("</", "<\\/")
        html = re.sub(r"/\*SOURCES\*/.*?/\*END\*/", lambda _: f"/*SOURCES*/{data}/*END*/", viewer.read_text(), flags=re.S)
        viewer.write_text(html)
        print(f"updated {viewer.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
