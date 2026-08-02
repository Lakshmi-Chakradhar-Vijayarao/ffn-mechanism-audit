#!/usr/bin/env python3
"""Regenerate draft/paper_draft.md from draft/latex/main.tex.

Keeps the markdown mirror in sync with the LaTeX source, including the
appendix structure (\\appendix -> "Appendix A:", "Appendix B:", ...).

Usage:
    python3 code/90_sync_paper_draft_md.py                 # de-anonymized header
    python3 code/90_sync_paper_draft_md.py --anonymous     # anonymized header
    python3 code/90_sync_paper_draft_md.py --out FILE
"""
import argparse
import os
import re
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TEX = os.path.join(ROOT, "draft", "latex", "main.tex")
OUT = os.path.join(ROOT, "draft", "paper_draft.md")

DEANON_HEADER = (
    "**Author names withheld for review**\n"
    "Independent Researcher\n"
)
ANON_HEADER = "**Anonymous Author(s)**\nPaper under double-blind review\n"

MATH_SYMBOLS = [
    (r"\\binom\{([^{}]*)\}\{([^{}]*)\}", r"C(\1, \2)"),
    (r"\\times", "x"), (r"\\geq", ">="), (r"\\leq", "<="),
    (r"\\approx", "~"), (r"\\pm", "+/-"), (r"\\to\b", "->"),
    (r"\\rightarrow", "->"), (r"\\alpha", "alpha"), (r"\\kappa", "kappa"),
    (r"\\Delta", "Delta"), (r"\\delta", "delta"), (r"\\chi", "chi"),
    (r"\\in\b", " in "), (r"\\cdot", "."), (r"\\ldots", "..."),
    (r"\\text\{([^{}]*)\}", r"\1"), (r"\\mathrm\{([^{}]*)\}", r"\1"),
    (r"\\emph\{([^{}]*)\}", r"*\1*"),
    (r"\\sqrt\{([^{}]*)\}", r"sqrt\1"),
    (r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"\1/\2"),
    (r"\{,\}", ","), (r"\{:\}", ":"), (r"\{=\}", "="),
]


def strip_comments(s):
    return re.sub(r"(?<!\\)%.*", "", s)


def innermost(pattern, repl, s):
    """Apply a brace-taking command repeatedly, innermost first."""
    prev = None
    while prev != s:
        prev = s
        s = re.sub(pattern, repl, s)
    return s


def demath(s):
    def fix(m):
        body = m.group(1)
        for pat, rep in MATH_SYMBOLS:
            body = re.sub(pat, rep, body)
        body = body.replace("^{", "^").replace("}", "")
        body = body.replace("{", "").replace("\\", "")
        return body
    s = re.sub(r"\$([^$]*)\$", fix, s)
    return s


def convert_inline(s, refmap):
    s = strip_comments(s)
    # references first (before brace stripping)
    def ref(m):
        return refmap.get(m.group(1), "?")
    s = re.sub(r"§\\ref\{([^{}]*)\}", lambda m: "§" + ref(m), s)
    s = re.sub(r"Appendix~\\ref\{([^{}]*)\}", lambda m: "Appendix " + ref(m), s)
    s = re.sub(r"Appendices~\\ref\{([^{}]*)\}", lambda m: "Appendices " + ref(m), s)
    s = re.sub(r"Tables~\\ref\{([^{}]*)\}", lambda m: "Tables " + ref(m), s)
    s = re.sub(r"Table~\\ref\{([^{}]*)\}", lambda m: "Table " + ref(m), s)
    s = re.sub(r"Figure~\\ref\{([^{}]*)\}", lambda m: "Figure " + ref(m), s)
    s = re.sub(r"\\ref\{([^{}]*)\}", ref, s)
    s = re.sub(r"\\label\{[^{}]*\}", "", s)
    s = demath(s)
    s = innermost(r"\\texttt\{([^{}]*)\}", r"`\1`", s)
    s = innermost(r"\\textbf\{([^{}]*)\}", r"**\1**", s)
    s = innermost(r"\\emph\{([^{}]*)\}", r"*\1*", s)
    s = innermost(r"\\textit\{([^{}]*)\}", r"*\1*", s)
    s = re.sub(r"\\(small|footnotesize|centering|toprule|midrule|bottomrule|"
               r"hline|scriptsize|normalsize)\b", "", s)
    s = re.sub(r"\\addlinespace(\[[^\]]*\])?", "", s)
    s = re.sub(r"\\multicolumn\{\d+\}\{[^{}]*\}\{([^{}]*)\}", r"\1", s)
    s = s.replace("\\\\", " ")
    s = s.replace("\\%", "%").replace("\\_", "_").replace("\\&", "&")
    s = s.replace("\\#", "#").replace("\\$", "$")
    s = s.replace("\\textbackslash", "\x00BS\x00")
    s = s.replace("\\{", "\x00LB\x00").replace("\\}", "\x00RB\x00")
    s = s.replace("\\ ", " ")
    s = re.sub(r"\\[a-zA-Z]+\b", "", s)
    s = s.replace("~", " ").replace("{", "").replace("}", "")
    s = s.replace("\x00BS\x00", "\\").replace("\x00LB\x00", "{").replace("\x00RB\x00", "}")
    s = s.replace("``", '"').replace("''", '"')
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def build_refmap(tex):
    """Map every \\label to its rendered number/letter."""
    refmap = {}
    sec_n, app_i, tab_n, fig_n = 0, 0, 0, 0
    sub_n = 0
    in_app = False
    cur = ""
    lines = tex.split("\n")
    for i, l in enumerate(lines):
        if l.startswith("\\appendix"):
            in_app = True
            continue
        m = re.match(r"\\section\{", l)
        if m and not l.startswith("\\section*"):
            if in_app:
                app_i += 1
                cur = chr(ord("A") + app_i - 1)
            else:
                sec_n += 1
                cur = str(sec_n)
            sub_n = 0
        elif re.match(r"\\subsection\{", l):
            sub_n += 1
            cur = ("%s.%d" % (sec_n, sub_n)) if not in_app else ("%s.%d" % (chr(ord("A") + app_i - 1), sub_n))
        elif re.match(r"\\begin\{table\}", l):
            tab_n += 1
            cur = str(tab_n)
        elif re.match(r"\\begin\{figure\}", l):
            fig_n += 1
            cur = str(fig_n)
        for lm in re.finditer(r"\\label\{([^{}]*)\}", l):
            refmap[lm.group(1)] = cur
    return refmap


def tabular_to_md(block, refmap):
    rows = []
    body = re.sub(r"\\begin\{tabular\}\{[^{}]*(\{[^{}]*\}[^{}]*)*\}", "", block)
    body = body.replace("\\end{tabular}", "")
    for raw in body.split("\\\\"):
        raw = raw.strip()
        if not raw:
            continue
        raw = re.sub(r"\\(toprule|midrule|bottomrule)\b", "", raw)
        raw = re.sub(r"\\addlinespace(\[[^\]]*\])?", "", raw)
        cells = [convert_inline(c, refmap) for c in raw.split("&")]
        cells = [c for c in cells if c != ""]
        if not cells:
            continue
        rows.append("- " + " | ".join(cells))
    return "\n\n".join(rows)


def convert_float(block, refmap):
    out = []
    cap = re.search(r"\\caption\{", block)
    caption = ""
    if cap:
        i = cap.end() - 1
        depth, j = 0, i
        while j < len(block):
            if block[j] == "{":
                depth += 1
            elif block[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        caption = convert_inline(block[i + 1:j], refmap)
    tab = re.search(r"\\begin\{tabular\}.*?\\end\{tabular\}", block, re.S)
    if tab:
        out.append(tabular_to_md(tab.group(0), refmap))
    inc = re.search(r"\\includegraphics(\[[^\]]*\])?\{([^{}]*)\}", block)
    if inc:
        out.append("![figure](%s)" % inc.group(2))
    if caption:
        out.append("*" + caption + "*")
    return "\n\n".join(x for x in out if x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anonymous", action="store_true")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    tex = open(TEX, encoding="utf-8").read()
    refmap = build_refmap(tex)

    title = re.search(r"\\title\{(.*?)\}\s*\n", tex, re.S)
    title_txt = re.sub(r"\s+", " ", title.group(1)).strip() if title else ""

    start = tex.index("\\begin{abstract}")
    end = tex.index("\\end{document}")
    body = tex[start:end]

    out = ["# " + title_txt, ""]
    out.append(ANON_HEADER if a.anonymous else DEANON_HEADER)

    sec_n, app_i, sub_n = 0, 0, 0
    in_app = False
    i = 0
    lines = body.split("\n")
    buf = []

    def flush():
        if not buf:
            return
        para = convert_inline(" ".join(buf), refmap)
        if para:
            out.append(para)
            out.append("")
        del buf[:]

    while i < len(lines):
        l = lines[i]
        if l.startswith("\\begin{abstract}"):
            flush(); out.append("## Abstract"); out.append(""); i += 1; continue
        if l.startswith("\\end{abstract}"):
            flush(); i += 1; continue
        if l.startswith("\\appendix"):
            flush(); in_app = True; sub_n = 0; i += 1; continue
        if l.startswith("\\maketitle") or l.startswith("\\bibliographystyle") \
           or l.startswith("\\bibliography"):
            i += 1; continue
        m = re.match(r"\\section\*?\{(.*)$", l)
        if m:
            flush()
            head = m.group(1)
            while head.count("{") + 1 > head.count("}") + 1 or not head.rstrip().endswith("}"):
                i += 1
                head += " " + lines[i]
            head = head.rstrip().rstrip("}")
            head = convert_inline(head, refmap)
            if l.startswith("\\section*"):
                out.append("## " + head)
            elif in_app:
                app_i += 1
                out.append("## Appendix %s: %s" % (chr(ord("A") + app_i - 1), head))
            else:
                sec_n += 1
                out.append("## %d. %s" % (sec_n, head))
            sub_n = 0
            out.append("")
            i += 1
            continue
        m = re.match(r"\\subsection\{(.*)$", l)
        if m:
            flush()
            head = m.group(1)
            while not head.rstrip().endswith("}"):
                i += 1
                head += " " + lines[i]
            head = head.rstrip().rstrip("}")
            head = convert_inline(head, refmap)
            sub_n += 1
            pref = chr(ord("A") + app_i - 1) if in_app else str(sec_n)
            out.append("### %s.%d %s" % (pref, sub_n, head))
            out.append("")
            i += 1
            continue
        m = re.match(r"\\begin\{(table|figure)\}", l)
        if m:
            flush()
            envname = m.group(1)
            block = []
            while i < len(lines) and not lines[i].startswith("\\end{%s}" % envname):
                block.append(lines[i]); i += 1
            block.append(lines[i] if i < len(lines) else "")
            i += 1
            conv = convert_float("\n".join(block), refmap)
            if conv:
                out.append(conv); out.append("")
            continue
        if re.match(r"\\begin\{enumerate\}", l) or re.match(r"\\end\{enumerate\}", l):
            flush(); i += 1; continue
        if re.match(r"\\setcounter", l):
            i += 1; continue
        if re.match(r"\s*\\item\b", l):
            flush()
            item = [re.sub(r"^\s*\\item\s*", "", l)]
            i += 1
            while i < len(lines) and not re.match(r"\s*\\item\b", lines[i]) \
                    and not re.match(r"\\end\{enumerate\}", lines[i]) \
                    and not re.match(r"\\begin\{", lines[i]) \
                    and lines[i].strip() != "":
                item.append(lines[i]); i += 1
            txt = convert_inline(" ".join(item), refmap)
            if txt:
                out.append("-  " + txt); out.append("")
            continue
        if l.strip() == "":
            flush(); i += 1; continue
        buf.append(l)
        i += 1
    flush()

    wrapped = []
    for blk in out:
        if blk.startswith("#") or blk.startswith("!") or not blk.strip() \
           or len(blk) <= 78:
            wrapped.append(blk)
        elif blk.startswith("-  "):
            wrapped.append("\n".join(textwrap.wrap(
                blk, 78, subsequent_indent="   ")))
        else:
            wrapped.append("\n".join(textwrap.wrap(blk, 78)))
    txt = "\n".join(wrapped)
    txt = re.sub(r"\n{3,}", "\n\n", txt).rstrip() + "\n"
    open(a.out, "w", encoding="utf-8").write(txt)
    print("wrote %s (%d lines)" % (a.out, txt.count("\n")))


if __name__ == "__main__":
    main()
