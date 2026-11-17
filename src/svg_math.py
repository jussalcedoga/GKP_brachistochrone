"""Math typography for the editable (no-LaTeX) SVG versions of the figures (figures/FigN/make_figN_svg.py).

Option 2 of the Nov 16 discussion: words and numbers in Helvetica/Arial, every math symbol in LaTeX's math font
(Latin Modern Math, the OpenType Computer Modern), so the math in the figures looks like the math in the captions.

    setup_fonts(font_manager)  registers Latin Modern Math; returns the rcParams for matplotlib's math text
    cm_math(source)            rewrites the math in the string literals of a plotting script:
                                 Latin and Greek letters -> mathematical italic characters (U+1D434..., U+1D6FC...),
                                 so they are drawn in Computer Modern italic; the contents of \\mathrm{...},
                                 \\text{...}, \\mathsf{...} and \\mathcal{...} are left alone (upright Re, etc.);
                                 numbers computed at run time (f-string fields such as {nb:.2f}) -> \\mathsf{...},
                                 i.e. Helvetica/Arial like every other number in the figures;
                                 strings that are only a number ("$%d$", "$0$") -> plain text (Helvetica/Arial).
"""
from __future__ import annotations

import pathlib
import re
import subprocess

LMM = "Latin Modern Math"
_GREEK = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda", "mu", "nu",
          "xi", "omicron", "pi", "rho", "varsigma", "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega"]
_KEEP_GROUP = {"mathrm", "text", "mathsf", "mathcal", "operatorname"}


def _find_lmm(extra_dirs=()):
    cands = [pathlib.Path(d) / "latinmodern-math.otf" for d in extra_dirs]
    try:
        out = subprocess.run(["kpsewhich", "latinmodern-math.otf"], capture_output=True, text=True).stdout.strip()
        if out:
            cands.insert(0, pathlib.Path(out))
    except OSError:
        pass
    for c in cands:
        if c.is_file():
            return c
    raise FileNotFoundError("latinmodern-math.otf not found: install TeX's lm-math package, or put the font next to "
                            "the make_figN_svg.py script")


def setup_fonts(font_manager, text_family, extra_dirs=()):
    """register Latin Modern Math; rcParams: math in Latin Modern Math, \\mathsf (run-time numbers) in the text font"""
    font_manager.fontManager.addfont(str(_find_lmm(extra_dirs)))
    return {"mathtext.fontset": "custom", "mathtext.rm": LMM, "mathtext.it": LMM, "mathtext.bf": LMM,
            "mathtext.cal": LMM, "mathtext.sf": text_family, "mathtext.fallback": "stix"}


def _italic(ch):
    if ch == "h":
        return "ℎ"                                   # Planck constant: the math-italic h
    if "a" <= ch <= "z":
        return chr(0x1D44E + ord(ch) - ord("a"))
    return chr(0x1D434 + ord(ch) - ord("A"))


def _wrap(ch, f):
    """matplotlib finds the math-italic characters of Latin Modern Math only through its \\mathcal font slot
    (set to Latin Modern Math in setup_fonts): through the italic slot they fall back to STIX"""
    return ("\\mathcal{{" + ch + "}}") if f else ("\\mathcal{" + ch + "}")


def _group_end(s, i, f):
    """index after the TeX group starting at s[i] ('{' in a raw string, '{{' in an f-string)"""
    op, cl = ("{{", "}}") if f else ("{", "}")
    depth, j = 0, i
    while j < len(s):
        if s.startswith(op, j):
            depth += 1
            j += len(op)
        elif s.startswith(cl, j):
            depth -= 1
            j += len(cl)
            if depth == 0:
                return j
        else:
            j += 1
    return len(s)


def _field_end(s, i):
    """index after the f-string replacement field starting at s[i] == '{' (not '{{')"""
    depth, j = 0, i
    while j < len(s):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return j + 1
        j += 1
    return len(s)


def _math(seg, f):
    out, i = [], 0
    while i < len(seg):
        c = seg[i]
        if c == "\\":
            m = re.match(r"\\([A-Za-z]+)", seg[i:])
            if not m:                                     # \, \! \; \| ...
                out.append(seg[i:i + 2])
                i += 2
                continue
            name = m.group(1)
            i += len(m.group(0))
            if name in ("geq", "leq", "neq", "approx"):        # relations: LaTeX's thick space on both sides
                out.append("\\,\\" + name + "\\,")
                continue
            if name in _GREEK:
                out.append(_wrap(chr(0x1D6FC + _GREEK.index(name)), f))
            elif name in _KEEP_GROUP and i < len(seg) and seg[i] == "{":
                j = _group_end(seg, i, f)
                out.append("\\" + name + seg[i:j])
                i = j
            else:
                out.append("\\" + name)
            continue
        if f and seg.startswith("{{", i):
            out.append("{{")
            i += 2
            continue
        if f and seg.startswith("}}", i):
            out.append("}}")
            i += 2
            continue
        if f and c == "{":                                # run-time value: a number, in the text font
            j = _field_end(seg, i)
            out.append("\\mathsf{{" + seg[i:j] + "}}")
            i = j
            continue
        if c == "=":                                      # relation spacing as in LaTeX
            out.append("\\,=\\,")
            i += 1
            continue
        if c.isascii() and c.isalpha():
            out.append(_wrap(_italic(c), f))
            i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _body(body, f):
    if re.fullmatch(r"\$[-+\d.\s%d]*\$", body):          # only a number: plain text
        return body[1:-1]
    parts = body.split("$")
    if len(parts) < 3:
        return body
    return "$".join(_math(p, f) if k % 2 else p for k, p in enumerate(parts))


def cm_math(source):
    """rewrite the math of every raw string literal (r"..." / rf"...") of a plotting script"""
    def sub(m):
        pre, body = m.group(1), m.group(2)
        if "$" not in body:
            return m.group(0)
        return pre + '"' + _body(body, "f" in pre.lower()) + '"'
    return re.sub(r'(?<![A-Za-z0-9_])((?:rf|fr|r))"([^"\n]*)"', sub, source)


# ---------------------------------------------------------------------------------------------- Greek-only (in use)
# Nov 16 (author): option 2 above was tried and rejected; the figures use this instead: only the Greek letters in the
# serif math font (Latin Modern Math italic, as Computer Modern's math italic in the LaTeX original), everything else --
# words, numbers, Latin variables, symbols -- in Helvetica/Arial.

# LaTeX's \epsilon and \phi are the lunate/closed forms; \varepsilon, \varphi the open ones
_GREEK_CP = {"epsilon": 0x1D716, "varepsilon": 0x1D700, "phi": 0x1D719, "varphi": 0x1D711, "vartheta": 0x1D717}


def _greek_cp(name):
    return _GREEK_CP.get(name, 0x1D6FC + _GREEK.index(name) if name in _GREEK else None)


def setup_fonts_greek(font_manager, text_family, extra_dirs=()):
    """register Latin Modern Math; rcParams: math text in the text font, \\mathcal (the Greek letters) in Latin Modern"""
    font_manager.fontManager.addfont(str(_find_lmm(extra_dirs)))
    return {"mathtext.fontset": "custom", "mathtext.rm": text_family, "mathtext.it": text_family + ":italic",
            "mathtext.bf": text_family + ":bold", "mathtext.sf": text_family, "mathtext.cal": LMM,
            "mathtext.fallback": "stixsans"}


def greek_math(source):
    """\\pi -> the mathematical-italic pi of Latin Modern Math (U+1D70B), likewise for every lowercase Greek letter;
    in f-strings the TeX braces are doubled"""
    pat = r"\\(" + "|".join(sorted(set(_GREEK) | set(_GREEK_CP), key=len, reverse=True)) + r")(?![A-Za-z])"

    def lit(m):
        pre, body = m.group(1), m.group(2)
        op, cl = ("{{", "}}") if "f" in pre.lower() else ("{", "}")
        body = re.sub(pat, lambda g: "\\mathcal" + op + chr(_greek_cp(g.group(1))) + cl, body)
        return pre + '"' + body + '"'
    return re.sub(r'(?<![A-Za-z0-9_])((?:rf|fr|r|f)?)"([^"\n]*)"', lit, source)
