"""Make figures/Fig2/fig2.svg: the chosen Fig. 2 candidate with editable text (no LaTeX) and one Inkscape layer per panel.
Usage, from the repository root:   source env.sh && python figures/Fig2/make_fig2_svg.py
"""
import sys, pathlib, re
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from common import style as S

repo = pathlib.Path(__file__).resolve().parents[2]
here = repo / "figures/Fig2"
for f in pathlib.Path("/usr/share/fonts/truetype/liberation").glob("LiberationSans-*.ttf"):
    fm.fontManager.addfont(str(f))
FAM = "Liberation Sans"   # Arial/Helvetica metrics
# Nov 16 (author): Greek letters in the serif math font (Latin Modern Math, install latinmodern-math.otf for Inkscape),
# all other text and math in Helvetica/Arial; see src/svg_math.py
from svg_math import greek_math, setup_fonts_greek
_MATH_RC = setup_fonts_greek(fm, FAM, [pathlib.Path(__file__).resolve().parent])

LETTERS = "abcdef"

_style = S.style
def style(usetex=True):
    _style(usetex=False)
    plt.rcParams.update({"svg.fonttype": "none", "font.sans-serif": [FAM, "DejaVu Sans"],
                         "mathtext.fontset": "custom", "mathtext.rm": FAM, "mathtext.it": FAM + ":italic",
                         "mathtext.bf": FAM + ":bold", "mathtext.sf": FAM, "mathtext.fallback": "stixsans", "mathtext.cal": "Latin Modern Math",
                         "svg.hashsalt": "fig2"})
    plt.rcParams.update(_MATH_RC)
S.style = style

def save(fig, path, dpi=300):
    path = pathlib.Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    for ax, l in zip(fig.axes, LETTERS):
        ax.set_gid(f"panel_{l}")
    for i, lg in enumerate(fig.legends):
        lg.set_gid("legend" if i == 0 else f"legend{i}")
    fig.savefig(path.with_suffix(".svg"), format="svg", bbox_inches="tight", pad_inches=0.12, dpi=dpi)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", pad_inches=0.12, dpi=110)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
S.save = save

script = here / "plot_fig2.py"
src = (script.read_text().replace(r"\mathcal{{F}}", "ℱ").replace(r"\mathcal{F}", "ℱ")
       .replace(r"\bar n", r"\overline{n}").replace(r"\int_0^t\!\bar", r"\int_0^t\,\bar")
       .replace(r"\int_0^t\!\overline", r"\int_0^t\,\overline"))
src = greek_math(src)
_args = sys.argv[1:]
sys.argv = [str(script)]
g = {"__name__": "__main__", "__file__": str(script)}
exec(compile(src, str(script), "exec"), g)
cand = _args[0] if _args else g["CHOOSE"]      # e.g. "D" for the proposed n-vs-fidelity bottom row
raw = pathlib.Path(g['HERE']) / "candidates" / f"fig2_{cand}.svg"

# ---- regroup into Inkscape layers ----
SVG = "http://www.w3.org/2000/svg"; INK = "http://www.inkscape.org/namespaces/inkscape"
ET.register_namespace("", SVG); ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
ET.register_namespace("inkscape", INK)
tree = ET.parse(raw); root = tree.getroot()
figg = root.find(f"{{{SVG}}}g[@id='figure_1']")
def layer(lid, lab):
    return ET.Element(f"{{{SVG}}}g", {"id": lid, f"{{{INK}}}label": lab, f"{{{INK}}}groupmode": "layer"})
bg = layer("background", "Background"); bg.set(f"{{{INK}}}insensitive", "true")
layers = {"legend": layer("layer_legend", "Legend")}
for l in LETTERS:
    layers[f"panel_{l}"] = layer(f"layer_{l}", f"({l})")
for el in list(figg):
    i = el.get("id", "")
    key = "legend" if i.startswith("legend") else i
    (layers[key] if key in layers else bg).append(el)
root.remove(figg); root.append(bg)
for v in layers.values():
    if len(v): root.append(v)
out = here / ("fig2.svg" if cand == g["CHOOSE"] else f"fig2_{cand}.svg")
tree.write(out, xml_declaration=True, encoding="utf-8")
s = out.read_text(encoding="utf-8")
s = re.sub(r"font-family:\s*'Liberation Sans'(, 'DejaVu Sans', sans-serif)?",
           "font-family: Arial, Helvetica, 'Liberation Sans', 'Nimbus Sans', sans-serif", s)
s = s.replace("font-family: 'Latin Modern Math'", "font-family: 'Latin Modern Math', 'STIX Two Math', 'Cambria Math', serif")
s = s.replace("font-family: 'STIXGeneral'", "font-family: 'STIX Two Math', 'Cambria Math', 'STIXGeneral', 'DejaVu Sans', serif")
out.write_text(s, encoding="utf-8")
print("wrote", out)
