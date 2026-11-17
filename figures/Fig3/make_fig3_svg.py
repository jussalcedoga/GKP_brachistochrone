"""Make figures/Fig3/fig3.svg: Fig. 3 with editable text (no LaTeX) and one Inkscape layer per panel.
Usage, from the repository root:   source env.sh && python figures/Fig3/make_fig3_svg.py
"""
import sys, pathlib, re
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from common import style as S

repo = pathlib.Path(__file__).resolve().parents[2]
here = repo / "figures/Fig3"
for f in pathlib.Path("/usr/share/fonts/truetype/liberation").glob("LiberationSans-*.ttf"):
    fm.fontManager.addfont(str(f))
FAM = "Liberation Sans"   # Arial/Helvetica metrics
# Nov 16 (author): Greek letters in the serif math font (Latin Modern Math, install latinmodern-math.otf for Inkscape),
# all other text and math in Helvetica/Arial; see src/svg_math.py
from svg_math import greek_math, setup_fonts_greek
_MATH_RC = setup_fonts_greek(fm, FAM, [pathlib.Path(__file__).resolve().parent])

NAMES = ["panel_b_schematic", "panel_a_prep_time", "panel_a_prep_photon", "panel_a_flip_time", "panel_a_flip_photon", "panel_a_colorbar"]

_style = S.style
def style(usetex=True):
    _style(usetex=False)
    plt.rcParams.update({"svg.fonttype": "none", "font.sans-serif": [FAM, "DejaVu Sans"],
                         "mathtext.fontset": "custom", "mathtext.rm": FAM, "mathtext.it": FAM + ":italic",
                         "mathtext.bf": FAM + ":bold", "mathtext.sf": FAM, "mathtext.fallback": "stixsans", "mathtext.cal": "Latin Modern Math",
                         "svg.hashsalt": "fig3"})
    plt.rcParams.update(_MATH_RC)
S.style = style

def save(fig, path, dpi=300):
    path = pathlib.Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    for ax, n in zip(fig.axes, NAMES):
        ax.set_gid(n)
    for i, lg in enumerate(fig.legends):
        lg.set_gid("legend" if i == 0 else f"legend{i}")
    fig.savefig(path.with_suffix(".svg"), format="svg", bbox_inches="tight", pad_inches=0.12, dpi=dpi)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", pad_inches=0.12, dpi=110)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
S.save = save

script = here / "plot_fig3.py"
src = (script.read_text().replace('rf"$\\bar n=', 'rf"$\\overline{{n}}=').replace(r"\mathcal{{F}}", "ℱ").replace(r"\mathcal{F}", "ℱ")
       .replace(r"\bar n", r"\overline{n}").replace(r"\int_0^t\!\bar", r"\int_0^t\,\bar")
       .replace(r"\int_0^t\!\overline", r"\int_0^t\,\overline"))
# \dot p: matplotlib's math text writes the dot as a lone combining accent, which no font places over the p; keep
# "p(t_k)" as text and draw the dot as a small marker above the p
src = src.replace('''ax.text(*(pk + 0.26 * tang + np.array([0.03, 0.02])), r"$\\dot p(t_k)$", fontsize=FS["ann"], color=PATHC, zorder=9,
        ha="left", va="bottom")''', '''_xy = pk + 0.26 * tang + np.array([0.03, 0.02])
_lt = ax.annotate(r"$p(t_k)$", _xy, xytext=(0, 0.22 * FS["ann"]), textcoords="offset points", fontsize=FS["ann"],
                  color=PATHC, zorder=9, ha="left", va="baseline")
ax.plot([_xy[0]], [_xy[1]], ls="none", marker="o", ms=0.17 * FS["ann"], color=PATHC, zorder=9,
        transform=ax.transData + matplotlib.transforms.ScaledTranslation(0.36 * FS["ann"] / 72,
                                                                           (0.22 + 0.80) * FS["ann"] / 72,
                                                                           fig.dpi_scale_trans))''')
assert "_lt = ax.annotate" in src, "dot-p replacement did not apply"
src = greek_math(src)
_args = sys.argv[1:]
sys.argv = [str(script)]
g = {"__name__": "__main__", "__file__": str(script)}
exec(compile(src, str(script), "exec"), g)
raw = pathlib.Path(g['HERE']) / "fig3.svg"

# ---- regroup into Inkscape layers ----
SVG = "http://www.w3.org/2000/svg"; INK = "http://www.inkscape.org/namespaces/inkscape"
ET.register_namespace("", SVG); ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
ET.register_namespace("inkscape", INK)
tree = ET.parse(raw); root = tree.getroot()
figg = root.find(f"{{{SVG}}}g[@id='figure_1']")
def layer(lid, lab):
    return ET.Element(f"{{{SVG}}}g", {"id": lid, f"{{{INK}}}label": lab, f"{{{INK}}}groupmode": "layer"})
bg = layer("background", "Background"); bg.set(f"{{{INK}}}insensitive", "true")
layers = {"a": layer("layer_a", "(a) Wigner functions on the paths"), "b": layer("layer_b", "(b) path-following schematic")}
for el in list(figg):
    i = el.get("id", "")
    if i.startswith("panel_a"):
        el.set(f"{{{INK}}}label", i); layers["a"].append(el)
    elif i.startswith("panel_b"):
        el.set(f"{{{INK}}}label", i); layers["b"].append(el)
    elif i.startswith("text"):                   # figure-level text: panel letters and the pair titles
        txt = "".join(el.itertext())
        layers["b" if "(b)" in txt else "a"].append(el)
    else:
        bg.append(el)
root.remove(figg); root.append(bg)
for v in layers.values():
    if len(v): root.append(v)
out = here / "fig3.svg"
tree.write(out, xml_declaration=True, encoding="utf-8")
s = out.read_text(encoding="utf-8")
s = re.sub(r"font-family:\s*'Liberation Sans'(, 'DejaVu Sans', sans-serif)?",
           "font-family: Arial, Helvetica, 'Liberation Sans', 'Nimbus Sans', sans-serif", s)
s = s.replace("font-family: 'Latin Modern Math'", "font-family: 'Latin Modern Math', 'STIX Two Math', 'Cambria Math', serif")
s = s.replace("font-family: 'STIXGeneral'", "font-family: 'STIX Two Math', 'Cambria Math', 'STIXGeneral', 'DejaVu Sans', serif")
out.write_text(s, encoding="utf-8")
print("wrote", out)
