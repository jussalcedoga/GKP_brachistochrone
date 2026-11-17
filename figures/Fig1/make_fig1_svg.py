"""Make figures/Fig1/fig1.svg: Fig. 1 with editable text (no LaTeX) and one Inkscape layer per panel.
Usage, from the repository root:   source env.sh && python figures/Fig1/make_fig1_svg.py
"""
import sys, pathlib, re, io
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from common import style as S

repo = pathlib.Path(__file__).resolve().parents[2]; out = repo / "figures/Fig1/fig1_raw.svg"
for f in pathlib.Path("/usr/share/fonts/truetype/liberation").glob("LiberationSans-*.ttf"):
    fm.fontManager.addfont(str(f))
FAM = "Liberation Sans"   # Arial/Helvetica metrics, so the layout matches the published figure
# Nov 16 (author): Greek letters in the serif math font (Latin Modern Math, install latinmodern-math.otf for Inkscape),
# all other text and math in Helvetica/Arial; see src/svg_math.py
from svg_math import greek_math, setup_fonts_greek
_MATH_RC = setup_fonts_greek(fm, FAM, [pathlib.Path(__file__).resolve().parent])


_style = S.style
def style(usetex=True):
    _style(usetex=False)
    plt.rcParams.update({"svg.fonttype": "none", "font.sans-serif": [FAM, "DejaVu Sans"],
                         "mathtext.fontset": "custom", "mathtext.rm": FAM, "mathtext.it": FAM + ":italic",
                         "mathtext.bf": FAM + ":bold", "mathtext.sf": FAM, "mathtext.fallback": "stixsans", "mathtext.cal": "Latin Modern Math",
                         "svg.hashsalt": "fig1"})
    plt.rcParams.update(_MATH_RC)
S.style = style

def save(fig, path, dpi=300):
    axs = fig.axes
    # order of creation in plot_fig1.py: b, a, n-colorbar, (c) 5 + cbar, (d) 5 + cbar
    names = ["panel_b_sphere", "panel_a_sphere", "colorbar_photon_number"]
    for r in "cd":
        names += [f"panel_{r}_t{k+1}" for k in range(5)] + [f"panel_{r}_colorbar"]
    for ax, n in zip(axs, names):
        ax.set_gid(n)
    for lg in fig.legends:
        lg.set_gid("legend")
    fig.savefig(out, format="svg", bbox_inches="tight", pad_inches=0.12, dpi=dpi)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", pad_inches=0.12, dpi=80)
    plt.close(fig)
S.save = save

script = repo / "figures/Fig1/plot_fig1.py"
src = script.read_text().replace(r"\mathcal{{F}}", "ℱ").replace(r"\bar n", r"\overline{n}")   # script F as a Unicode character (no LaTeX)
src = greek_math(src)
code = compile(src, str(script), "exec")
exec(code, {"__name__": "__main__", "__file__": str(script)})


# ---- regroup into Inkscape layers ----
src, dst = str(out), str(out.with_name('fig1.svg'))
SVG = "http://www.w3.org/2000/svg"; INK = "http://www.inkscape.org/namespaces/inkscape"
ET.register_namespace("", SVG); ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
ET.register_namespace("inkscape", INK)
tree = ET.parse(src); root = tree.getroot()
fig = root.find(f"{{{SVG}}}g[@id='figure_1']")
layers = [("legend", "Legend", lambda i: i == "legend"),
          ("layer_a", "(a) preparation sphere", lambda i: i.startswith("panel_a")),
          ("layer_b", "(b) logical-flip sphere", lambda i: i.startswith("panel_b")),
          ("layer_nbar", "Photon-number colorbar", lambda i: i == "colorbar_photon_number"),
          ("layer_c", "(c) Wigner snapshots, preparation", lambda i: i.startswith("panel_c")),
          ("layer_d", "(d) Wigner snapshots, logical flip", lambda i: i.startswith("panel_d"))]
kids = list(fig)
newg = {}
for lid, lab, _ in layers:
    g = ET.Element(f"{{{SVG}}}g", {"id": lid, f"{{{INK}}}label": lab, f"{{{INK}}}groupmode": "layer"})
    newg[lid] = g
other = []
for el in kids:
    i = el.get("id", "")
    for lid, lab, test in layers:
        if test(i):
            el.set(f"{{{INK}}}label", i); newg[lid].append(el); fig.remove(el); break
    else:
        if i != "patch_1": other.append(i)
# background patch stays first; layers go to the root so Inkscape treats them as real layers
root.remove(fig)
bg = ET.Element(f"{{{SVG}}}g", {"id": "background", f"{{{INK}}}label": "Background", f"{{{INK}}}groupmode": "layer",
                                f"{{{INK}}}insensitive": "true"})
for el in list(fig): bg.append(el)
root.append(bg)
for lid, *_ in layers: root.append(newg[lid])
tree.write(dst, xml_declaration=True, encoding="utf-8")
s = open(dst, encoding="utf-8").read()
import re as _re
s = _re.sub(r"font-family:\s*'Liberation Sans'(, 'DejaVu Sans', sans-serif)?",
            "font-family: Arial, Helvetica, 'Liberation Sans', 'Nimbus Sans', sans-serif", s)
s = s.replace("font-family: 'Latin Modern Math'", "font-family: 'Latin Modern Math', 'STIX Two Math', 'Cambria Math', serif")
s = s.replace("font-family: 'STIXGeneral'", "font-family: 'STIX Two Math', 'Cambria Math', 'STIXGeneral', 'DejaVu Sans', serif")
open(dst, "w", encoding="utf-8").write(s)
print("left ungrouped:", other)

pathlib.Path(src).unlink(); out.with_suffix('.png').unlink(missing_ok=True)
print('wrote', dst)
