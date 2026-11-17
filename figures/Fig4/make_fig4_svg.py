"""Make figures/Fig4/fig4.svg: Fig. 4 (Letter two-column layout, --wide) with editable text (no LaTeX), Greek letters
in the serif math font, and one Inkscape layer per panel.
Usage, from the repository root:   source env.sh && python figures/Fig4/make_fig4_svg.py
"""
import sys, pathlib, re
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from common import style as S

repo = pathlib.Path(__file__).resolve().parents[2]
here = repo / "figures/Fig4"
for f in pathlib.Path("/usr/share/fonts/truetype/liberation").glob("LiberationSans-*.ttf"):
    fm.fontManager.addfont(str(f))
FAM = "Liberation Sans"   # Arial/Helvetica metrics
from svg_math import greek_math, setup_fonts_greek
_MATH_RC = setup_fonts_greek(fm, FAM, [pathlib.Path(__file__).resolve().parent])

_style = S.style
def style(usetex=True):
    _style(usetex=False)
    plt.rcParams.update({"svg.fonttype": "none", "font.sans-serif": [FAM, "DejaVu Sans"], "svg.hashsalt": "fig4"})
    plt.rcParams.update(_MATH_RC)
S.style = style

import mfcommon                                  # LaTeX-only spellings in the shared names
mfcommon.GATE_NAME = {k: v.replace("--", "–") for k, v in mfcommon.GATE_NAME.items()}

def _cal_F_label(fig):
    """Nov 16 (author): the y label F_T of (c) with the calligraphic F of the LaTeX original (Computer Modern's
    \\mathcal{F}, from cmsy10), drawn as an outline since no installed font has that glyph at a Unicode code point;
    the subscript T in Helvetica/Arial italic, as in the original (sfmath)"""
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import PathPatch
    from matplotlib.textpath import TextPath
    from matplotlib.transforms import Affine2D
    ax = next(a for a in fig.axes if a.get_gid() == "panel_c_i")
    lab = ax.yaxis.label
    fs = lab.get_fontsize()
    fig.canvas.draw()
    bb = lab.get_window_extent(fig.canvas.get_renderer())
    # centre of the original label in axes coordinates: anchored to the axes, the shape moves with the panel when
    # savefig crops the figure (bbox_inches="tight"); absolute page coordinates would leave it behind
    cxa, cya = ax.transAxes.inverted().transform((0.5 * (bb.x0 + bb.x1), 0.5 * (bb.y0 + bb.y1)))
    pF = TextPath((0, 0), r"$\mathcal{F}$", size=fs, prop=FontProperties(math_fontfamily="cm"))
    eF = pF.get_extents()
    pT = TextPath((eF.x1 + 0.04 * fs, -0.22 * fs), "T", size=0.70 * fs,
                  prop=FontProperties(family=FAM, style="italic"))
    eT = pT.get_extents()
    x0, x1 = min(eF.x0, eT.x0), max(eF.x1, eT.x1)
    from matplotlib.transforms import ScaledTranslation
    tr = (Affine2D().translate(-0.5 * (x0 + x1), -0.30 * fs).rotate_deg(90).scale(1 / 72) + fig.dpi_scale_trans
          + ScaledTranslation(cxa, cya, ax.transAxes))
    for pth, gid in ((pF, "label_calF"), (pT, "label_calF_T")):
        patch = PathPatch(pth, transform=tr, fc=lab.get_color(), ec="none", lw=0)
        patch.set_gid(gid)
        ax.add_artist(patch)
        patch.set_clip_on(False)
    lab.set_visible(False)


def save(fig, path, dpi=300):
    path = pathlib.Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    names = []
    for col in ("i", "ii", "iii"):
        names += [f"panel_a_{col}", f"panel_b_{col}", f"panel_c_{col}"]
    for row in ("path", "CS", "CD", "SNAP"):
        names += [f"panel_d_{row}_t{j + 1}" for j in range(5)]
    names += ["panel_d_colorbar"]
    for ax, n in zip(fig.axes, names):
        ax.set_gid(n)
    for i, lg in enumerate(fig.legends):
        lg.set_gid("legend" if i == 0 else f"legend{i}")
    _cal_F_label(fig)
    fig.savefig(path.with_suffix(".svg"), format="svg", bbox_inches="tight", pad_inches=0.12, dpi=dpi)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", pad_inches=0.12, dpi=110)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
S.save = save

script = here / "plot_fig4.py"
src = (script.read_text().replace(r"\mathcal{{F}}", "ℱ").replace(r"\mathcal{F}", "ℱ")
       .replace(r"\enspace", " ").replace(r"\%", "%"))
src = greek_math(src)
sys.argv = [str(script), "--wide"]
g = {"__name__": "__main__", "__file__": str(script)}
exec(compile(src, str(script), "exec"), g)
raw = pathlib.Path(g["OUT"]) / f"{g['STEM']}.svg"

# ---- regroup into Inkscape layers ----
SVG = "http://www.w3.org/2000/svg"; INK = "http://www.inkscape.org/namespaces/inkscape"
ET.register_namespace("", SVG); ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
ET.register_namespace("inkscape", INK)
tree = ET.parse(raw); root = tree.getroot()
figg = root.find(f"{{{SVG}}}g[@id='figure_1']")
def layer(lid, lab):
    return ET.Element(f"{{{SVG}}}g", {"id": lid, f"{{{INK}}}label": lab, f"{{{INK}}}groupmode": "layer"})
bg = layer("background", "Background"); bg.set(f"{{{INK}}}insensitive", "true")
L = {"a": layer("layer_a", "(a) state in the plane of |0> and |chi>"), "b": layer("layer_b", "(b) closeness F"),
     "c": layer("layer_c", "(c) target fidelity"), "d": layer("layer_d", "(d) Wigner functions"),
     "legend": layer("layer_legend", "Legend")}
for el in list(figg):
    i = el.get("id", "")
    m = re.match(r"panel_([abcd])", i)
    if m:
        el.set(f"{{{INK}}}label", i); L[m.group(1)].append(el)
    elif i.startswith("legend"):
        L["legend"].append(el)
    elif i.startswith("text"):                   # figure-level text: row labels of (d) and W(q,p)
        L["d"].append(el)
    else:
        bg.append(el)
root.remove(figg); root.append(bg)
for v in L.values():
    if len(v): root.append(v)
out = here / "fig4.svg"
tree.write(out, xml_declaration=True, encoding="utf-8")
s = out.read_text(encoding="utf-8")
s = re.sub(r"font-family:\s*'Liberation Sans'(, 'DejaVu Sans', sans-serif)?",
           "font-family: Arial, Helvetica, 'Liberation Sans', 'Nimbus Sans', sans-serif", s)
s = s.replace("font-family: 'Latin Modern Math'", "font-family: 'Latin Modern Math', 'STIX Two Math', 'Cambria Math', serif")
s = s.replace("font-family: 'STIXGeneral'", "font-family: 'STIX Two Math', 'Cambria Math', 'STIXGeneral', 'DejaVu Sans', serif")
out.write_text(s, encoding="utf-8")
print("wrote", out)
