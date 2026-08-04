"""Interactive browser viewer for FormationTemps flux / formation-temperature spectra."""

import argparse, os, sys, webbrowser

import h5py
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Mask bit convention, mirroring scripts/bad_lines_mask.jl. Used only for files written
# before the mask_bit_meanings attribute existed; current files carry their own.
fallback_mask_bits = ((0x01, "boundary"), (0x02, "chromo"),
                      (0x04, "nlte"), (0x08, "linedata"))

# One colour per reason so a flagged stretch says why it was flagged. Boundary keeps the
# red it has always had. Fills are the same hues at low alpha, used for the shaded bands.
reason_colors = dict(boundary="#d62728", chromo="#9467bd",
                     nlte="#ff7f0e", linedata="#8c564b")
reason_fills = dict(boundary="rgba(214,39,40,0.18)", chromo="rgba(148,103,189,0.18)",
                    nlte="rgba(255,127,14,0.18)", linedata="rgba(140,86,75,0.18)")


# --- data loading --------------------------------------------------------


def as_str(x):
    # h5py hands back bytes or str depending on how the string was written.
    return x.decode() if isinstance(x, bytes) else str(x)


def read_mask_bits(f):
    # Decode the file's own bit->name table ("0x02 chromo") so this viewer cannot drift
    # from the Julia writer. Falls back to the built-in convention for older files, where
    # the mask was a plain boolean and bit 0 is still boundary contamination.
    raw = f.attrs.get("mask_bit_meanings", None)
    if raw is None:
        return list(fallback_mask_bits)
    bits = []
    for item in np.atleast_1d(raw):
        code, _, name = as_str(item).partition(" ")
        bits.append((int(code, 16), name.strip()))
    assert bits, "mask_bit_meanings present but empty"
    return bits


def decode_mask(mask_raw, mask_bits):
    # One boolean array per reason. A pixel can carry several bits at once — a saturated
    # core inside a curated region is both boundary-contaminated and NLTE-suspect — so
    # these overlap rather than partition.
    return {name: (mask_raw & bit) != 0 for bit, name in mask_bits}


def read_line_mask(g):
    # Provenance for the curated model-validity mask: one record per applied line. Absent
    # for groups with no curated line in range, and for files predating the mask.
    if "line_mask" not in g:
        return []
    lm = g["line_mask"]
    labels = [as_str(s) for s in lm["label"][:].ravel()]
    return [dict(lo=float(a), hi=float(b), flag=int(fl), label=lb)
            for a, b, fl, lb in zip(lm["lambda_lo"][:].ravel(),
                                    lm["lambda_hi"][:].ravel(),
                                    lm["flag"][:].ravel(), labels)]


def orient_cfunc(cfunc, n_wav, n_rows):
    # HDF5.jl stores Julia (Natm-1, Nlambda) arrays with reversed dims, so h5py
    # may read either orientation. Move the wavelength axis to axis 1.
    assert cfunc.ndim == 2, "cfunc must be 2D"
    if cfunc.shape == (n_rows, n_wav):
        return cfunc
    if cfunc.shape == (n_wav, n_rows):
        return cfunc.T
    raise ValueError(f"cfunc shape {cfunc.shape} matches neither "
                     f"({n_rows}, {n_wav}) nor ({n_wav}, {n_rows})")


def read_group(g, Ts, tau_ref, mask_bits):
    # Read one chunk group into the dict shape build_1d_figure expects.
    wavs = g["wavs"][:].ravel()
    flux = g["flux"][:].ravel()
    temp = g["temp"][:].ravel()
    # mask is a UInt8 bitflag; nonzero means flagged for at least one reason
    mask_raw = g["mask"][:].ravel()
    mask = mask_raw.astype(bool)
    reasons = decode_mask(mask_raw, mask_bits)
    line_centers = g["line_centers"][:].ravel()
    n_wav = wavs.size
    n_rows = Ts.size - 1                       # cfunc has Natm-1 layers
    cfunc = orient_cfunc(g["cfunc"][:], n_wav, n_rows)
    # mid-layer temperatures, matching plot_chunk_flux.jl
    t_mids = 0.5 * (Ts[:n_rows] + Ts[1:n_rows + 1])
    # Per-interval width of the reference optical-depth grid, in dex. The stored cfunc is a
    # per-interval INTEGRAL (cfunc_flux * Δτ_λ); the native MARCS grid changes spacing by 2x
    # at log τ_ref = -3 and +1, so plotting the integral directly imprints a horizontal,
    # grid-driven step of that size. Dividing by Δlog τ_ref gives the contribution density
    # per dex, which varies smoothly with depth. The physics (flux, formation temp) is
    # unaffected either way — those are sums, in which the interval width cancels.
    #
    # Not Δτ_ref: dividing by the linear interval is also a valid density, but dF/dτ_ref
    # varies by only ~1.5x between log τ_ref = -3 and the peak, against ~1300x for the per-dex
    # density, so it flattens three decades of real depth structure into one colour.
    dlogtau_ref = np.diff(np.log10(tau_ref))   # length n_rows
    return dict(wavs=wavs, flux=flux, temp=temp, mask=mask, reasons=reasons,
                line_mask=read_line_mask(g), line_centers=line_centers,
                cfunc=cfunc, t_mids=t_mids, dlogtau_ref=dlogtau_ref)


def read_tau_ref(f):
    # Fetch the per-layer reference optical depth (τ_ref) from model_atmosphere. The
    # dataset name carries a Unicode τ, so match defensively.
    ma = f["model_atmosphere"]
    key = next((k for k in ma.keys() if k.startswith("τ") or "tau" in k.lower()), None)
    assert key is not None, "no τ_ref dataset in model_atmosphere"
    return ma[key][:].ravel()


def load_1d(path):
    # Read the spliced single-spectrum group and everything the viewer needs.
    with h5py.File(path, "r") as f:
        assert "chunk_0001" in f, f"no 'chunk_0001' group in {path} (use --chunks?)"
        Ts = f["model_atmosphere"]["Ts"][:].ravel()
        return read_group(f["chunk_0001"], Ts, read_tau_ref(f), read_mask_bits(f))


def load_chunks(path):
    # Read every per-chunk group as a separate (wavs, flux, temp) record.
    chunks = []
    with h5py.File(path, "r") as f:
        names = sorted(n for n in f.keys() if n.startswith("chunk_"))
        assert names, f"no 'chunk_*' groups in {path}"
        for name in names:
            g = f[name]
            chunks.append(dict(name=name,
                               wavs=g["wavs"][:].ravel(),
                               flux=g["flux"][:].ravel(),
                               temp=g["temp"][:].ravel()))
    return chunks


def load_stitched(path):
    # Read the spliced/1D spectrum (wavs, flux, temp) to overlay on chunk pages. The 1D
    # file stores the spliced spectrum in a single group ("chunk_0001"); fall back to
    # top-level datasets if that group is absent.
    with h5py.File(path, "r") as f:
        g = f["chunk_0001"] if "chunk_0001" in f else f
        return dict(wavs=g["wavs"][:].ravel(),
                    flux=g["flux"][:].ravel(),
                    temp=g["temp"][:].ravel())


# --- transforms ----------------------------------------------------------


def range_keep(wavs, lo, hi):
    # Boolean keep-mask for a [lo, hi] wavelength window (None = unbounded).
    keep = np.ones(wavs.size, dtype=bool)
    if lo is not None:
        keep &= wavs >= lo
    if hi is not None:
        keep &= wavs <= hi
    return keep


def window_1d(d, lo, hi):
    # Restrict a 1D-group data dict to [lo, hi]; None if nothing survives.
    keep = range_keep(d["wavs"], lo, hi)
    if not keep.any():
        return None
    out = dict(d)
    for key in ("wavs", "flux", "temp", "mask"):
        out[key] = d[key][keep]
    out["reasons"] = {name: m[keep] for name, m in d["reasons"].items()}
    out["cfunc"] = d["cfunc"][:, keep]
    lc = d["line_centers"]
    out["line_centers"] = lc[range_keep(lc, lo, hi)]
    # keep curated regions that overlap the window, not only those contained in it
    lo_w, hi_w = float(out["wavs"].min()), float(out["wavs"].max())
    out["line_mask"] = [r for r in d["line_mask"]
                        if r["hi"] >= lo_w and r["lo"] <= hi_w]
    return out


def block_max_decimate(wavs, z, max_cols):
    # Reduce a (rows, Nlambda) array to <= max_cols columns via per-block max,
    # which preserves narrow line cores better than averaging. wavs collapses
    # to per-block means.
    n = z.shape[1]
    if n <= max_cols:
        return wavs, z
    edges = np.linspace(0, n, max_cols + 1).astype(int)
    z_dec = np.empty((z.shape[0], max_cols), dtype=z.dtype)
    w_dec = np.empty(max_cols, dtype=wavs.dtype)
    for k in range(max_cols):
        a, b = edges[k], max(edges[k + 1], edges[k] + 1)
        z_dec[:, k] = z[:, a:b].max(axis=1)
        w_dec[k] = wavs[a:b].mean()
    return w_dec, z_dec


# --- figure construction -------------------------------------------------


def mask_runs(mask):
    # Contiguous [start, stop) index runs where mask is True. A 7M-pixel spectrum yields
    # only a few hundred runs, so shading them costs far less than a full-length overlay.
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def mask_band_trace(wavs, mask, y_lo, y_hi, color, name, showlegend):
    # Shade flagged wavelength runs as background rectangles, one trace per reason.
    #
    # Mask state has to read as background rather than as line colour. A curated region is
    # routinely wider than the viewport — Halpha spans 34 Angstrom — and when the whole view
    # is flagged, recolouring the line leaves nothing to compare it against, so the mask
    # becomes invisible at exactly the zoom where it matters most.
    #
    # SVG Scatter rather than Scattergl: fill="toself" with NaN-separated polygons is
    # reliable there, and a few hundred rectangles are far cheaper than the data lines.
    runs = mask_runs(mask)
    if not runs:
        return None
    xs, ys = [], []
    last = wavs.size - 1
    for a, b in runs:
        # b is the first unflagged index, so the run's last flagged pixel is b - 1
        lo, hi = float(wavs[a]), float(wavs[min(b - 1, last)])
        xs += [lo, hi, hi, lo, lo, np.nan]
        ys += [y_lo, y_lo, y_hi, y_hi, y_lo, np.nan]
    return go.Scatter(x=xs, y=ys, mode="lines", fill="toself",
                      fillcolor=color, line=dict(width=0),
                      name=name, legendgroup=name, showlegend=showlegend,
                      hoverinfo="skip")


def region_band_trace(regions, y_band):
    # One toggleable trace of horizontal segments (None-separated) marking the curated
    # model-validity regions, drawn at the top of the flux panel with the line name on
    # hover. Overlapping regions stay separate segments rather than being merged, so the
    # provenance stays one-to-one with the HDF5 line_mask table.
    n = len(regions)
    xs = np.empty(3 * n)
    ys = np.full(3 * n, float(y_band))
    labels = []
    for k, r in enumerate(regions):
        xs[3 * k], xs[3 * k + 1], xs[3 * k + 2] = r["lo"], r["hi"], np.nan
        ys[3 * k + 2] = np.nan
        labels += [r["label"], r["label"], ""]
    return go.Scattergl(x=xs, y=ys, mode="lines",
                        line=dict(color="rgba(148,103,189,0.9)", width=6),
                        name="Curated regions", legendgroup="regions",
                        customdata=labels,
                        hovertemplate="%{customdata}<extra>curated</extra>")


def line_marker_trace(line_centers, y_lo, y_hi):
    # One toggleable trace of full-height vertical segments (None-separated) on
    # the flux panel; kept as a single trace so thousands of lines stay light.
    n = line_centers.size
    xs = np.empty(3 * n)
    ys = np.empty(3 * n)
    xs[0::3] = line_centers
    xs[1::3] = line_centers
    xs[2::3] = np.nan
    ys[0::3] = y_lo
    ys[1::3] = y_hi
    ys[2::3] = np.nan
    return go.Scattergl(x=xs, y=ys, mode="lines",
                        line=dict(color="rgba(120,120,120,0.35)", width=0.5),
                        name="Line centers", legendgroup="lines",
                        hovertemplate="%{x:.3f} Å<extra>line</extra>",
                        visible="legendonly")


def build_1d_figure(d, max_cols, stitched=None):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, row_heights=[0.25, 0.25, 0.5],
                        subplot_titles=("", "", ""))

    # Panel y-ranges, needed before the shading since the bands span them.
    f_lo, f_hi = float(np.nanmin(d["flux"])), float(np.nanmax(d["flux"]))
    t_lo, t_hi = float(np.nanmin(d["temp"])), float(np.nanmax(d["temp"]))

    # Mask shading goes in first so the data lines draw on top of it. One legend entry per
    # reason toggles both panels through the shared legendgroup.
    for name, m in d["reasons"].items():
        if not m.any():
            continue
        fill = reason_fills.get(name, "rgba(214,39,40,0.18)")
        label = f"Flagged ({name})"
        for row, (y0, y1) in ((1, (f_lo, f_hi)), (2, (t_lo, t_hi))):
            band = mask_band_trace(d["wavs"], m, y0, y1, fill, label, row == 1)
            if band is not None:
                fig.add_trace(band, row=row, col=1)

    # panel 1: flux, drawn unbroken so the profile stays comparable across a masked region
    fig.add_trace(go.Scattergl(x=d["wavs"], y=d["flux"], mode="lines",
                               line=dict(color="black", width=0.8),
                               name="Flux"), row=1, col=1)

    # panel 2: formation temperature
    fig.add_trace(go.Scattergl(x=d["wavs"], y=d["temp"], mode="lines",
                               line=dict(color="#1f77b4", width=0.8),
                               name="Formation temp"), row=2, col=1)

    # optional overlay: the stitched/spliced model, windowed to this chunk. One legend
    # entry ("Stitched model") toggles both panels via the shared legendgroup; hidden by
    # default (visible="legendonly") so it acts as a check box you enable when wanted.
    if stitched is not None:
        lo_w, hi_w = float(d["wavs"].min()), float(d["wavs"].max())
        m = (stitched["wavs"] >= lo_w) & (stitched["wavs"] <= hi_w)
        if m.any():
            sw = stitched["wavs"][m]
            fig.add_trace(go.Scattergl(x=sw, y=stitched["flux"][m], mode="lines",
                                       line=dict(color="#2ca02c", width=1.0, dash="dot"),
                                       name="Stitched model", legendgroup="stitched",
                                       visible="legendonly"), row=1, col=1)
            fig.add_trace(go.Scattergl(x=sw, y=stitched["temp"][m], mode="lines",
                                       line=dict(color="#2ca02c", width=1.0, dash="dot"),
                                       name="Stitched model", legendgroup="stitched",
                                       showlegend=False, visible="legendonly"), row=2, col=1)

    # panel 3: contribution-function heatmap, log-scaled like plot_chunk_flux.jl.
    # Divide the per-interval integral by Δlog τ_ref → contribution density dF/dlog τ_ref,
    # which is smooth in depth (removes the grid-driven horizontal step on the native grid).
    wavs_h, cfunc_h = block_max_decimate(d["wavs"], d["cfunc"], max_cols)
    cfunc_h = cfunc_h / d["dlogtau_ref"][:, None]
    zmax = float(cfunc_h.max())
    floor = zmax * 1e-4                       # clip 4 decades below peak
    z = np.log10(np.clip(cfunc_h, floor, None))
    lo_exp = int(np.floor(np.log10(floor)))
    hi_exp = int(np.ceil(np.log10(zmax)))
    tickvals = list(range(lo_exp, hi_exp + 1))
    fig.add_trace(go.Heatmap(x=wavs_h, y=d["t_mids"], z=z,
                             colorscale="Inferno",
                             colorbar=dict(title=dict(text="dF/dlog τ_ref<br>(log scale)"),
                                           len=0.5, y=0.0, yanchor="bottom",
                                           tickvals=tickvals,
                                           ticktext=[f"1e{t}" for t in tickvals]),
                             name="cfunc", hoverinfo="skip"),
                  row=3, col=1)

    # line-center markers span the flux panel's y-range
    fig.add_trace(line_marker_trace(d["line_centers"], f_lo, f_hi), row=1, col=1)

    # named curated regions, if this group has any. The shading says where the mask is; this
    # says which line put it there, so it is on by default.
    if d["line_mask"]:
        fig.add_trace(region_band_trace(d["line_mask"], f_hi), row=1, col=1)

    fig.update_yaxes(title_text="Normalized Flux", row=1, col=1)
    fig.update_yaxes(title_text="Formation Temp [K]", row=2, col=1)
    fig.update_yaxes(title_text="Temperature [K]", autorange="reversed", row=3, col=1)
    fig.update_xaxes(title_text="Wavelength [Å]", row=3, col=1)
    fig.update_layout(hovermode="x unified", template="plotly_white",
                      legend=dict(orientation="h", y=1.06),
                      margin=dict(t=40, r=90))
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor")
    return fig


def build_chunks_figure(chunks):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04)
    for c in chunks:
        fig.add_trace(go.Scattergl(x=c["wavs"], y=c["flux"], mode="lines",
                                   line=dict(width=0.8), name=c["name"],
                                   legendgroup=c["name"]), row=1, col=1)
        fig.add_trace(go.Scattergl(x=c["wavs"], y=c["temp"], mode="lines",
                                   line=dict(width=0.8), name=c["name"],
                                   legendgroup=c["name"], showlegend=False), row=2, col=1)
    fig.update_yaxes(title_text="Normalized Flux", row=1, col=1)
    fig.update_yaxes(title_text="Formation Temp [K]", row=2, col=1)
    fig.update_xaxes(title_text="Wavelength [Å]", row=2, col=1)
    fig.update_layout(hovermode="x unified", template="plotly_white",
                      margin=dict(t=40))
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor")
    return fig


# --- paged output --------------------------------------------------------


index_template = """<!doctype html>
<html><head><meta charset="utf-8"><title>Spectrum chunks</title>
<style>
  html,body{margin:0;height:100%;font-family:system-ui,sans-serif}
  #nav{display:flex;gap:.5rem;align-items:center;padding:.4rem .6rem;
       background:#222;color:#eee}
  #nav button,#nav select{font-size:14px;padding:.2rem .5rem}
  #pos{margin-left:auto;font-variant-numeric:tabular-nums}
  #view{border:0;width:100%;height:calc(100vh - 2.6rem);display:block}
</style></head><body>
<div id="nav">
  <button id="prev">&#8592; Prev</button>
  <button id="next">Next &#8594;</button>
  <label>Chunk: <select id="sel"></select></label>
  <span id="pos"></span>
</div>
<iframe id="view"></iframe>
<script>
const files = __FILES__;
let i = 0;
const frame = document.getElementById('view');
const sel = document.getElementById('sel');
const pos = document.getElementById('pos');
files.forEach((f, k) => { const o = document.createElement('option');
  o.value = k; o.textContent = f.label; sel.appendChild(o); });
function go(n){ i = Math.max(0, Math.min(files.length - 1, n));
  frame.src = files[i].file; sel.value = i;
  pos.textContent = (i + 1) + ' / ' + files.length; }
sel.onchange = e => go(+e.target.value);
document.getElementById('prev').onclick = () => go(i - 1);
document.getElementById('next').onclick = () => go(i + 1);
document.addEventListener('keydown', e => {
  if (e.key === 'ArrowLeft') go(i - 1);
  else if (e.key === 'ArrowRight') go(i + 1); });
go(0);
</script></body></html>
"""


def write_index(outdir, entries):
    # Static pager: dropdown + Prev/Next + arrow keys swap an <iframe> src.
    import json
    html = index_template.replace("__FILES__", json.dumps(entries))
    with open(os.path.join(outdir, "index.html"), "w") as fh:
        fh.write(html)


def write_paged(path, outdir, lo, hi, max_cols, stitched=None):
    # One HTML per chunk sharing a single plotly.min.js, plus an index pager.
    os.makedirs(outdir, exist_ok=True)
    entries = []
    with h5py.File(path, "r") as f:
        Ts = f["model_atmosphere"]["Ts"][:].ravel()
        tau_ref = read_tau_ref(f)
        mask_bits = read_mask_bits(f)
        names = sorted(n for n in f.keys() if n.startswith("chunk_"))
        assert names, f"no 'chunk_*' groups in {path}"
        for name in names:
            d = window_1d(read_group(f[name], Ts, tau_ref, mask_bits), lo, hi)
            if d is None:                      # chunk lies fully outside the range
                continue
            fname = name + ".html"
            fig = build_1d_figure(d, max_cols, stitched)
            # "directory" makes every page reference one shared plotly.min.js
            fig.write_html(os.path.join(outdir, fname),
                           include_plotlyjs="directory", full_html=True)
            lo_w, hi_w = float(d["wavs"].min()), float(d["wavs"].max())
            entries.append(dict(file=fname, label=f"{lo_w:.1f}–{hi_w:.1f} Å"))
    assert entries, f"no chunks with pixels in range [{lo}, {hi}]"
    write_index(outdir, entries)
    return os.path.join(outdir, "index.html")


# --- driver --------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", help="HDF5 file (spliced 1D by default)")
    p.add_argument("--paged", action="store_true",
                   help="per-chunk file: write one HTML page per chunk + an index.html pager")
    p.add_argument("--chunks", action="store_true",
                   help="per-chunk file: overplot all chunks in one figure, no heatmap")
    p.add_argument("--out", default=None,
                   help="output HTML file (default mode) or directory (--paged)")
    p.add_argument("--wav-lo", type=float, default=None, help="min wavelength [Angstrom]")
    p.add_argument("--wav-hi", type=float, default=None, help="max wavelength [Angstrom]")
    p.add_argument("--max-cols", type=int, default=3000,
                   help="heatmap column cap after block-max decimation")
    p.add_argument("--stitched", default=None,
                   help="path to the spliced/1D HDF5; overlays the stitched model on each "
                        "--paged chunk page as a legend-toggleable trace (off by default)")
    p.add_argument("--open", action="store_true", help="open the result in a browser")
    args = p.parse_args()

    assert os.path.isfile(args.input), f"input not found: {args.input}"

    if args.paged:
        outdir = args.out or os.path.splitext(args.input)[0] + "_pages"
        stitched = load_stitched(args.stitched) if args.stitched else None
        index = write_paged(args.input, outdir, args.wav_lo, args.wav_hi, args.max_cols, stitched)
        print(f"wrote {index}", flush=True)
        if args.open:
            webbrowser.open("file://" + os.path.abspath(index))
        return

    out = args.out or os.path.splitext(args.input)[0] + ".html"

    if args.chunks:
        chunks = load_chunks(args.input)
        if args.wav_lo is not None or args.wav_hi is not None:
            for c in chunks:
                keep = range_keep(c["wavs"], args.wav_lo, args.wav_hi)
                for key in ("wavs", "flux", "temp"):
                    c[key] = c[key][keep]
        fig = build_chunks_figure(chunks)
    else:
        d = window_1d(load_1d(args.input), args.wav_lo, args.wav_hi)
        assert d is not None, f"no pixels in range [{args.wav_lo}, {args.wav_hi}]"
        fig = build_1d_figure(d, args.max_cols)

    fig.write_html(out, include_plotlyjs=True, full_html=True)
    print(f"wrote {out}", flush=True)

    if args.open:
        webbrowser.open("file://" + os.path.abspath(out))


if __name__ == "__main__":
    sys.exit(main())
