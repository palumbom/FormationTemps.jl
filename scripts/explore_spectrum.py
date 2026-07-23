"""Interactive browser viewer for FormationTemps flux / formation-temperature spectra."""

import argparse, os, sys, webbrowser

import h5py
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# --- data loading --------------------------------------------------------


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


def read_group(g, Ts):
    # Read one chunk group into the dict shape build_1d_figure expects.
    wavs = g["wavs"][:].ravel()
    flux = g["flux"][:].ravel()
    temp = g["temp"][:].ravel()
    mask = g["mask"][:].ravel().astype(bool)
    line_centers = g["line_centers"][:].ravel()
    n_wav = wavs.size
    n_rows = Ts.size - 1                       # cfunc has Natm-1 layers
    cfunc = orient_cfunc(g["cfunc"][:], n_wav, n_rows)
    # mid-layer temperatures, matching plot_chunk_flux.jl
    t_mids = 0.5 * (Ts[:n_rows] + Ts[1:n_rows + 1])
    return dict(wavs=wavs, flux=flux, temp=temp, mask=mask,
                line_centers=line_centers, cfunc=cfunc, t_mids=t_mids)


def load_1d(path):
    # Read the spliced single-spectrum group and everything the viewer needs.
    with h5py.File(path, "r") as f:
        assert "chunk_0001" in f, f"no 'chunk_0001' group in {path} (use --chunks?)"
        Ts = f["model_atmosphere"]["Ts"][:].ravel()
        return read_group(f["chunk_0001"], Ts)


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
    out["cfunc"] = d["cfunc"][:, keep]
    lc = d["line_centers"]
    out["line_centers"] = lc[range_keep(lc, lo, hi)]
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


def split_mask(y, mask):
    # Split a series into (trusted, flagged), NaN elsewhere, so each draws as its
    # own gapped line. Flagged = boundary-contaminated pixels.
    return np.where(~mask, y, np.nan), np.where(mask, y, np.nan)


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


def build_1d_figure(d, max_cols):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, row_heights=[0.25, 0.25, 0.5],
                        subplot_titles=("", "", ""))

    mask = d["mask"]

    # panel 1: flux, split trusted vs boundary-flagged
    f_trust, f_flag = split_mask(d["flux"], mask)
    fig.add_trace(go.Scattergl(x=d["wavs"], y=f_trust, mode="lines", connectgaps=False,
                               line=dict(color="black", width=0.8),
                               name="Flux"), row=1, col=1)
    fig.add_trace(go.Scattergl(x=d["wavs"], y=f_flag, mode="lines", connectgaps=False,
                               line=dict(color="#d62728", width=0.8),
                               name="Flagged (boundary)", legendgroup="flagged"), row=1, col=1)

    # panel 2: formation temperature, same split; shares the flagged legend toggle
    t_trust, t_flag = split_mask(d["temp"], mask)
    fig.add_trace(go.Scattergl(x=d["wavs"], y=t_trust, mode="lines", connectgaps=False,
                               line=dict(color="#1f77b4", width=0.8),
                               name="Formation temp"), row=2, col=1)
    fig.add_trace(go.Scattergl(x=d["wavs"], y=t_flag, mode="lines", connectgaps=False,
                               line=dict(color="#d62728", width=0.8),
                               name="Flagged (boundary)", legendgroup="flagged",
                               showlegend=False), row=2, col=1)

    # panel 3: contribution-function heatmap, log-scaled like plot_chunk_flux.jl
    wavs_h, cfunc_h = block_max_decimate(d["wavs"], d["cfunc"], max_cols)
    zmax = float(cfunc_h.max())
    floor = zmax * 1e-4                       # clip 4 decades below peak
    z = np.log10(np.clip(cfunc_h, floor, None))
    lo_exp = int(np.floor(np.log10(floor)))
    hi_exp = int(np.ceil(np.log10(zmax)))
    tickvals = list(range(lo_exp, hi_exp + 1))
    fig.add_trace(go.Heatmap(x=wavs_h, y=d["t_mids"], z=z,
                             colorscale="Inferno",
                             colorbar=dict(title=dict(text="Cont. Fn.<br>(log scale)"),
                                           len=0.5, y=0.0, yanchor="bottom",
                                           tickvals=tickvals,
                                           ticktext=[f"1e{t}" for t in tickvals]),
                             name="cfunc", hoverinfo="skip"),
                  row=3, col=1)

    # line-center markers span the flux panel's y-range
    y_lo, y_hi = float(np.nanmin(d["flux"])), float(np.nanmax(d["flux"]))
    fig.add_trace(line_marker_trace(d["line_centers"], y_lo, y_hi), row=1, col=1)

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


def write_paged(path, outdir, lo, hi, max_cols):
    # One HTML per chunk sharing a single plotly.min.js, plus an index pager.
    os.makedirs(outdir, exist_ok=True)
    entries = []
    with h5py.File(path, "r") as f:
        Ts = f["model_atmosphere"]["Ts"][:].ravel()
        names = sorted(n for n in f.keys() if n.startswith("chunk_"))
        assert names, f"no 'chunk_*' groups in {path}"
        for name in names:
            d = window_1d(read_group(f[name], Ts), lo, hi)
            if d is None:                      # chunk lies fully outside the range
                continue
            fname = name + ".html"
            fig = build_1d_figure(d, max_cols)
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
    p.add_argument("--open", action="store_true", help="open the result in a browser")
    args = p.parse_args()

    assert os.path.isfile(args.input), f"input not found: {args.input}"

    if args.paged:
        outdir = args.out or os.path.splitext(args.input)[0] + "_pages"
        index = write_paged(args.input, outdir, args.wav_lo, args.wav_hi, args.max_cols)
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
