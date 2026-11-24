"""Compare held-out real and single-draw whole-flight generator traces."""

from __future__ import annotations

import json
from pathlib import Path
from string import Template

import matplotlib.pyplot as plt
import numpy as np
from plotly.offline import get_plotlyjs

from plots.timeframe.renderers import ROTOR_COLORS

_PAGE = Template("""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Whole-flight response comparison</title>
<style>body{font:16px/1.45 system-ui,sans-serif;color:#263643;max-width:1400px;margin:1rem auto;padding:0 1rem}h1{font-size:1.45rem;margin:.2rem 0}p{margin:.4rem 0}label{display:inline-block;margin:.35rem 1rem .35rem 0}select,input{font:inherit}#chart{width:100%}</style>
<h1>Measured vs generated motor speeds</h1><p>Draw 0, on the recording clock. The grey panel label identifies the measurement; candidate order is shuffled for each flight. All panels use the same speed scale. Cruise shows its middle scored window.</p>
<label>Flight <select id="flight"></select></label><label>Phase <select id="phase"></select></label><label><input type="checkbox" id="reveal"> Reveal methods</label><div id="chart"></div>
<script>$plotly</script><script type="application/json" id="data">$payload</script><script>
const d=JSON.parse(document.getElementById('data').textContent), field=id=>document.getElementById(id);
const flight=field('flight'), phase=field('phase'), reveal=field('reveal');
for(const [i,f] of d.flights.entries())flight.add(new Option(f.label,i));
for(const p of Object.keys(d.flights[0].phases))phase.add(new Option(p,p));
function show(){ const f=d.flights[+flight.value], cell=f.phases[phase.value], keys=['real',...f.order];
const traces=[],layout={height:Math.ceil(keys.length/2)*220+50,margin:{t:70,l:68,b:55,r:18},grid:{rows:Math.ceil(keys.length/2),columns:2,pattern:'independent',ygap:.27,xgap:.12},title:{text:f.label+' · '+phase.value+' · '+cell.start.toFixed(2)+' s from recording start',font:{size:16}},annotations:[],legend:{orientation:'h',y:-.07},hovermode:'closest',template:'plotly_white'};
let lo=Infinity,hi=-Infinity; for(const source of keys) for(const row of cell.paths[source])for(const v of row){lo=Math.min(lo,v);hi=Math.max(hi,v);}const pad=Math.max(.4,(hi-lo)*.05);
keys.forEach((source,i)=>{const s=i===0?'':String(i+1), x='x'+s,y='y'+s;
layout['xaxis'+s]={title:i>=keys.length-2?'Time in window (s)':'',matches:i?'x':undefined};layout['yaxis'+s]={range:[lo-pad,hi+pad],title:i%2===0?'rev/s':'',matches:i?'y':undefined};
layout.annotations.push({text:i===0?'Measured':(reveal.checked?d.names[source]:'Candidate '+String.fromCharCode(64+i)),showarrow:false,xref:x+' domain',yref:y+' domain',x:0,y:1,xanchor:'left',yanchor:'bottom',font:{size:13}});
cell.paths[source].forEach((row,k)=>traces.push({type:'scatter',mode:'lines',x0:0,dx:1/d.fs,y:row,xaxis:x,yaxis:y,name:d.rotors[k],legendgroup:String(k),showlegend:i===0,line:{width:1.25,color:d.colors[k]},hovertemplate:'%{x:.2f} s  ·  %{y:.2f} rev/s<extra>'+d.rotors[k]+'</extra>'}));});
Plotly.react('chart',traces,layout,{responsive:true,scrollZoom:true,displaylogo:false}); }
for(const control of [flight,phase,reveal])control.addEventListener('change',show); show();
</script></html>""")


def render(
    cells, names: dict[str, str], out: Path, fs: float, *, primary: str = "unified_hierarchical"
) -> None:
    rng = np.random.default_rng(2026)
    grouped = {}
    for rig, flight, phase, start, paths in cells:
        if (rig, flight) not in grouped:
            grouped[rig, flight] = {
                "label": f"{rig} / {flight}",
                "order": rng.permutation(list(names)).tolist(),
                "phases": {},
            }
        window = {
            "start": start,
            "paths": {s: np.round(x.astype(float), 4).tolist() for s, x in paths.items()},
        }
        grouped[rig, flight]["phases"][phase] = window
    payload = {
        "fs": fs,
        "names": names,
        "colors": ROTOR_COLORS,
        "rotors": ["Front right", "Front left", "Rear left", "Rear right"],
        "flights": list(grouped.values()),
    }
    (out / "flight_comparison.html").write_text(
        _PAGE.substitute(
            plotly=get_plotlyjs(),
            payload=json.dumps(payload, separators=(",", ":"), allow_nan=False).replace(
                "<", r"\u003c"
            ),
        )
    )
    # The campaign's predefined illustrative held-out recording; no looking for a winning draw.
    ex = grouped["michaels", "FLY124"]["phases"]
    methods = ("real", primary, "ours", "phasevar2+hsmm3")
    fig, axs = plt.subplots(len(ex), len(methods), figsize=(12, 8.5), sharey="row")
    fig.subplots_adjust(left=0.082, right=0.991, top=0.88, bottom=0.065, hspace=0.64, wspace=0.09)
    for i, (phase, cell) in enumerate(ex.items()):
        for j, arm in enumerate(methods):
            ax = axs[i, j]
            traces = np.asarray(cell["paths"][arm])
            t = np.arange(traces.shape[1]) / fs
            for trace, color, rotor in zip(traces, ROTOR_COLORS, payload["rotors"]):
                ax.plot(t, trace, c=color, lw=0.85, label=rotor)
            ax.set_xlim(0, t[-1])
            ax.spines[["right", "top"]].set_visible(False)
            ax.tick_params(labelsize=8)
            ax.grid(axis="y", alpha=0.18)
            if j == 0:
                ax.set_ylabel(phase.capitalize() + "\nrev/s", fontsize=9)
            if i == 0:
                ax.set_title(names.get(arm, "Measured"), fontsize=10)
            if i == len(ex) - 1:
                ax.set_xlabel("Window time (s)", fontsize=9)
    fig.suptitle("MD2 / FLY124  ·  held-out schedule, first draw", fontsize=15, weight="bold")
    fig.legend(
        *axs[0, 0].get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.956),
        ncol=4,
        frameon=False,
    )
    fig.savefig(out / "example_FLY124.png", dpi=175)
    plt.close(fig)
