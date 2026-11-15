"""Interactive, self-contained HTML report for spot checks.

Plan or layout segmentation as background, room polygons coloured by type,
OCR boxes coloured by confidence, tables of rooms and texts (with the image
crop next to each reading). Hover explains, click jumps to the spot.
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .schemas import PlanGraph

SEG_COLORS = np.array([(255, 255, 255), (170, 220, 170), (40, 40, 40), (250, 200, 120), (180, 210, 250),
                       (240, 170, 200), (140, 220, 230), (230, 230, 160), (120, 120, 120), (200, 180, 150),
                       (190, 190, 230), (220, 220, 220)], dtype=np.float32)


def layout_blend(img: Image.Image, room_prob: np.ndarray) -> Image.Image:
    """Segmentation colours of the layout model over the plan (original size)."""
    base = np.asarray(img.convert("RGB"), dtype=np.float32)
    return Image.fromarray((0.45 * base + 0.55 * SEG_COLORS[room_prob.argmax(0)]).astype(np.uint8))


def _b64(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def html_report(img: Image.Image, graph: PlanGraph, path: Path, thr: float | None = None,
                layout_img: Image.Image | None = None, answers: list[dict] | None = None) -> None:
    """answers: dicts with question, prediction, confidence, evidence, explanation[, answer, correct].
    thr: initial position of the text-confidence slider (default: the graph's text threshold)."""
    mc = graph.min_conf
    if thr is None:
        thr = mc.get("texts", 0.8) if isinstance(mc, dict) else (mc or 0.8)
    mc_txt = (f"Texte {mc.get('texts')}, Räume {mc.get('rooms')}, Symbole {mc.get('objects')}"
              if isinstance(mc, dict) else str(mc))
    data = {"img": _b64(img), "layout": _b64(layout_img) if layout_img is not None else None,
            "w": img.width, "h": img.height, "thr": thr, "source": graph.source,
            "rooms": [r.model_dump(mode="json") for r in graph.rooms],
            "texts": [t.model_dump(mode="json") for t in graph.texts],
            "objects": [o.model_dump(mode="json") for o in graph.objects],
            "min_conf": mc_txt, "dropped": graph.dropped,
            "answers": answers or []}
    path.write_text(_HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False)), encoding="utf-8")


_HTML = r"""<!DOCTYPE html><html lang="de"><head><meta charset="utf-8"><title>Plan-Check</title>
<style>
body{font-family:system-ui,sans-serif;margin:0;color:#222}
header{padding:8px 14px;border-bottom:1px solid #ddd;display:flex;gap:16px;align-items:center;flex-wrap:wrap;font-size:14px}
main{display:grid;grid-template-columns:minmax(0,3fr) minmax(360px,2fr);height:calc(100vh - 52px)}
#view{overflow:auto;background:#f4f4f4}#stage{position:relative}#stage img{display:block}
svg{position:absolute;left:0;top:0}
.t{fill-opacity:.15;stroke-width:2;cursor:pointer;vector-effect:non-scaling-stroke}
.room{fill-opacity:.18;stroke-width:2;cursor:pointer;vector-effect:non-scaling-stroke}
.sel{fill-opacity:.4;stroke-width:4}
.ok{color:#1e963c}.wrong{color:#d72828}
.obj{fill:none;stroke-width:2;stroke-dasharray:4 2;cursor:pointer;vector-effect:non-scaling-stroke}
.obj.sel{stroke-width:4;stroke-dasharray:none}
.rl{font-size:12px;paint-order:stroke;stroke:#fff;stroke-width:3px;pointer-events:none}
#side{overflow:auto;border-left:1px solid #ddd}h3{font-size:14px;margin:10px 8px 4px}
table{border-collapse:collapse;width:100%;font-size:12.5px}
td,th{padding:3px 6px;border-bottom:1px solid #eee;text-align:left;vertical-align:middle}
tr{cursor:pointer}tr.sel{background:#fff3c4}
.sw{display:inline-block;width:11px;height:11px;border:1px solid #888;vertical-align:-1px}
.crop{width:120px;height:28px;background-repeat:no-repeat;border:1px solid #ddd}
#legend{display:flex;gap:10px;flex-wrap:wrap;padding:2px 8px;font-size:12px}
#tip{position:fixed;pointer-events:none;background:#fff;border:1px solid #999;padding:6px 8px;font-size:12px;display:none;max-width:340px;z-index:9}
small{color:#666}.note{padding:8px;color:#a33;font-size:13px}
</style></head><body>
<header><b id="title"></b>
<label>Hintergrund <select id="bg"><option value="plan">Plan</option><option value="layout">Layout-Segmentierung</option></select></label>
<label><input id="rooms" type="checkbox" checked> Räume</label>
<label><input id="texts" type="checkbox" checked> Texte</label>
<label><input id="objs" type="checkbox" checked> Symbole</label>
<label>Konfidenz &lt; <input id="thr" type="range" min="0" max="1" step="0.05"> <span id="thrv"></span></label>
<label><input id="doubt" type="checkbox"> nur unsichere Texte</label>
<label>Zoom <input id="zoom" type="range" min="0.5" max="4" step="0.25" value="1"></label>
<span id="stats"></span></header>
<main><div id="view"><div id="stage"><img id="img"><svg id="svg"></svg></div></div>
<div id="side">
 <div id="answers"></div>
 <h3>Räume (Layout-Modell)</h3><div id="legend"></div><div id="note"></div>
 <table><thead><tr><th></th><th>ID</th><th>Room type</th><th>Model class</th><th>OCR label</th><th>Fläche</th><th>Konf.</th><th>Ausstattung</th><th>Probleme</th></tr></thead><tbody id="roomrows"></tbody></table>
 <h3>Texte (OCR)</h3>
 <table><thead><tr><th></th><th>Ausschnitt</th><th>Text</th><th>Art</th><th>Konf.</th><th>Raum</th></tr></thead><tbody id="rows"></tbody></table>
</div></main><div id="tip"></div>
<script>
const D=__DATA__, NS='http://www.w3.org/2000/svg', $=id=>document.getElementById(id);
const C={ok:'#1e963c',low:'#eb8c00',bad:'#d72828'};
const RT={bathroom:'#1aa3b8',sauna:'#c2185b',kitchen:'#e69500',living_room:'#3b7ddd',bedroom:'#d64f8a',hallway:'#8d9a2a',
 storage:'#8d6e63',garage:'#5d4037',balcony:'#43a047',other:'#9e9e9e'};
const EN={bathroom:'Bathroom',sauna:'Sauna',kitchen:'Kitchen',living_room:'Living room',bedroom:'Bedroom',hallway:'Hallway',
 storage:'Storage',garage:'Garage',balcony:'Balcony / outdoor',other:'Other room'};
const rooms=Object.fromEntries(D.rooms.map(r=>[r.id,r])), all={...rooms, ...Object.fromEntries(D.texts.map(t=>[t.id,t])), ...Object.fromEntries(D.objects.map(o=>[o.id,o]))};
const OC={door:'#e65100',window:'#1565c0',toilet:'#6a1b9a',sink:'#00838f',bathtub:'#2e7d32',closet:'#795548',sauna_bench:'#c2185b',fireplace:'#bf360c'};
$('title').textContent=D.source; $('thr').value=D.thr; $('svg').setAttribute('viewBox',`0 0 ${D.w} ${D.h}`);
if(!D.layout)$('bg').disabled=true;
const esc=s=>String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const st=(t,thr)=>t.conf>=thr?'ok':(t.conf>=.5?'low':'bad');
const el=(tag,a)=>{const e=document.createElementNS(NS,tag);for(const k in a)e.setAttribute(k,a[k]);return e};
function tip(ev,h){const t=$('tip');t.innerHTML=h;t.style.display='block';t.style.left=ev.clientX+14+'px';t.style.top=ev.clientY+14+'px'}
const untip=()=>$('tip').style.display='none';
const roomTip=r=>`<b>${r.id}</b> ${EN[r.room_type]} <small>(model class: ${esc(r.model_class)})</small>`+
 `${r.label_text?'<br>OCR label „'+esc(r.label_text)+'“':''}<br>${r.area_m2!=null?'Fläche laut Plan: <b>'+r.area_m2+' m²</b>':'keine Flächenangabe gefunden'}`+
 `<br><small>${esc(r.type_reason)}</small>${r.issues.length?'<br><small>Probleme: '+r.issues.join(', ')+'</small>':''}`;
const textTip=t=>{const r=rooms[t.room_id];return `<b>${t.id}</b> „${esc(t.text)}“ (${t.conf.toFixed(2)})<br>${t.kind}${t.kind==='area'?' = '+t.value+' m²':''}`+
 `<br>${r?'Raum '+r.id+' '+EN[r.room_type]+(r.label_text?' ('+esc(r.label_text)+')':''):'kein Raum'}${t.flags.length?'<br><small>'+t.flags.join(', ')+'</small>':''}`};
function hover(e,html,id){e.onmousemove=ev=>tip(ev,html);e.onmouseleave=untip;e.onclick=()=>select(id)}
function crop(b){const pad=3,bw=b[2]-b[0]+2*pad,bh=b[3]-b[1]+2*pad,s=Math.min(120/bw,28/bh);
 return `background-image:url(${D.img});background-size:${D.w*s}px ${D.h*s}px;background-position:${-(b[0]-pad)*s}px ${-(b[1]-pad)*s}px`}
let sel=null;
function select(ids){ids=[].concat(ids);sel=ids;document.querySelectorAll('.sel').forEach(e=>e.classList.remove('sel'));
 ids.forEach(id=>document.querySelectorAll(`[data-id="${id}"]`).forEach(e=>e.classList.add('sel')));
 const bs=ids.map(id=>all[id]).filter(Boolean).map(x=>x.box_px); if(!bs.length)return;
 const b=[Math.min(...bs.map(b=>b[0])),Math.min(...bs.map(b=>b[1])),Math.max(...bs.map(b=>b[2])),Math.max(...bs.map(b=>b[3]))];
 const z=+$('zoom').value,v=$('view');
 v.scrollTo({left:(b[0]+b[2])/2*z-v.clientWidth/2,top:(b[1]+b[3])/2*z-v.clientHeight/2,behavior:'smooth'})}
function renderAnswers(){if(!D.answers.length){$('answers').innerHTML='';return}
 let h='<h3>Antworten</h3><table><thead><tr><th>Frage</th><th>Antwort</th><th>Konf.</th><th>Begründung</th></tr></thead><tbody>';
 D.answers.forEach((a,i)=>{const truth=a.answer!==undefined?` <small class="${a.correct?'ok':'wrong'}">(${a.correct?'✓':'✗ richtig: '+a.answer})</small>`:'';
  h+=`<tr data-ans="${i}"><td>${esc(a.question)}</td><td><b>${a.prediction??'–'}</b>${truth}</td><td>${(a.confidence??0).toFixed(2)}${a.fallback?'<br><small>Fallback</small>':''}</td><td><small>${esc(a.explanation)}</small></td></tr>`});
 $('answers').innerHTML=h+'</tbody></table>';
 document.querySelectorAll('[data-ans]').forEach(tr=>tr.onclick=()=>{document.querySelectorAll('tr.sel').forEach(e=>e.classList.remove('sel'));select(D.answers[+tr.dataset.ans].evidence);tr.classList.add('sel')})}
const objTip=o=>`<b>${o.id}</b> Symbol: ${o.kind}<br>Wahrscheinlichkeit ${o.conf.toFixed(2)}${o.room_id?'<br>in Raum '+o.room_id:''}`;
function render(){
 const thr=+$('thr').value,svg=$('svg'); $('thrv').textContent=thr.toFixed(2);
 $('img').src=$('bg').value==='layout'&&D.layout?D.layout:D.img; svg.innerHTML=''; $('rows').innerHTML=''; $('roomrows').innerHTML='';
 $('legend').innerHTML=[...new Set(D.rooms.map(r=>r.room_type))].map(t=>`<span><span class="sw" style="background:${RT[t]}"></span> ${EN[t]}</span>`).join('');
 $('note').innerHTML=D.rooms.length?'':'<div class="note">Keine Räume: Layout-Modell nicht gelaufen oder nichts erkannt.</div>';
 for(const r of D.rooms){const c=RT[r.room_type],b=r.box_px;
  if($('rooms').checked){const e=el('polygon',{points:r.polygon_px.map(p=>p.join(',')).join(' '),class:'room'});
   e.dataset.id=r.id;e.style.fill=c;e.style.stroke=c;hover(e,roomTip(r),r.id);svg.appendChild(e);
   const l=el('text',{x:b[0]+3,y:b[1]+13,class:'rl',fill:c});l.textContent=`${r.id} ${EN[r.room_type]}${r.label_text?' ('+r.label_text+')':''}`;svg.appendChild(l)}
  const tr=document.createElement('tr');tr.dataset.id=r.id;hover(tr,roomTip(r),r.id);
  tr.innerHTML=`<td><span class="sw" style="background:${c}"></span></td><td>${r.id}</td><td>${EN[r.room_type]}</td><td><small>${esc(r.model_class)}</small></td>`+
   `<td>${esc(r.label_text||'–')}</td><td>${r.area_m2!=null?r.area_m2+' m²':'–'}</td><td>${r.type_conf.toFixed(2)}</td><td>${r.fixtures.join(', ')||'–'}</td><td><small>${r.issues.join(', ')}</small></td>`;
  $('roomrows').appendChild(tr)}
 if($('objs').checked)for(const o of D.objects){const b=o.box_px,c=OC[o.kind]||'#555';
  const e=el('rect',{x:b[0]-1,y:b[1]-1,width:b[2]-b[0]+2,height:b[3]-b[1]+2,class:'obj'});e.dataset.id=o.id;e.style.stroke=c;hover(e,objTip(o),o.id);svg.appendChild(e)}
 const order={bad:0,low:1,ok:2},n={ok:0,low:0,bad:0};
 for(const t of [...D.texts].sort((a,b)=>order[st(a,thr)]-order[st(b,thr)]||a.conf-b.conf)){const s=st(t,thr),b=t.box_px;n[s]++;
  if($('doubt').checked&&s==='ok')continue;
  if($('texts').checked){const e=el('rect',{x:b[0],y:b[1],width:Math.max(1,b[2]-b[0]),height:Math.max(1,b[3]-b[1]),class:'t'});
   e.dataset.id=t.id;e.style.stroke=C[s];e.style.fill=C[s];hover(e,textTip(t),t.id);svg.appendChild(e)}
  const tr=document.createElement('tr'),r=rooms[t.room_id];tr.dataset.id=t.id;hover(tr,textTip(t),t.id);
  tr.innerHTML=`<td><span class="sw" style="background:${C[s]};border-radius:50%"></span></td><td><div class="crop" style="${crop(b)}"></div></td>`+
   `<td>${esc(t.text)}</td><td>${t.kind}${t.kind==='area'?'<br><small>'+t.value+' m²</small>':''}</td><td>${t.conf.toFixed(2)}</td><td>${r?r.id+' '+esc(r.label_text||''):'–'}</td>`;
  $('rows').appendChild(tr)}
 $('stats').innerHTML=`<small>${D.rooms.length} Räume · ${D.objects.length} Symbole · ${D.texts.length} Texte`+
  (D.dropped&&Object.keys(D.dropped).length?` · verworfen (unter Schwellwert: ${D.min_conf}): ${D.dropped.texts} Texte, ${D.dropped.rooms} Räume, ${D.dropped.objects} Symbole`:'')+`</small>`;
 if(sel)select(sel)}
$('zoom').oninput=()=>{const z=+$('zoom').value;$('img').style.width=D.w*z+'px';$('svg').setAttribute('width',D.w*z);$('svg').setAttribute('height',D.h*z)};
['thr','doubt','rooms','texts','objs','bg'].forEach(id=>$(id).oninput=render);
$('img').onload=()=>{if(window._init)return;window._init=1;const v=$('view').clientWidth;
 $('zoom').value=D.w>v?Math.max(.5,v/D.w):Math.min(4,Math.floor(v/D.w));$('zoom').oninput();render();renderAnswers()};
$('img').src=D.img;
</script></body></html>"""
