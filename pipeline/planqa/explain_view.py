"""Self-contained explain.html for an explanation.json: three views on the drawing.

  Belege       answer card (value, confidence, why, sources) + evidence on the plan (counted / excluded /
               below threshold, numbered) + how the answer came about (steps, click -> ids light up)
  Unsicherheit entropy heat map of the layout model / OCR per channel, ranked hotspots with crops,
               top classes and how each was resolved
  Agent        what the agent looked at, replayable step by step (scrubber): seen elements, viewed
               crops, elements it could not see (below threshold), coverage per element kind
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

from PIL import Image


def _b64(img: Image.Image, fmt: str, max_side: int = 2000, quality: int = 85) -> str:
    s = min(1.0, max_side / max(img.size))
    if s < 1:
        img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
    buf = io.BytesIO()
    if fmt == "JPEG":
        img.convert("RGB").save(buf, "JPEG", quality=quality)
    else:
        img.save(buf, fmt, optimize=True)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


def explain_html(doc, image: Image.Image, heat: dict, path) -> None:
    data = json.loads(doc.model_dump_json(exclude_none=True))
    payload = json.dumps({"doc": data, "img": _b64(image, "JPEG"),
                          "heat": {c: _b64(im, "PNG", 1600) for c, im in heat.items()}},
                         ensure_ascii=False).replace("</", "<\\/")
    Path(path).write_text(TEMPLATE.replace("/*PAYLOAD*/", payload), encoding="utf-8")


TEMPLATE = r"""<!DOCTYPE html><html lang="de"><head><meta charset="utf-8"><title>PlanQA – Erklärungen</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:Inter,"Segoe UI",system-ui,sans-serif;background:#eef1f5;color:#1f2937;font-size:14px;height:100vh;display:flex;flex-direction:column}
.top{height:60px;background:#fff;border-bottom:1px solid #dde2ea;display:flex;align-items:center;padding:0 20px;gap:16px;flex:none}
.logo{font-weight:700;font-size:16px;white-space:nowrap}.logo span{color:#2563eb}
select{flex:1;height:38px;border:1.5px solid #2563eb;border-radius:10px;padding:0 12px;font-size:14px;background:#f8faff;font-family:inherit}
.file{font-size:12.5px;color:#6b7280;background:#f3f4f6;padding:6px 10px;border-radius:8px;white-space:nowrap}
.main{flex:1;display:flex;gap:16px;padding:16px 20px;min-height:0}
.left{flex:1;background:#fff;border-radius:14px;border:1px solid #dde2ea;display:flex;flex-direction:column;min-width:0}
.tabs{display:flex;gap:8px;padding:10px 14px;border-bottom:1px solid #eef0f4;align-items:center;flex-wrap:wrap}
.tab{padding:6px 12px;border-radius:7px;font-size:13px;color:#4b5563;background:#f3f4f6;cursor:pointer;white-space:nowrap;user-select:none}
.tab.on{background:#1f2937;color:#fff}
.chip{padding:4px 9px;border-radius:6px;font-size:12px;background:#eef2ff;color:#3730a3;cursor:pointer;white-space:nowrap}
.chip.on{background:#3730a3;color:#fff}
.legend{margin-left:auto;display:flex;gap:12px;font-size:12px;color:#374151;align-items:center;flex-wrap:wrap}
.dot{display:inline-block;width:12px;height:12px;border-radius:50%;vertical-align:-1px;margin-right:4px}
.stage{flex:1;position:relative;overflow:hidden;min-height:0}
.stage svg{width:100%;height:100%;display:block}
.bottom{border-top:1px solid #eef0f4;padding:10px 16px;display:flex;gap:16px;align-items:center;font-size:12.5px;color:#4b5563;flex-wrap:wrap}
.right{width:540px;display:flex;flex-direction:column;gap:12px;overflow-y:auto;flex:none;padding-right:2px}
.card{background:#fff;border-radius:14px;border:1px solid #dde2ea;padding:13px 16px}
.h{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:#6b7280;font-weight:600;margin-bottom:9px;display:flex;gap:8px;align-items:center}
.num{background:#e0e7ff;color:#3730a3;border-radius:50%;width:18px;height:18px;display:inline-flex;align-items:center;justify-content:center;font-size:11px}
.ans{display:flex;align-items:flex-end;gap:14px}
.big{font-size:52px;font-weight:700;line-height:.9}.big small{font-size:20px;font-weight:600;color:#374151;margin-left:6px}
.badge{display:inline-flex;align-items:center;gap:6px;font-weight:600;padding:5px 10px;border-radius:999px;font-size:12.5px}
.badge i{width:9px;height:9px;border-radius:50%;display:inline-block}
.why{margin-top:10px;font-size:14px;line-height:1.5}
.src{display:flex;gap:8px;margin-top:11px}
.s{flex:1;border:1px solid #e5e7eb;border-radius:10px;padding:7px 10px;font-size:11.5px;color:#6b7280}
.s b{display:block;font-size:19px;color:#111827;margin-top:1px}
.ok{border-color:#86efac;background:#f0fdf4}.ok b{color:#15803d}
.bad{border-color:#fecaca;background:#fef2f2}.bad b{color:#b91c1c}
.warnc{border-color:#fde68a;background:#fffbeb}.warnc b{color:#b45309}
.step{display:flex;gap:9px;padding:6px 7px;border-radius:9px;font-size:12.8px;line-height:1.38;cursor:pointer;border:1.5px solid transparent}
.step:hover{background:#f8fafc}.step.on{background:#fff7ed;border-color:#fdba74}.step.todo{opacity:.38}
.ic{width:22px;height:22px;border-radius:6px;flex:none;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;color:#fff}
code{font-family:"DejaVu Sans Mono",Consolas,monospace;font-size:11px;background:#f3f4f6;padding:1px 4px;border-radius:4px;color:#374151;word-break:break-all}
.idc{display:inline-block;font-family:"DejaVu Sans Mono",Consolas,monospace;font-size:10.5px;background:#eff6ff;color:#1d4ed8;border:1px solid #bfdbfe;border-radius:4px;padding:0 4px;margin:2px 3px 0 0}
.warn{background:#fff7ed;border:1px solid #fed7aa;border-radius:10px;padding:8px 11px;font-size:12.8px;color:#7c2d12;line-height:1.4;margin-top:9px}
.cov{display:grid;grid-template-columns:110px 1fr 54px;gap:6px 10px;align-items:center;font-size:12.8px}
.pbar{height:8px;border-radius:4px;background:#f1f5f9;overflow:hidden}.pbar i{display:block;height:8px;border-radius:4px}
.hs{display:flex;gap:10px;align-items:flex-start;padding:8px 0;border-top:1px solid #f1f3f6;cursor:pointer}
.hs:first-of-type{border-top:none}.hs:hover{background:#fafafa}
.rank{width:22px;height:22px;border-radius:50%;background:#1f2937;color:#fff;font-size:11px;font-weight:700;display:flex;align-items:center;justify-content:center;flex:none}
canvas.thumb{width:72px;height:72px;border-radius:8px;border:1px solid #e5e7eb;flex:none}
.pop{position:absolute;background:#fff;border:1.5px solid #2563eb;border-radius:12px;box-shadow:0 8px 24px rgba(37,99,235,.18);padding:11px 13px;font-size:12.8px;line-height:1.45;width:270px;display:none;z-index:5}
.muted{color:#6b7280}
input[type=range]{accent-color:#f97316}
</style></head><body>
<div class="top"><div class="logo">Plan<span>QA</span> · Erklärungen</div><select id="q"></select><div class="file" id="file"></div></div>
<div class="main">
 <div class="left">
  <div class="tabs" id="tabs"></div>
  <div class="stage" id="stage"><svg id="svg"></svg><div class="pop" id="pop"></div></div>
  <div class="bottom" id="bottom"></div>
 </div>
 <div class="right" id="right"></div>
</div>
<script>
const P=/*PAYLOAD*/;
const D=P.doc, E=D.elements, W=D.source.image_size[0], H=D.source.image_size[1];
const R=Math.max(W,H)/85, NS="http://www.w3.org/2000/svg";
let tab="belege", qi=0, chan="combined", hopac=0.85, k=null, hiIds=[];
const $=id=>document.getElementById(id);
const esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const fmt=v=>v==null?"–":(Math.round(v*100)/100).toLocaleString("de-DE");
const KDE={door:"Türen",window:"Fenster",toilet:"WC-Becken",sink:"Waschbecken",bathtub:"Badewannen",sauna_bench:"Saunabänke",closet:"Schränke",room:"Räume",text:"Texte",sauna:"Saunen"};
$("file").textContent=D.source.file+" · Schwellen T "+D.run.thresholds.texts+" / S "+D.run.thresholds.objects+" / R "+D.run.thresholds.rooms;
D.answers.forEach((a,i)=>{const o=document.createElement("option");o.value=i;o.textContent=a.question;$("q").appendChild(o)});
$("q").onchange=e=>{qi=+e.target.value;k=null;hiIds=[];render()};
const A=()=>D.answers[qi];
function center(e){const b=e.geometry.bbox;return b?[(b[0]+b[2])/2,(b[1]+b[3])/2]:[0,0]}
function el(tag,attrs,parent){const n=document.createElementNS(NS,tag);for(const[a,v]of Object.entries(attrs||{}))n.setAttribute(a,v);if(parent)parent.appendChild(n);return n}
function shape(e,attrs,g){ // room polygon or element box
  if(e.geometry.polygon&&e.geometry.polygon.length>2) return el("polygon",{points:e.geometry.polygon.map(p=>p.join(",")).join(" "),...attrs},g);
  const b=e.geometry.bbox; return el("rect",{x:b[0],y:b[1],width:Math.max(2,b[2]-b[0]),height:Math.max(2,b[3]-b[1]),...attrs},g);}
function marker(e,fill,stroke,label,g,dash){const[x,y]=center(e);const c=el("g",{},g);
  el("circle",{cx:x,cy:y,r:R,fill,stroke:stroke||"none","stroke-width":R*.22,"stroke-dasharray":dash||""},c);
  if(label!=null){const t=el("text",{x,y:y+R*.36,"text-anchor":"middle","font-size":R*.95,"font-weight":700,fill:fill==="#fff"||fill==="#fffbeb"?stroke:"#fff","font-family":"Inter,sans-serif"},c);t.textContent=label}
  c.style.cursor="pointer";c.onclick=ev=>{ev.stopPropagation();popup(e.id,ev)};return c}
function base(svg){svg.innerHTML="";svg.setAttribute("viewBox",`0 0 ${W} ${H}`);
  el("image",{href:P.img,x:0,y:0,width:W,height:H,preserveAspectRatio:"none",opacity:tab==="agent"?.5:1},svg);
  svg.onclick=()=>{$("pop").style.display="none"};return el("g",{},svg)}
function popup(id,ev){const e=E[id];if(!e)return;const a=A();const evd=(a.evidence||[]).find(x=>x.element===id);
  const st={kept:"im Graphen",below_threshold:"unter Schwelle – verworfen",confirmed:"bestätigt",rejected:"verworfen"}[e.status];
  let h=`<b style="font-size:14px">${esc(e.label||e.kind)}</b> <span class="muted">${esc(id)} · ${esc(e.kind)}</span>
   <div style="display:grid;grid-template-columns:1fr auto;gap:2px 10px;margin-top:6px"><span>Status</span><b>${esc(st)}</b>
   <span>Modell-Konfidenz</span><b>${fmt(e.conf)}</b>${e.room?`<span>Raum</span><b>${esc(e.room)}${E[e.room]?" · "+esc(E[e.room].label):""}</b>`:""}</div>`;
  if(e.class_probs&&e.class_probs.length)h+=`<div style="margin-top:7px" class="muted">Layout-Modell:</div>`+e.class_probs.map(c=>`<div style="display:flex;gap:6px;align-items:center"><span style="width:110px">${esc(c.label)}</span><div class="pbar" style="flex:1"><i style="width:${c.p*100}%;background:#6366f1"></i></div><span>${Math.round(c.p*100)}%</span></div>`).join("");
  if(evd)h+=`<div style="margin-top:7px">Rolle: <b>${{counted:"gezählt",excluded:"ausgeschlossen",considered:"betrachtet, nicht gezählt",context:"Kontext"}[evd.role]}</b>${evd.order?" (#"+evd.order+")":""}<br><span class="muted">${esc(evd.reason||"")}</span></div>`;
  const p=$("pop"),r=$("stage").getBoundingClientRect();p.innerHTML=h;p.style.display="block";
  p.style.left=Math.min(r.width-285,ev.clientX-r.left+14)+"px";p.style.top=Math.min(r.height-200,Math.max(4,ev.clientY-r.top-20))+"px";}
function tabs(){const T=[["belege","Belege"],["unsicherheit","Unsicherheit"],["agent","Agent"]];
  let h=T.map(([id,l])=>`<span class="tab ${tab===id?"on":""}" data-t="${id}">${l}</span>`).join("");
  if(tab==="belege")h+=`<div class="legend"><span><i class="dot" style="background:#16a34a"></i>gezählt</span><span><i class="dot" style="background:#fff;border:2px dashed #d97706"></i>unter Schwelle, nicht gezählt</span><span><i class="dot" style="background:#dc2626"></i>vom Agenten ausgeschlossen</span><span><i class="dot" style="background:#fff;border:2.5px solid #2563eb"></i>gewählter Schritt</span></div>`;
  if(tab==="unsicherheit")h+=`<div class="legend">`+[["combined","kombiniert"],["rooms","Räume"],["symbols","Symbole"],["text","OCR-Text"]].filter(([c])=>P.heat[c]).map(([c,l])=>`<span class="chip ${chan===c?"on":""}" data-c="${c}">${l}</span>`).join("")+`</div>`;
  if(tab==="agent")h+=`<div class="legend"><span><i class="dot" style="background:#2563eb"></i>gesehen</span><span><i class="dot" style="background:#fff;border:2px solid #94a3b8"></i>nicht gesehen</span><span><i class="dot" style="background:#faf5ff;border:2px dashed #a855f7"></i>für Agent unsichtbar</span><span><i class="dot" style="background:#fff;border:2px dashed #f97316;border-radius:2px"></i>angesehener Ausschnitt</span></div>`;
  $("tabs").innerHTML=h;
  document.querySelectorAll(".tab").forEach(n=>n.onclick=()=>{tab=n.dataset.t;$("pop").style.display="none";render()});
  document.querySelectorAll(".chip[data-c]").forEach(n=>n.onclick=()=>{chan=n.dataset.c;render()});}

/* ------------------------------------------------------------------ Belege */
function viewBelege(){const g=base($("svg")),a=A(),ev=a.evidence||[];
  ev.filter(x=>E[x.element]&&E[x.element].kind==="room"&&x.role==="counted").forEach(x=>shape(E[x.element],{fill:"#16a34a","fill-opacity":.18,stroke:"#16a34a","stroke-width":R*.18},g));
  ev.filter(x=>x.role==="considered").forEach(x=>marker(E[x.element],"#fffbeb","#d97706","?",g,`${R*.4} ${R*.25}`));
  ev.filter(x=>x.role==="excluded").forEach(x=>{if(E[x.element].kind==="room")shape(E[x.element],{fill:"#dc2626","fill-opacity":.12,stroke:"#dc2626","stroke-width":R*.18,"stroke-dasharray":`${R*.5} ${R*.3}`},g);marker(E[x.element],"#dc2626",null,"✕",g)});
  ev.filter(x=>x.role==="counted").forEach(x=>marker(E[x.element],"#16a34a",null,x.order,g));
  hiIds.forEach(id=>{if(E[id]){const[x,y]=center(E[id]);el("circle",{cx:x,cy:y,r:R*1.6,fill:"none",stroke:"#2563eb","stroke-width":R*.2},g)}});
  const nc=ev.filter(x=>x.role==="counted").length,nz=ev.filter(x=>x.role==="considered").length,nx=ev.filter(x=>x.role==="excluded").length;
  $("bottom").innerHTML=`<b>${nc}</b> gezählt · <b>${nz}</b> unter Schwelle verworfen · <b>${nx}</b> vom Agenten ausgeschlossen <span style="margin-left:auto">Element anklicken → Details · Schritt anklicken → beteiligte Elemente</span>`;
  const conf=a.confidence,col=conf>=.8?["#dcfce7","#166534","#16a34a"]:conf>=.5?["#fef3c7","#92400e","#d97706"]:["#fee2e2","#991b1b","#dc2626"];
  const s=a.sources||{},cls=v=>v==null?"":a.truth!=null?(Math.abs(v-a.truth)<1e-6?"ok":"bad"):(a.agreement==="agree"?"ok":"warnc");
  let h=`<div class="card"><div class="h"><span class="num">1</span>Antwort</div><div class="ans"><div class="big">${a.abstained?"–":fmt(a.value)}<small>${a.unit==="count"?"":esc(a.unit||"")}</small></div>
   <div style="margin-left:auto;text-align:right"><span class="badge" style="background:${col[0]};color:${col[1]}"><i style="background:${col[2]}"></i>${({hoch:"hohe",mittel:"mittlere",niedrig:"niedrige"})[a.confidence_label]||a.confidence_label} Sicherheit · ${Math.round(conf*100)} %</span>
   <div class="muted" style="font-size:12px;margin-top:5px">${{agree:"Regeln und Agent stimmen überein",conflict:"Regeln und Agent widersprechen sich",single:"nur eine Quelle"}[a.agreement]||""}</div></div></div>
   ${a.abstained?`<div class="warn"><b>Enthaltung:</b> ${esc(a.abstain_reason||"")}</div>`:""}
   <div class="why"><b>Warum?</b> ${esc(a.summary)}</div><div class="src">
   ${s.rules?`<div class="s ${cls(s.rules.value)}">Regeln<b>${fmt(s.rules.value)}</b></div>`:""}
   ${s.agent?`<div class="s ${cls(s.agent.value)}">LLM-Agent<b>${fmt(s.agent.value)}</b></div>`:""}
   ${a.truth!=null?`<div class="s">Richtige Antwort<b>${fmt(a.truth)}</b></div>`:""}</div>
   ${s.agent&&s.agent.text?`<div class="muted" style="margin-top:9px;font-size:12.8px">Agent: „${esc(s.agent.text)}“</div>`:""}</div>`;
  h+=`<div class="card"><div class="h"><span class="num">3</span>So ist die Antwort entstanden</div>`+steps(a,null)+
     (a.limitations||[]).map(l=>`<div class="warn">⚠ <b>Grenze:</b> ${esc(l)}</div>`).join("")+
     (a.references&&a.references.length?`<div class="muted" style="margin-top:8px;font-size:12px">weitere Belege: ${a.references.map(r=>`<code>${esc(r)}</code>`).join(" ")}</div>`:"")+`</div>`;
  $("right").innerHTML=h;bindSteps(n=>{hiIds=(a.steps.find(s=>s.n===n)||{}).ids||[];render()});}
function steps(a,cur){const C={route:"#64748b",read:"#64748b",tool:"#2563eb",view:"#f97316",check:"#d97706",final:"#16a34a"};
  return a.steps.map(s=>`<div class="step ${cur!=null&&s.n===cur?"on":""} ${cur!=null&&s.n>cur?"todo":""}" data-n="${s.n}"><div class="ic" style="background:${C[s.kind]||"#64748b"}">${s.n}</div><div><b>${esc(s.title)}</b>${s.result?` → ${esc(s.result)}`:""}
   ${s.ids&&s.ids.length?`<br>`+s.ids.slice(0,14).map(i=>`<span class="idc">${esc(i)}</span>`).join("")+(s.ids.length>14?`<span class="muted"> +${s.ids.length-14}</span>`:""):""}</div></div>`).join("")}
function bindSteps(f){document.querySelectorAll(".step").forEach(n=>n.onclick=()=>f(+n.dataset.n))}

/* ------------------------------------------------------------------ Unsicherheit */
function viewUnsicherheit(){const g=base($("svg"));
  el("image",{href:P.heat[chan],x:0,y:0,width:W,height:H,preserveAspectRatio:"none",opacity:hopac},g);
  (D.hotspots||[]).forEach(hs=>{const b=hs.geometry.bbox;el("rect",{x:b[0],y:b[1],width:b[2]-b[0],height:b[3]-b[1],fill:"none",stroke:"#1f2937","stroke-width":R*.12,"stroke-dasharray":`${R*.4} ${R*.3}`},g);
    const c=el("g",{},g);el("circle",{cx:b[0],cy:b[1],r:R*.9,fill:"#1f2937"},c);const t=el("text",{x:b[0],y:b[1]+R*.33,"text-anchor":"middle","font-size":R*.9,"font-weight":700,fill:"#fff"},c);t.textContent=hs.rank});
  hiIds.forEach(id=>{if(E[id])shape(E[id],{fill:"none",stroke:"#2563eb","stroke-width":R*.2},g)});
  $("bottom").innerHTML=`<b>Farbe</b> = Schwanken des Modells (Entropie) <span style="display:inline-block;width:160px;height:10px;border-radius:5px;background:linear-gradient(90deg,#fffacd,#feb24c,#f03b20,#80005a,#280032)"></span> eindeutig → unsicher
    <span style="margin-left:14px">Deckkraft <input type="range" min="0" max="100" value="${hopac*100}" id="op"></span><span style="margin-left:auto">${fmt((D.assets.uncertain_area||0)*100)} % der Planfläche unsicher</span>`;
  $("op").oninput=e=>{hopac=e.target.value/100;viewUnsicherheit()};
  const sh=D.assets.uncertainty_share||{},CL={rooms:"#c026d3",symbols:"#7c3aed",text:"#f59e0b"},CN={rooms:"Raumtypen",symbols:"Symbole",text:"OCR-Texte"};
  let h=`<div class="card"><div class="h">Was zeigt diese Ansicht?</div><div style="font-size:13.5px;line-height:1.5">Für jeden Pixel liefert das Layout-Modell Wahrscheinlichkeiten (Bad, WC, Flur, Tür, Fenster …), die OCR eine Lesesicherheit. Wo sich die Modelle <b>nicht festlegen</b>, ist die Karte dunkel – dort entstehen die meisten Fehler.</div></div>
   <div class="card"><div class="h">Unsicherste Stellen – automatisch gefunden</div>`+
   (D.hotspots||[]).map(hs=>`<div class="hs" data-h="${hs.id}"><div class="rank">${hs.rank}</div><canvas class="thumb" width="144" height="144" data-b="${hs.geometry.bbox.join(",")}"></canvas>
     <div style="flex:1;font-size:12.8px;line-height:1.4"><b>${esc({rooms:"Raumtyp unklar",symbols:"Symbol unklar",text:"Text schwer lesbar"}[hs.channel])}</b> · Entropie ${fmt(hs.entropy)}
     <div class="muted">${(hs.top_classes||[]).map(c=>`${esc(c.label)} ${Math.round(c.p*100)} %`).join(" · ")}</div>
     <div style="font-weight:600;margin-top:2px;color:${hs.resolved_by==="unresolved"?"#b91c1c":"#15803d"}">→ ${esc(hs.resolution||"")}</div>
     ${hs.elements&&hs.elements.length?hs.elements.map(i=>`<span class="idc">${esc(i)}</span>`).join(""):""}</div></div>`).join("")+
   (!(D.hotspots||[]).length?`<div class="muted">Keine deutlich unsicheren Bereiche gefunden.</div>`:"")+`</div>`;
  if(Object.keys(sh).length)h+=`<div class="card"><div class="h">Woher kommt die Unsicherheit?</div><div class="cov">`+Object.entries(sh).map(([c,v])=>`<span>${CN[c]||c}</span><div class="pbar"><i style="width:${v*100}%;background:${CL[c]}"></i></div><b>${Math.round(v*100)} %</b>`).join("")+`</div><div class="muted" style="font-size:12px;margin-top:7px">Anteil an der unsicheren Fläche – zeigt, welches Modell verbessert werden sollte.</div></div>`;
  const a=A(),rel=(D.hotspots||[]).filter(hs=>(hs.elements||[]).some(i=>(a.evidence||[]).some(x=>x.element===i)));
  h+=`<div class="card"><div class="h">Bezug zur gewählten Frage</div><div style="font-size:13.5px;line-height:1.5">${rel.length?`${rel.length} der ${(D.hotspots||[]).length} unsichersten Stellen betreffen Elemente dieser Antwort (Stelle ${rel.map(r=>r.rank).join(", ")}). Ein Klick auf eine Stelle markiert ihre Elemente.`:"Keine der unsichersten Stellen betrifft Elemente dieser Antwort."}</div></div>`;
  $("right").innerHTML=h;
  document.querySelectorAll(".hs").forEach(n=>n.onclick=()=>{hiIds=(D.hotspots.find(x=>x.id===n.dataset.h)||{}).elements||[];viewUnsicherheit()});
  thumbs();}
const IMGS={};function img(src){if(!IMGS[src]){const i=new Image();i.src=src;IMGS[src]=i}return IMGS[src]}
function thumbs(){const pi=img(P.img),hi=img(P.heat[chan]);
  const draw=()=>document.querySelectorAll("canvas.thumb").forEach(c=>{const[x0,y0,x1,y1]=c.dataset.b.split(",").map(Number);
    const s=Math.max(x1-x0,y1-y0)*1.5+20,cx=(x0+x1)/2,cy=(y0+y1)/2,ctx=c.getContext("2d");ctx.fillStyle="#fff";ctx.fillRect(0,0,144,144);
    const sx=pi.naturalWidth/W,hx=hi.naturalWidth/W;ctx.drawImage(pi,(cx-s/2)*sx,(cy-s/2)*sx,s*sx,s*sx,0,0,144,144);ctx.globalAlpha=.85;
    ctx.drawImage(hi,(cx-s/2)*hx,(cy-s/2)*hx,s*hx,s*hx,0,0,144,144);ctx.globalAlpha=1});
  if(pi.complete&&hi.complete)draw();else{pi.onload=hi.onload=()=>{if(pi.complete&&hi.complete)draw()}}}

/* ------------------------------------------------------------------ Agent */
function viewAgent(){const a=A(),g=base($("svg")),cov=a.coverage,st=a.steps;
  if(k==null)k=st.length;
  const AG=["tool","view","check","final"];   // only what the agent itself read (not the pipeline steps)
  const seen=new Set();st.filter(s=>s.n<=k&&AG.includes(s.kind)).forEach(s=>(s.ids||[]).forEach(i=>seen.add(i)));
  const kinds=new Set((a.evidence||[]).map(x=>E[x.element]&&E[x.element].kind).filter(Boolean));
  Object.values(E).filter(e=>e.status==="kept"&&e.kind==="room").forEach(e=>shape(e,{fill:seen.has(e.id)?"#2563eb":"#dbeafe","fill-opacity":seen.has(e.id)?.22:.0,stroke:seen.has(e.id)?"#2563eb":"#cbd5e1","stroke-width":R*.12},g));
  Object.values(E).filter(e=>e.status==="kept"&&e.kind!=="room"&&e.kind!=="text").forEach(e=>{if(seen.has(e.id))marker(e,"#2563eb",null,null,g);else if(kinds.has(e.kind))marker(e,"#fff","#94a3b8",null,g)});
  Object.values(E).filter(e=>e.status==="kept"&&e.kind==="text"&&seen.has(e.id)).forEach(e=>shape(e,{fill:"#2563eb","fill-opacity":.25,stroke:"#2563eb","stroke-width":R*.1},g));
  const blind=new Set((cov&&cov.blind_spots)||[]);(D.blind_spots||[]).filter(b=>blind.has(b.id)).forEach(b=>{const bb=b.geometry.bbox,p=R*.5;
    const r=el("rect",{x:bb[0]-p,y:bb[1]-p,width:bb[2]-bb[0]+2*p,height:bb[3]-bb[1]+2*p,rx:R*.4,fill:"#faf5ff","fill-opacity":.6,stroke:"#a855f7","stroke-width":R*.18,"stroke-dasharray":`${R*.5} ${R*.3}`},g);
    const t=el("text",{x:(bb[0]+bb[2])/2,y:(bb[1]+bb[3])/2+R*.35,"text-anchor":"middle","font-size":R,"font-weight":700,fill:"#7e22ce"},g);t.textContent="?";
    r.style.cursor="pointer";r.onclick=ev=>{ev.stopPropagation();const id=b.id.slice(2);if(E[id])popup(id,ev)}});
  st.filter(s=>s.n<=k&&s.region).forEach((s,j)=>{const b=s.region;el("rect",{x:b[0],y:b[1],width:b[2]-b[0],height:b[3]-b[1],fill:"none",stroke:"#f97316","stroke-width":R*.15,"stroke-dasharray":`${R*.6} ${R*.3}`},g);
    const t=el("text",{x:b[0]+R*.4,y:b[1]+R*1.2,"font-size":R*.95,"font-weight":700,fill:"#c2410c"},g);t.textContent="Blick "+(j+1)+" (Schritt "+s.n+")"});
  const cur=st.find(s=>s.n===k);(cur&&cur.ids||[]).forEach(id=>{if(E[id]){const[x,y]=center(E[id]);el("circle",{cx:x,cy:y,r:R*1.5,fill:"none",stroke:"#f97316","stroke-width":R*.22},g)}});
  $("bottom").innerHTML=`<b>Abdeckung bei Schritt ${k} von ${st.length}</b><span>Alles, was bis hier in einem Werkzeug-Ergebnis vorkam, ist blau – der Rest wurde (noch) nicht gelesen.</span>`;
  let h="";
  if(!cov){h=`<div class="card"><div class="h">Was hat der Agent angesehen?</div><div style="font-size:13.5px">Für diese Antwort gibt es keinen Agenten-Trace (nur Regeln oder Fallback).</div></div>`}
  else{const by={};Object.values(E).filter(e=>e.status==="kept").forEach(e=>{by[e.kind]=by[e.kind]||[0,0];by[e.kind][1]++;if(seen.has(e.id))by[e.kind][0]++});
    const order=Object.keys(by).sort((x,y)=>(kinds.has(y)-kinds.has(x))||by[y][0]-by[x][0]);
    h+=`<div class="card"><div class="h">Was hat der Agent angesehen? <span class="muted" style="text-transform:none;letter-spacing:0">(bis Schritt ${k})</span></div><div class="cov">`+
      order.map(kd=>`<span>${KDE[kd]||kd}${kinds.has(kd)?" ★":""}</span><div class="pbar"><i style="width:${by[kd][1]?by[kd][0]/by[kd][1]*100:0}%;background:${kinds.has(kd)?"#2563eb":"#93c5fd"}"></i></div><b style="${by[kd][0]?"":"color:#9ca3af"}">${by[kd][0]} / ${by[kd][1]}</b>`).join("")+
      `</div><div class="muted" style="font-size:12px;margin-top:7px">★ = Elementart der Antwort. ${(cov.views||[]).length} Bildausschnitt(e) angesehen.</div></div>`;}
  h+=`<div class="card"><div class="h">Ablauf – abspielbar</div><div style="display:flex;gap:10px;align-items:center;margin-bottom:8px">
     <span class="tab" id="prev">◀</span><span class="tab" id="next">▶</span><input type="range" min="1" max="${st.length}" value="${k}" id="scr" style="flex:1"><span>Schritt ${k} / ${st.length}</span></div>`+steps(a,k)+`</div>`;
  const nb=(cov&&cov.blind_spots||[]).length,rk=[...kinds],tot=rk.reduce((s,kd)=>s+Object.values(E).filter(e=>e.status==="kept"&&e.kind===kd).length,0),sk=rk.reduce((s,kd)=>s+Object.values(E).filter(e=>e.status==="kept"&&e.kind===kd&&seen.has(e.id)).length,0);
  if(cov)h+=`<div class="card"><div class="h">Was sagt mir das?</div><div style="font-size:13.5px;line-height:1.55">
     ${tot?(sk===tot?`✓ Der Agent hat <b>alle ${tot}</b> relevanten Elemente gesehen.`:`⚠ Der Agent hat <b>${sk} von ${tot}</b> relevanten Elementen gesehen.`):""}
     ${nb?`<br>⚠ <b>${nb} Stelle(n)</b> waren für ihn <b>unsichtbar</b> (unter Schwelle verworfen oder ohne Detektion) – lila markiert. Hier sollte ein Mensch nachsehen.`:"<br>✓ Keine relevanten Kandidaten unter der Schwelle."}</div></div>`;
  $("right").innerHTML=h;
  $("scr").oninput=e=>{k=+e.target.value;viewAgent()};$("prev").onclick=()=>{k=Math.max(1,k-1);viewAgent()};$("next").onclick=()=>{k=Math.min(st.length,k+1);viewAgent()};
  bindSteps(n=>{k=n;viewAgent()});}

function render(){tabs();({belege:viewBelege,unsicherheit:viewUnsicherheit,agent:viewAgent})[tab]();
  history.replaceState(null,"",`#tab=${tab}&q=${qi+1}`+(tab==="agent"&&k!=null?`&k=${k}`:""))}
(()=>{const h=new URLSearchParams(location.hash.slice(1));   // explain.html#tab=agent&q=2&k=5
  if(["belege","unsicherheit","agent"].includes(h.get("tab")))tab=h.get("tab");
  if(h.get("q")){qi=Math.max(0,Math.min(D.answers.length-1,+h.get("q")-1));$("q").value=qi}
  if(h.get("k"))k=+h.get("k");})();
render();
</script></body></html>"""
