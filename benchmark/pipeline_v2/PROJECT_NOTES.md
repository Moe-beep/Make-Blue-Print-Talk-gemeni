# Junction 2026 – Fragen zu Bauzeichnungen beantworten (Projektnotizen)

Stand: 14.11.2026. Übergabedokument für neue Chat-Sessions („lies PROJECT_NOTES.md“).
Projekt: `/home/aegid/PycharmProjects/Junction` (Linux, Python 3.12, venv `.venv`, nur CPU).

## Challenge
„Build a program that answers natural language questions about construction drawings and
building information models accurately and verifiably.“ Eingaben: PNG, JPG, PDF, DXF, IFC,
mehrsprachig (en, fi, de, da, et …). **Fokus des Nutzers: Explainability** – jede Antwort zeigt,
woher sie kommt (Evidenz-IDs, Begründung, Konfidenz), und das System sagt explizit, wenn es
unsicher ist oder eine Frage nicht beantworten kann (Abstention statt Raten).

## Daten (`questions.jsonl`, 7990 Fragen; Felder path, question, answer, unit, type)
| Quelle | Fragen | Art |
|---|---|---|
| cubicasa5k PNG | 4270 | 12 feste Zählfragen (Türen, Fenster, Bäder, WC, Sauna, …), en/fi/de |
| swiss_dwellings PDF | 2358 | Flächen/Anzahlen, PDFs mit Textlayer, aber ohne gedruckte Flächen |
| ifc_models (IFC + PDF) | 1049 | 827 IFC: Raumfläche „NAME (NR)“ auf Geschoss, Anzahlen/Summen pro Geschoss |
| ifc_bench IFC | 249 | Freitext-Fragen (Materialien, Typen, MEP, Brandschutz, …), Antwort ist Text |
| opening_design | 64 | |

IFC-Fragen verteilen sich auf 8 Dateien (`data/ifc_models/…`, `data/ifc_bench/…`), z. B.
bsc-esplanades (486), bsc-molio (340), digital_hub_arc, duplex_arc, hitos_arc, ac20_arc,
fantasy_office_building_3_arc, bsc-clinic. **Die IFC-Dateien liegen noch nicht im `data/`-Ordner.**

## Architektur
### Bilder (PNG/JPG, PDF-Seite)
1. **OCR** – PaddleOCR 3.7 (PP-OCRv6 `small` Standard; `tiny`/`medium` wählbar), Bild vorher
   geschärft (Lanczos-Upscale auf `--ocr-scale` 1600 + Entrauschen + CLAHE + Unsharp), Kacheln bei großen Bildern.
   `enable_mkldnn=False` (sonst Crash „ConvertPirAttribute2RuntimeAttribute“ in Paddle 3.3).
2. **Layout-Modell** – CubiCasa5K (hg_furukawa_original, 44 Kanäle: 21 Heatmaps, 12 Raum-, 11 Icon-Klassen),
   Gewichte `models/model_best_val_loss_var.pkl` (Legacy-Pickle, `weights_only=False`), Netz vendored in
   `planqa/cubicasa_net.py` (CC BY-NC 4.0). 4-fach-Rotations-TTA, max. Seite 1600 px.
   Liefert Raumpolygone, Symbole (Tür, Fenster, WC, Waschbecken, Wanne, Saunabank, …), Geschossumrisse.
   Kein Duschsymbol → Heuristik: je Bad ohne Wanne eine Dusche.
3. **Plan-Graph** (`build_graph.py`) – Texte/Symbole per Punkt-in-Polygon den Räumen zugeordnet;
   Raumtyp: OCR-Label (finnische Kürzel KH/PESUH/WC/S/MH/K/PARVEKE …) schlägt Modellklasse.
   Detektionen unter Schwellwert werden verworfen (Anzahl in `dropped`).
4. **Antworten** – Standard: **LLM-Agent** (Gemini, Tools auf `plan_graph.json`), der die
   **Regel-Antwort** (`qa.py`, Regex-Routing + Zählen) als Schätzung prüft; Regeln = Fallback.
5. **Report** – `out/<bild>/report.html`: Plan/Layout umschaltbar, Ebenen, Antworten anklicken → Evidenz markiert.

### Gemeinsames Schema `building-graph/1` (`planqa/building_schema.py`)
IFC und Pläne liefern **dieselbe Struktur**; ein Plan füllt nur weniger. Fehlendes = Spalte fehlt / null / `notes`
(nie 0). Teile: `source` (kind ifc|image|pdf, `extraction` mit Schwellwerten und verworfenen Detektionen),
`building`, `storeys` (L*), Tabelle `spaces`, `elements.<gruppe>` (doors, windows, walls, …, **sanitary, furniture**),
`annotations` (OCR-Texte), `types`, `inventory` (MEP/Sonstiges), `mep`, `schedules` + `legends` (Platz für
PDF-Stücklisten, noch kein Extraktor), `relations`, **`facts`** (Standard-Anzahlen/Flächen, für alle Quellen gleich
berechnet, mit `facts.how`), `summary`, `notes`.
Gemeinsame Spalten: id, name, type, storey, space, category/categories, conf (nur Detektion), src
(ifc | symbol | room_polygon | room_label | inferred), box_px bzw. guid. Ganz leere Spalten werden weggelassen.
Gemeinsames Vokabular: `space_categories()` (mehrsprachig fi/de/en/et/fr/it + finnische Kürzel; OH+K = living +
kitchen), `spaces.use` (grobe Gruppe = campi `function_category`), Sanitär toilet/washbasin/sink/shower/bathtub/….
Plan-Ids bleiben r*/t*/o* (Report-Markierung); Dusche wird je Bad ohne Wanne als `src=inferred` angenommen,
Spüle vs. Waschbecken nach Raum. `plan_to_graph.py` konvertiert; die Agent-Tools konvertieren plan_graph.json selbst,
`pipeline.py` schreibt zusätzlich `building_graph.json`. `facts` des Plans = Regel-Antworten (auf 10370 geprüft).
**Abgleich mit `schema_campi.json`** (Feldliste der Fragen): `planqa/campi_map.py` ordnet jedem der 501 Felder
eine Stelle + Status zu (fact/query/derive/interpret/gap/external), `schema_coverage.py` prüft die Stellen gegen die
Extraktoren und schreibt `schema_coverage.md`. Stand: floor_plan 82 %, bim_model 94 %, construction_drawings 3 %
(nur Schema-Platz, PDF-Extraktor fehlt) beantwortbar; 13 Felder extern (Abstention).

### IFC
`ifc_to_json.py` → `out/<name>/ifc_graph.json` (Schema `building-graph/1`, früher `ifc-graph/1`):
SI-Einheiten, Raumbaum Gebäude › Geschosse (L*) › Räume (S*), Typen einmal in `types` (T*),
Elemente als Tabellen (columns + rows), Inventar (Möbel/Sanitär/MEP je Typ, by_storey/by_space),
MEP-Systeme, Relationen adjacent / connected_via / faces_outside / connects_storeys
(aus IfcRelSpaceBoundary oder 5-cm-Raster der Geometrie), vorberechnete Summen, `notes` zu Datenlücken.
`ifc_api.py` = einziger Teil mit ifcopenshell (**noch nie gegen echte IFC-Dateien gelaufen**),
`ifc_extract.py` = Aufbau (mit synthetischem Gebäude getestet).

### LLM-Agent (`planqa/agent.py`)
Modell sieht nie das ganze JSON: Tools `overview`, `get(path)`, `query_table(where, sort, limit)`,
`aggregate(count/sum/avg/min/max, group_by)`, `search(text)`, `calculate` (sicherer Rechner) und
`final_answer(answer, value, unit, evidence, confidence, abstained, reason)`. Jeder Tool-Aufruf landet im Trace.
Backend Gemini über `google-genai` (manuelles Function Calling, Retry bei 429/503), API-Version fest
`v1beta` (`$GEMINI_API_VERSION`; auf `v1` gab es 404 für `gemini-flash-latest`). Modell `--model` /
`$GEMINI_MODEL` / `gemini-flash-latest`; wird beim Start gegen ListModels geprüft, sonst automatisch das
neueste Flash-Modell (`GeminiBackend.pick_model`). `--judge` bewertet Freitext gegen die Referenz.
Jede Frage startet einen **neuen Verlauf**; innerhalb einer Frage wird der ganze Verlauf bei jedem Schritt neu
gesendet (Prompt-Tokens wachsen). Tempo: `overview` (mit `facts`) steht schon in der Frage (spart eine Runde),
`--thinking off|low|medium|high|default` (Standard low; Gemini 2.x thinking_budget, 3.x thinking_level, Rückfall auf
default wenn nicht unterstützt), `query_table` blendet box_px/guid/… aus, Limit 25, Ergebnis max. 4000 Zeichen,
`--llm-workers N` (Fragen eines Bildes parallel, Standard 1). Erster echter Lauf (10321, altes Plan-Format, vor
diesen Änderungen): 15/24 Fragen am Schrittlimit, Ø 45 s und 40 000 Prompt-Tokens je Frage.
Zweiter Lauf (neues Schema, thinking low): 2–4 Schritte statt bis zu 10, 10321 in 221 s statt 1100 s; LLM 63/90,
Regeln 58/90. 7 Abbrüche „'str' object has no attribute 'get'“: Gemini schickte Filter als Text → `_where()` versteht
Text/JSON-Strings, jeder Tool-Fehler geht als Fehlermeldung an das Modell zurück statt die Frage abzubrechen.
**`--vision`**: Tool `view(id | box_px | 'plan')` zeigt dem Modell einen Ausschnitt der Originalzeichnung (max. 3 pro
Frage, PNG als eigener Teil nach den Funktionsantworten, nicht im Trace). Gedacht zum Lesen fehlender/verschriebener
Labels, Prüfen einzelner Räume und der Geschosszahl – nicht zum Nachzählen vieler Symbole. `ask.py --image`.
Plan-Labels robuster (`plan_to_graph.py`, wirkt auf `facts` UND die Raum-Regeln in answer.py, `room_answer`):
OCR-Tippfehler per Editierdistanz gegen `FUZZY_WORDS` (M.AKUUHVONE, ETENEN), Kategorie auch aus anderen Texten im
Raum (OH+K, MH4(VARAUS) – als reserviert markiert), zweites Raumlabel im Raum = eigener Raum (Sauna „s“);
Maßangaben wie 9x21 als annotations.kind opening_size. facts auf den 4 Benchmark-Plänen: 28/45 (vorher 27/45).
**Entscheidung des Nutzers: kein Tuning auf dem Testset.** Auf den 4 Benchmark-Bildern abgeleitete Sonderregeln
(Kürzel SA/RT, „Teilstück eines beschrifteten Raums nicht zählen“ mit Wandabstand und Raumarten-Liste) wurden wieder
entfernt; bekannte Folge: 10321 zählt 4 statt 2 Küchen (OH+RT+K-Räume + Küchen-Polygone des Layout-Modells).
Neue Regeln nur noch nach Messung auf einem getrennten Validierungs-Split (offener Punkt 11).
Lauf mit --vision (14566 86 %, 2609 77 %, 10370 73 %, 10321 71 %): view() löste Balkone, WC, VARAUS, Duschen; neue Fehler
durch Nachzählen per Auge (14566 Türen 9 aus der Gesamtansicht, 10321 Waschbecken 8). en/fi-Paare antworten 4× verschieden
→ ±1 Frage pro Bild ist Rauschen. Von 20 Abweichungen Agent≠Daten waren 12 richtig (Daten 3).
Daraufhin: (1) **Abweichungen einzeln begründen**: weicht der Zählwert von der Schätzung (Regel = facts) ab, muss
final_answer `changes` liefern (add/remove je Element, mit Daten-Id oder Detail-Ansicht `view:<id>`, Summe = Differenz),
sonst bleibt die Schätzung (`Agent._check_override`, Felder llm_changes / llm_override, Ausgabe „OVERRIDE …“);
`--free-overrides` schaltet das ab (A/B-Vergleich). (2) `view('plan')` = Übersicht in reduzierter Auflösung, nur für
Layout/Geschosse, zählt nicht als Beleg; Ausschnitte nie unter Originalauflösung (sonst Hinweis „reduced“).

## Dateien
| Datei | Rolle |
|---|---|
| `answer.py` | Benchmark/QA für Bilder: Modelle → Graph → Agent/Regeln → Tabelle, `out/predictions.jsonl` |
| `extract_plan.py` | nur Graph + Report für Bilder |
| `ask.py` | Agent für einzelne Fragen (`--file .ifc/.json`) und IFC-Evaluation aus `questions.jsonl` |
| `ifc_to_json.py` | IFC → `ifc_graph.json`, `--schema` exportiert `building_graph.schema.json` |
| `bench_ocr.py` | OCR-Konfigurationen vergleichen (Tempo vs. Qualität) |
| `test_questions.jsonl` | **Benchmark des Nutzers**: 90 Fragen auf 4 Bildern, Feld `difficulty` |
| `planqa/ocr.py, layout.py, cubicasa_net.py` | Modelle |
| `planqa/build_graph.py, schemas.py, pipeline.py` | Graph, Schwellwerte, Pipeline (Modelle einmal geladen) |
| `planqa/qa.py` | Regeln: INTENTS, LABELS (Kürzel), PRIOR (häufigste Antwort = Baseline) |
| `planqa/visualize.py` | HTML-Report |
| `planqa/agent.py` | LLM-Agent + Tools |
| `planqa/ifc_schema.py, ifc_api.py, ifc_extract.py` | IFC (`ifc_schema.py` = Alias auf das gemeinsame Schema) |
| `planqa/building_schema.py` | gemeinsames Schema, Vokabular, `compute_facts` |
| `planqa/plan_to_graph.py` | plan_graph → building-graph |
| `planqa/campi_map.py`, `schema_coverage.py`, `schema_campi.json` | Abgleich mit der Feldliste der Fragen → `schema_coverage.md` |
| `tests/test_offline.py, test_ifc_offline.py, test_agent_offline.py, test_building_graph_offline.py` | Offline-Tests (ohne Modelle/Netz) |

## Befehle
```
pip install -r requirements.txt          # torch CPU separat: pip install torch --index-url https://download.pytorch.org/whl/cpu
export GEMINI_API_KEY=...                # Hackathon-Key

python answer.py --min-conf-text 0.65 --min-conf-symbols 0.65 --min-conf-rooms 0.65   # Benchmark mit LLM
python answer.py --no-llm                # nur Regeln
python answer.py --sweep                 # + Schwellwert-Raster 0.3–0.9 (nur Regeln, Sekunden)
python answer.py --image data/x.png --question "How many windows ...?"
python ask.py --list-models
python ask.py --file data/ifc_bench/duplex_arc.ifc --question "How many doors are there?"
python ask.py --questions questions.jsonl --sample 30 [--judge] [--types text] [--files duplex]
python ifc_to_json.py data/ifc_models/*.ifc [--no-geometry]
python tests/test_offline.py && python tests/test_ifc_offline.py && python tests/test_agent_offline.py \
  && python tests/test_building_graph_offline.py
python schema_coverage.py                # schema_campi.json vs. building-graph -> schema_coverage.md
python ifc_to_json.py --schema           # building_graph.schema.json
```
Schwellwerte: `--min-conf-text` (0.6), `--min-conf-symbols` (0.6), `--min-conf-rooms` (0.8), `--min-conf X` setzt alle.

## Benchmark (4 Bilder, 90 Fragen = 4 Bilder × 2 Sprachen × Typen)
| Bild | Schwierigkeit |
|---|---|
| high_quality_architectural-10370.png | einfach |
| high_quality_architectural-14566.png | mittel |
| high_quality_architectural-2609.png | sehr schwer |
| high_quality_architectural-10321.png | extrem |

Schwierigkeit wurde aus den Ground-Truth-Anzahlen abgeleitet, nicht aus der Modellschwierigkeit.
`accuracy` = exakt richtig (Anzahlen), `baseline` = immer häufigste Antwort des Fragetyps,
`mean err` = Vorhersage − Wahrheit, `MAE` = mittlere Abweichung. n = 8 pro Typ heißt effektiv 4 Datenpunkte (en/fi gleiche Antwort).

**Letzter gemessener Stand (Regeln, alle Schwellwerte 0.8, ohne LLM):** 48,9 % vs. 40,0 % Baseline.
Fehler: **46 zu wenig, 0 zu viel** → systematisches Unterzählen. Größte Fehler Fenster (−4,75), Türen (−4,25),
Waschbecken (−1,5). Bei 10321 wurden 65 OCR-Texte verworfen und nur 49 behalten.
Ergebnisse mit gesenkten Schwellwerten und mit LLM: **noch nicht gemessen.**

## IFC-Benchmark (`test_questions_ifc.jsonl`, gebaut von `make_ifc_benchmark.py`, Seed 0)
| Modell | Schwierigkeit | Fragen |
|---|---|---|
| hitos_arc.ifc (64 MB, ~2000 Bauteile, 243 Räume, 7 Geschosse) | extrem | 24 Freitext |
| bsc-molio (74 MB, 10 Geschosse) | sehr schwer | 12 Sprachpaare en/da: 8 count, 8 area, 4 length, 4 volume |
| bsc-esplanades (285 Räume, 7 Geschosse, Einzelabfragen) | mittel | 12 Sprachpaare en/et: 12 count, 12 area |
| duplex_arc.ifc (1,7 MB, ~200 Bauteile, 21 Räume) | einfach | 24 Freitext |
Schwierigkeit aus Modellgröße und Fragetyp, nicht aus der Leistung des Systems. Auswertung:
`python ask.py --benchmark --llm-workers 4` (= --questions test_questions_ifc.jsonl --judge; Tabelle pro Modell wie
answer.py; Zahlen: count exakt, sonst 5 %; Freitext: LLM-Judge correct/partial/wrong – derselbe Gemini, also kein
unabhängiger Richter). Keine Regel-Schätzung für IFC → keine Abweichungsprüfung (`changes`) hier.
**ifc_api.py ist noch nie gegen echte IFC gelaufen** (ifcopenshell in Claudes Sandbox nicht installierbar): der erste
Benchmark-Lauf ist zugleich dieser Test; bei Problemen erst `python ifc_to_json.py data/ifc_bench/duplex_arc.ifc`.

Erster IFC-Lauf (molio 100 %, esplanades 50 %, duplex 37,5 %, hitos: Umwandlung gescheitert, Fehler noch offen):
- **Geometrie-Fehler behoben** (betraf jedes echte IFC): Raumkörper haben Boden UND Decke; ein cv2.fillPoly-Aufruf mit
  allen Dreiecken füllt nach der Gerade-Ungerade-Regel → Grundriss löscht sich aus (20 m² → 1,15 m²). Jetzt `_fill`
  (jedes Dreieck einzeln). Verfälscht waren Raumflächen aus Geometrie, BGF-Schätzung, Raum-Wand- und Raum-Raum-Relationen.
- **float32 bei georeferenzierten Modellen** (esplanades: x≈538 497, y≈6 591 600 → 0,5-m-Raster): Geometrie wird vor
  float32 auf einen lokalen Nullpunkt verschoben, Schwerpunkte wieder in Weltkoordinaten.
- **Angegebene Flächen vor Geometrie** (`space_area`): Standardmengen, dann jede Eigenschaft/Menge mit Flächen-Namen
  (mehrsprachig, Textwerte wie „5,1“), z. B. ArchiCAD AR_Ruum.120_Pindala, Revit „GSA Space Areas.GSA BIM Area“;
  Spalte `spaces.area_src`, Hinweis bei > 15 % Abweichung zur Geometrie. `ifc_api` erkennt Mengen-Sets über
  ifcopenshell (qtos_only), nicht nur am Namen. Geprüft mit STEP-Parser an den echten Dateien: esplanades 6/6, molio 3/3.
- ask.py: Referenzantwort blieb in agent_predictions.jsonl nicht erhalten (überschrieben) → Agent-Antwort jetzt `llm_answer`.
- hitos (2006, „IFC Engine DLL test“) scheiterte an 1742 Property-/Mengen-Sets ohne Namen (`$` → None → .startswith):
  unbenannte Sets heißen jetzt „Unnamed set“ (Werte zusammengeführt), ifc_extract ist gegen None-Namen robust.
- Zweiter IFC-Lauf: molio 100 %, esplanades 100 %, hitos 50 %, duplex 37,5 %. Danach (allgemein, an echten Dateien geprüft):
  `NetFootprintArea` (auch in unbenannten Mengen-Sets) als Raumfläche → hitos 243/243 Räume, Summe 8491,3 m² = Referenz;
  Raumfläche aus der eigenen Maske (überlappende Räume schneiden sich nicht mehr aus); Räume, die ≥ 2 andere enthalten
  (Bruttoareal-/Zonen-Räume) → Spalte `contains`, zählen nicht für Nachbarschaft/Türen; IsExternal fehlt → aus Geometrie
  (Wand berührt Außenseite des Geschossumrisses; Tür führt nach OUTSIDE), Hinweis in notes; Träger → `beams.spaces`
  (IfcBeam jetzt mit Geometrie); Typnamen nur „Trenner+Zahl“ abschneiden (T01-0 → T01, T03 bleibt); Sanitär zählt in
  mep.by_class; Storey.gross_floor_area_src („estimate…“ = nicht im Modell); letzter Agent-Schritt erzwingt final_answer.
  Nicht geändert: hitos „Vindusfelt“ (Fensterbänder 19,5 m) sind echte Daten, die Referenz zählt nur normale Fenster.
- Dritter IFC-Lauf: Zahlen 48/48 richtig, Freitext 52 % (+29 % partial); duplex 41,7 %, hitos 62,5 %. Danach:
  **Herkunft jeder Antwort** (Entscheidung des Nutzers, ersetzt „nie schätzen“): final_answer.provenance
  stated | derived | estimate | not_available + assumptions; nicht direkt im Modell → „Not directly in the model: …“,
  dann abgeleiteter Wert oder klar markierte Schätzung mit Annahmen; abstained nur, wenn beides nicht geht.
  Abgeleitetes in den Daten markiert: walls/doors.ext_src (Geometrie), Notiz zu Wandlängen aus Geometrie,
  mep.by_domain (Fachbereich aus IFC-Klasse/Typ, „DERIVED“ wenn keine IfcSystem), beams.spaces = {Raum: Anteil des
  Trägers darüber}. Container-Räume im Prompt neutral („sag, ob du sie einbezogen hast“).
  Verworfen: ext aus IfcRelSpaceBoundary – in hitos widersprüchlich (528 außen / 164 innen vs. Referenz ~267/582),
  die Geometrie-Ableitung (262 außen) liegt näher; 77 Wände ohne Zuordnung bleiben offen.
  Bewusst nicht angepasst (Referenz-Konventionen): Bruttoareal als größter Raum, Fensterbänder, Fenstertypen,
  Geländertyp, Fundamenttiefe, „wahrscheinlich tragend“.
- Offen/Entscheidung: Freitext-Referenzen erwarten oft „nicht im Modell, aber geschätzt ca. X“ – der Agent enthält sich
  ohne Schätzung (→ partial). Raumbezug von Trägern („steel beams spanning the living room“) fehlt im Graphen.

## Entscheidungen des Nutzers
- Gemini-Kachel-Extraktion aus der Pipeline entfernt (Ergebnisse unbrauchbar); Gemini nur noch als Agent auf dem JSON.
- Kein Cache, keine parallelen Worker; Detektionen werden bei jeder Evaluation neu erzeugt (Modelle bleiben pro Lauf geladen).
- Schwellwerte getrennt für Texte/Symbole/Räume (statt einheitlich 0.8).
- Balkone: unbeschriftete Außenflächen zählen nicht, wenn der Plan Außenflächen beschriftet (PARVEKE/TERASSI).
- Antworten auf Deutsch; Erklärungen im Report auf Deutsch, Raumtypen englisch.
- Google Stitch (UI-Design-Tool) ist für das Projekt nicht relevant.

## Offene Punkte (priorisiert)
1. Echten Lauf mit `GEMINI_API_KEY` machen: LLM vs. Regeln vergleichen, Fehler im Agenten beheben.
2. `--sweep` laufen lassen, beste Schwellwerte wählen.
3. Layout-Modell in überlappenden Kacheln in voller Auflösung (große Pläne wie 10321 verlieren kleine Symbole).
4. Verschmolzene Symbole trennen (zwei Türen nebeneinander = 1 Komponente), z. B. über typische Größe/Watershed.
5. Plausibilitätsregeln (jede Küche/jedes Bad ≥ 1 Waschbecken; Sauna braucht Bank oder Label S/SAUNA).
6. IFC-Dateien nach `data/` legen, `ifc_to_json.py` auf echten Modellen testen (ifcopenshell-API-Unterschiede möglich).
7. Regelbasierte Antworten für die festen IFC-Fragevorlagen (Raumfläche, Anzahlen/Summen pro Geschoss) –
   deterministisch, LLM nur für Freitext; dann IFC-Benchmark analog zu den Bildern.
8. PDFs (swiss_dwellings, ifc_models-PDFs) über Textlayer statt OCR; DXF noch gar nicht abgedeckt.
9. Lücken aus `schema_coverage.md`: PDF-Stücklisten → `schedules`/`legends`; Wohnungen (`spaces.unit`, IfcZone bzw.
   Wohnungsbuchstaben im Plan); Raumnummern im Plan (`spaces.number`); IfcApplication-Entwickler, Gebäude-Klassifikation,
   Fundamenttiefe, Träger über Räumen, Fluchtweglängen.
10. Restfehler auf dem Benchmark: Fenster immer zu wenig (36/44, 4/7, 21/25, Layout-Modell → Kacheln, Punkt 3),
    Waschbecken/WC-Symbole fehlen (10321, 10370), Geschosse 10321 (1 statt 4: nur ein Umriss erkannt, obere Hälfte
    des Blatts fehlt), unbeschriftete gespiegelte Balkone (10321 r02), Sauna „LO“ (= LH?) in 10321. Mit --vision testen.
12. Raster2Seq (SIGGRAPH 2026, Gewichte auf Hugging Face `haopt/Raster2Seq`, CubiCasa-Checkpoint): Räume, Türen, Fenster als
    Polygone. Nur als zweite Quelle für Räume/Geschossumrisse/Türen/Fenster mit Abgleich (Uneinigkeit = Unsicherheit) sinnvoll;
    256-px-Eingabe, keine Sanitärsymbole/Texte. Vorher prüfen, ob die Benchmark-Bilder im CubiCasa-Trainingssplit liegen
    (gilt auch für das jetzige Layout-Modell). Auf dem Validierungs-Split messen.
11. Validierungs-Split aus questions.jsonl (CubiCasa-Bilder ohne die 4 Benchmark-Bilder, feste Liste): Regeländerungen
    dort einzeln messen (mit/ohne), Label-Vokabular aus Häufigkeiten statt von Hand; die 4 Bilder nur für die Endmessung.

## Arbeitsweise / Stolperfallen
- Claude arbeitet in einer Cloud-Sandbox (ohne Netz, ohne ifcopenshell/torch/paddle/google-genai) und
  überträgt Dateien über die Desktop-Verbindung; Tests nutzen Fake-Module.
- **PyCharm überschreibt Dateien**, die mit ungespeicherten Änderungen offen sind, mit dem alten Editorinhalt.
  Vor Änderungen durch Claude: betroffene Tabs schließen bzw. danach File → Reload All from Disk.
  Notlösung: Dateien in `_update/` ablegen und mit `cp -r _update/* . && rm -r _update` übernehmen.
- Claude darf auf dem Rechner nichts löschen; Löschbefehle bekommt der Nutzer als Text.
