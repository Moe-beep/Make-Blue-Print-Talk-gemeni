import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export interface CorpusQA {
  path: string;
  source: string;
  question: string;
  answer: string | number;
  unit: string | null;
  type: string;
}

export interface CustomTrainingEntry extends CorpusQA {
  id: string;
  kind: 'uploaded' | 'correction';
  wrongAnswer?: string;
  ruleNote?: string;
  createdAt: string;
}

interface CustomTrainingStore {
  entries: CustomTrainingEntry[];
}

const CORPUS_GZ_PATH = path.join(__dirname, 'questionsCorpus.jsonl.gz');
const CUSTOM_STORE_PATH = path.join(__dirname, 'customTraining.json');

function normalizeStem(filePathOrName: string): string {
  const base = path.basename(filePathOrName).trim().toLowerCase();
  return base.replace(/\.[^.]+$/, '');
}

function normalizeFileName(filePathOrName: string): string {
  return path.basename(filePathOrName).trim().toLowerCase();
}

function classifyIntent(question: string): string[] {
  const q = question.toLowerCase();
  const intents: string[] = [];
  if (/lamp|light|luminaire|sconce|pendant|chandelier|valaisin|fixture/.test(q)) {
    intents.push('lighting');
  }
  if (/door|ovea|tür|porte|port/.test(q)) intents.push('door');
  if (/window|ikkuna|fenster|fenêtre/.test(q)) intents.push('window');
  if (
    /toilet|wc|sink|basin|lavabo|shower|suihku|bath|amme|wanne|sanitary|sauna/.test(
      q,
    )
  ) {
    intents.push('sanitary');
  }
  if (
    /area|pinta-ala|fläche|oppervlakte|superficie|sq\s*m|m²|ft²|sf|footprint/.test(
      q,
    )
  ) {
    intents.push('area');
  }
  if (/height|korkeus|höhe|ceiling|floor-to-floor|volume|m³/.test(q)) {
    intents.push('height_volume');
  }
  if (/wall|wand|muur|slab|stair|beam|column|schedule|finish/.test(q)) {
    intents.push('structural_schedule');
  }
  if (
    /room|bedroom|makuuhuone|schlafzimmer|wohnen|kitchen|keittiö|küche|space|apartment|wohnung/.test(
      q,
    )
  ) {
    intents.push('rooms');
  }
  if (intents.length === 0) intents.push('general');
  return intents;
}

class TrainingEngine {
  private baseCorpusCount = 0;
  private uniqueDrawings = new Set<string>();
  private byExactName = new Map<string, CorpusQA[]>();
  private byStem = new Map<string, CorpusQA[]>();
  private intentExemplars = new Map<string, CorpusQA[]>();
  private customEntries: CustomTrainingEntry[] = [];

  constructor() {
    this.loadAll();
  }

  private addToIndexes(item: CorpusQA) {
    const exact = normalizeFileName(item.path);
    const stem = normalizeStem(item.path);
    if (exact) {
      this.uniqueDrawings.add(item.path);
      const list = this.byExactName.get(exact);
      if (list) list.push(item);
      else this.byExactName.set(exact, [item]);
    }
    if (stem) {
      const list = this.byStem.get(stem);
      if (list) list.push(item);
      else this.byStem.set(stem, [item]);
    }
  }

  private loadCustomStore(): CustomTrainingEntry[] {
    if (!fs.existsSync(CUSTOM_STORE_PATH)) return [];
    try {
      const parsed = JSON.parse(
        fs.readFileSync(CUSTOM_STORE_PATH, 'utf-8'),
      ) as CustomTrainingStore;
      return Array.isArray(parsed.entries) ? parsed.entries : [];
    } catch {
      return [];
    }
  }

  private saveCustomStore() {
    const payload: CustomTrainingStore = { entries: this.customEntries };
    fs.writeFileSync(
      CUSTOM_STORE_PATH,
      JSON.stringify(payload, null, 2),
      'utf-8',
    );
  }

  private loadAll() {
    this.baseCorpusCount = 0;
    this.uniqueDrawings.clear();
    this.byExactName.clear();
    this.byStem.clear();
    this.intentExemplars.clear();

    const seenTemplates = new Map<string, Set<string>>();
    const recordExemplar = (key: string, row: CorpusQA) => {
      let bucket = this.intentExemplars.get(key);
      if (!bucket) {
        bucket = [];
        this.intentExemplars.set(key, bucket);
        seenTemplates.set(key, new Set());
      }
      if (bucket.length >= 25) return;
      const sig = `${row.source}:${row.question.slice(0, 48)}`;
      const seen = seenTemplates.get(key)!;
      if (seen.has(sig)) return;
      seen.add(sig);
      bucket.push(row);
    };

    if (fs.existsSync(CORPUS_GZ_PATH)) {
      const gz = fs.readFileSync(CORPUS_GZ_PATH);
      const raw = zlib.gunzipSync(gz).toString('utf-8');
      const lines = raw.split('\n');
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        try {
          const r = JSON.parse(trimmed) as {
            path?: string;
            question?: string;
            answer?: string | number;
            unit?: string | null;
            type?: string;
          };
          if (!r.question || r.answer === undefined) continue;
          const p = r.path || '';
          const src = p.includes('/') ? p.split('/')[1] : 'custom';
          const item: CorpusQA = {
            path: p,
            source: src,
            question: r.question,
            answer: r.answer,
            unit: r.unit ?? null,
            type: r.type ?? 'text',
          };
          this.baseCorpusCount++;
          this.addToIndexes(item);

          for (const intent of classifyIntent(item.question)) {
            recordExemplar(`${src}:${intent}`, item);
            recordExemplar(`all:${intent}`, item);
          }
        } catch {
          // Skip malformed lines
        }
      }
    }

    this.customEntries = this.loadCustomStore();
    for (const entry of this.customEntries) {
      this.addToIndexes(entry);
    }
  }

  getStatus() {
    const customQuestionsCount = this.customEntries.filter(
      (e) => e.kind === 'uploaded',
    ).length;
    const correctionsCount = this.customEntries.filter(
      (e) => e.kind === 'correction',
    ).length;
    return {
      baseCorpusQuestions: this.baseCorpusCount,
      baseUniqueDrawings: this.uniqueDrawings.size,
      customQuestionsCount,
      correctionsCount,
      recentCustomEntries: this.customEntries.slice(-20).reverse(),
    };
  }

  ingestTrainingPayload(
    rawText: string,
    defaultFileName = 'training-batch.json',
  ): { addedCount: number; totalCustom: number } {
    const trimmed = rawText.trim();
    if (!trimmed) {
      throw new Error('Uploaded training file is empty.');
    }

    let rawRecords: unknown[] = [];
    if (trimmed.startsWith('{') || trimmed.startsWith('[')) {
      try {
        const parsed = JSON.parse(trimmed) as unknown;
        if (Array.isArray(parsed)) {
          rawRecords = parsed;
        } else if (parsed && typeof parsed === 'object') {
          const obj = parsed as Record<string, unknown>;
          if (Array.isArray(obj.questions)) rawRecords = obj.questions;
          else if (Array.isArray(obj.data)) rawRecords = obj.data;
          else if (Array.isArray(obj.examples)) rawRecords = obj.examples;
          else if (Array.isArray(obj.items)) rawRecords = obj.items;
          else rawRecords = [obj];
        }
      } catch {
        // Fallback to JSONL line parsing
        rawRecords = trimmed
          .split('\n')
          .map((l) => l.trim())
          .filter(Boolean)
          .map((l) => JSON.parse(l) as unknown);
      }
    } else {
      rawRecords = trimmed
        .split('\n')
        .map((l) => l.trim())
        .filter(Boolean)
        .map((l) => JSON.parse(l) as unknown);
    }

    const now = new Date().toISOString();
    let addedCount = 0;

    for (const rec of rawRecords) {
      if (!rec || typeof rec !== 'object') continue;
      const r = rec as Record<string, unknown>;
      const q =
        (typeof r.question === 'string' && r.question) ||
        (typeof r.q === 'string' && r.q) ||
        (typeof r.prompt === 'string' && r.prompt) ||
        '';
      const ans =
        r.answer !== undefined
          ? r.answer
          : r.a !== undefined
            ? r.a
            : r.label !== undefined
              ? r.label
              : r.response;

      if (!q.trim() || ans === undefined || ans === null) continue;

      const p =
        (typeof r.path === 'string' && r.path) ||
        (typeof r.file === 'string' && r.file) ||
        (typeof r.fileName === 'string' && r.fileName) ||
        (typeof r.image === 'string' && r.image) ||
        defaultFileName;

      const entry: CustomTrainingEntry = {
        id: `up-${Date.now()}-${addedCount}`,
        path: p,
        source: 'user_uploaded',
        question: q.trim(),
        answer:
          typeof ans === 'number' || typeof ans === 'string'
            ? ans
            : JSON.stringify(ans),
        unit: typeof r.unit === 'string' ? r.unit : null,
        type:
          typeof r.type === 'string'
            ? r.type
            : typeof ans === 'number'
              ? 'count'
              : 'text',
        ruleNote: typeof r.note === 'string' ? r.note : undefined,
        kind: 'uploaded',
        createdAt: now,
      };
      this.customEntries.push(entry);
      this.addToIndexes(entry);
      addedCount++;
    }

    if (addedCount === 0) {
      throw new Error(
        'No valid { question, answer } entries were found in the uploaded JSON/JSONL file.',
      );
    }

    // Keep store bounded to latest 5000 custom entries for fast startup
    if (this.customEntries.length > 5000) {
      this.customEntries = this.customEntries.slice(-5000);
    }
    this.saveCustomStore();

    return { addedCount, totalCustom: this.customEntries.length };
  }

  recordCorrection(params: {
    fileName: string;
    question: string;
    wrongAnswer?: string;
    correctedAnswer: string;
    ruleNote?: string;
  }): CustomTrainingEntry {
    const entry: CustomTrainingEntry = {
      id: `cor-${Date.now()}`,
      path: params.fileName || 'general',
      source: 'user_correction',
      question: params.question.trim(),
      wrongAnswer: params.wrongAnswer?.trim(),
      answer: params.correctedAnswer.trim(),
      unit: null,
      type: 'text',
      ruleNote: params.ruleNote?.trim(),
      kind: 'correction',
      createdAt: new Date().toISOString(),
    };
    this.customEntries.push(entry);
    this.addToIndexes(entry);
    this.saveCustomStore();
    return entry;
  }

  clearCustomTraining() {
    this.customEntries = [];
    if (fs.existsSync(CUSTOM_STORE_PATH)) {
      fs.unlinkSync(CUSTOM_STORE_PATH);
    }
    this.loadAll();
  }

  buildTrainedSystemInstruction(): string {
    const customRules = this.customEntries
      .filter((e) => e.kind === 'correction' || e.ruleNote)
      .slice(-15)
      .map(
        (e) =>
          `- [${normalizeFileName(e.path)}] Q: "${e.question}" -> VERIFIED CORRECTION: ${JSON.stringify(e.answer)}${e.ruleNote ? ` (Rule: ${e.ruleNote})` : ''}`,
      );

    return `You are Stoa AI, an architectural, interior, and BIM/CAD quantity-takeoff spatial analyst trained directly on the full Junction 2026 corpus (${this.baseCorpusCount.toLocaleString()} verified Q&As across ${this.uniqueDrawings.size.toLocaleString()} drawings & 3D IFC/DXF/PDF/PNG models${this.customEntries.length > 0 ? `, plus ${this.customEntries.length} user-supplied training entries` : ''}).

LEARNED GROUND-TRUTH DOMAIN RULES (MANDATORY):
1. Multilingual Room & Label Taxonomies:
   - Finnish (CubiCasa5k): MH = Makuuhuone (Bedroom), OH = Olohuone (Living Room), K / KT / KK = Keittiö / Keittotila / Keittokomero (Kitchen — include open kitchens/kitchenettes when counting kitchens!), PH / KPH / KH / PESUH / SH = Pesuhuone / Kylpyhuone / Suihkuhuone (Bathroom/washroom — DO NOT double-count a separate WC inside a bathroom as a separate WC room!), WC = Separate Toilet Room (only count rooms specifically marked as separate WC rooms when asked for separate WC rooms, whereas "toilet bowls / WC seats" counts every drawn WC fixture across all rooms), S = Sauna, ET = Eteinen (Entry/Hallway), VH = Vaatehuone (Closet), KHH = Kodinhoitohuone (Utility Room), PARV / P = Parveke (Balcony, including glazed balconies).
   - German / French / Italian (Swiss Dwellings): Schlafzimmer / Chambre à coucher / Camera da letto = Bedroom; Zimmer / Pièce / Stanza = Room; Wohnen / Wohnen/Essen / Séjour / Soggiorno = Living/Dining; Küche / Cuisine / Cucina = Kitchen; Bad / Bain / Bagno = Bathroom; Korridor / Couloir / Corridoio = Hallway/Corridor; Reduit / Réduit / Ripostiglio = Storage/Utility Closet; Treppenhaus / Cage d'escalier / Vano scala = Stairwell; Balkon / Balcon / Balcone / Loggia = Exterior Balcony/Loggia.
   - Dutch / Danish / Spanish / Estonian (IFC Models): Ruimte / Rum / Espacio / Ruum = Space/Room; Begane grond / Grundplan / Sal / Korrus = Floor/Level; Elutuba+Köök = Living+Kitchen.
2. Exact Dataset Counting Rules:
   - DOORS: Count doorways fitted with a hinged/sliding door leaf or swing arc (interior doors + entrance doors together). Open archways or cased openings WITHOUT a door leaf MUST NOT be counted as doors. Double doors count as 1 door unit unless leaf count is explicitly requested.
   - WINDOWS: Count every distinct window opening/glazing unit along exterior and interior partitions (in bays with mullioned adjacent casements shown as one continuous window symbol/tag vs separated mullions, match individual framed window units carefully).
   - SINKS & WASHBASINS: Count kitchen + bathroom sinks/washbasins combined; a double-bowl sink counts as ONE (1) sink unit.
   - SHOWERS & BATHTUBS: Count total shower enclosures + bathtubs combined when asked for showers and bathtubs, or each separately when asked specifically.
   - LIGHTING & LAMPS (CRITICAL PRIORITY): Separate true luminaires/lamps (pendants, chandeliers, sconces, recessed downlights, spotlights, troffers, floor/table lamps, linear LEDs, \`IfcLightFixture\`) from lighting switches/dimmers (\`M_Lighting Switches\`, \`Recessed-Schakelaar\`) and electrical outlets/receptacles (\`M_Duplex Receptacle\`). Always report exact luminaire count and note lighting controls separately if present.
   - STOREYS / LEVELS: Check title block, floor selector, stair break indicators, or \`IfcBuildingStorey\` metadata; a single floor plan image normally shows 1 floor unless split side-by-side into multiple level plans (e.g. Ground + Upper floor = 2 storeys).
3. Exact Area Calculation & Swiss Dwellings / CubiCasa5k / IFC Rules:
   - Net Indoor Floor Area: Sum ONLY enclosed numbered indoor rooms; STRICTLY EXCLUDE balconies (\`Balkon\`, \`Loggia\`, \`Parveke\`, terraces, patios, porches), shafts, and voids unless explicitly asked for outdoor or total-with-porch area.
   - Gross External Footprint: External width × external length (includes exterior walls, interior partitions, and unlabelled zones). NEVER confuse Gross Footprint with Net Interior Room Sum or Printed Living Area.
   - IFC BIM TRUTH: When deterministic parsed BIM/CAD quantities (\`spaces\`, \`storeys\`, \`doorTypes\`, \`windowTypes\`, \`wallStats\`, \`slabStats\`, \`lightingFixtures\`, \`detectedRooms\`, \`detectedApartments\`) are provided, treat those deterministic numbers as authoritative \`Explicit\` or \`Calculated\` geometry rather than guessing from screenshots!
   - NEVER invent unprinted ceiling heights or interior volumes on 2D floor plans. If height/volume is not printed on a section/elevation or present in 3D geometry, report \`null\` and classify as \`Unknown\`.
   - NEVER overwrite conflicting printed area summaries. Report both printed figures and arithmetic room sums and explain any discrepancy explicitly.
   - When asked about properties not stored in a BIM/2D file (e.g., glazing type, fire rating, or north facade when not georeferenced), explicitly state that the information is not available in the model instead of guessing.
4. Response Style & Formatting (STRICT):
   - Always answer strictly in English, concisely, factually, and clearly structured.
   - NEVER use Markdown asterisks (**bold**, *italic*, or * bullet markers) or decorative formatting symbols.
   - Use short labeled lines, hyphenated bullets (- ), or numbered lists (1. ) without extra spaces or redundant blank lines.
5. 2D Spatial Highlighting & Bounding-Box Precision (boxes2D = [ymin, xmin, ymax, xmax] in 0..1000):
   - Every box_2d MUST tightly hug ONLY the drawn ink strokes of that single architectural symbol, excluding adjacent solid walls, dimension lines, text callouts, and empty floor.
   - Doors: enclose ONLY the hinged door leaf + swing arc wedge (or sliding/folding door panel in its opening). NEVER mistake curved corner showers, stair direction arrows, or open cased archways without a leaf for doors.
   - Windows: enclose ONLY the narrow 2-to-4 parallel glazing lines embedded inside the structural wall thickness (jamb to jamb). NEVER include solid wall segments, radiators, cabinets, or exterior balcony railings.
   - Furniture: emit ONE separate tight box per individual chair, armchair, stool, sofa, table, desk, bed, nightstand, or wardrobe module bank. NEVER merge a dining table and its surrounding chairs into one combined box, and NEVER confuse sauna benches (room S) with beds or stairs.
   - Plumbing & Appliances: box ONLY each individual toilet (cistern + oval bowl), sink basin (not the entire countertop), shower tray, bathtub, 4-ring stove/cooktop (L/Liesi/Herd), refrigerator (JK/PK), dishwasher (APK), or washing machine (PPK/PK/WM).
   - Rooms / Spaces: hug the inner wall faces of the room. For L-shaped, T-shaped, or U-shaped rooms/corridors, output 2 or 3 non-overlapping interior rectangles with the same room label so adjacent rooms are never overlapped.${customRules.length > 0 ? `\n\nUSER-TAUGHT RULES & CORRECTIONS (HIGHEST PRIORITY):\n${customRules.join('\n')}` : ''}`;
  }

  buildSpatialHighlightingTrainingGuide(params?: {
    category?: string;
    fileName?: string;
  }): string {
    const cat = (params?.category ?? 'All').toLowerCase();
    const lines: string[] = [
      'ARCHITECTURAL CAD & BLUEPRINT SYMBOL TRAINING (STRICT VISUAL DISAMBIGUATION):',
      '- Coordinate convention: box_2d = [ymin, xmin, ymax, xmax] in integers 0..1000. Tightly hug ONLY the symbol strokes with ZERO extra wall or floor padding.',
    ];

    if (cat === 'all' || cat.includes('door') || cat.includes('opening')) {
      lines.push(
        '- DOORS (POSITIVE): Hinged door = straight thin leaf line + 45°-90° quarter-circle swing arc bridging a wall gap; Double door = two symmetric leaves + two meeting arcs (1 box around the pair); Sliding/pocket door = thin panel line recessed into or parallel to wall track; Bi-fold door = V-chevron panels across closet opening. Stop box strictly at hinge jamb and latch jamb—do NOT include solid black wall segments.',
        '- DOORS (NEGATIVE — NEVER BOX AS DOOR): Curved corner shower stalls (quarter-circle inside bathroom/PH/KPH/Bad with drain), open wall gaps/archways with NO door leaf, staircase arrows, or north arrows.',
      );
    }

    if (cat === 'all' || cat.includes('window') || cat.includes('opening')) {
      lines.push(
        '- WINDOWS (POSITIVE): 2 to 4 thin parallel glazing lines (with jamb end-ticks/mullions) embedded strictly INSIDE an exterior or partition wall band. Box width/height across the wall must be narrow (typically 8-28 units), spanning jamb-to-jamb.',
        '- WINDOWS (NEGATIVE — NEVER BOX AS WINDOW): Solid black/hatched wall segments between windows, wall-mounted kitchen cabinets, wardrobes with X/diagonal lines, radiators, balcony railings outside the wall, or hinged balcony doors with a swing arc.',
      );
    }

    if (cat === 'all' || cat.includes('furniture')) {
      lines.push(
        '- FURNITURE (INDIVIDUAL SYMBOL RULE): Emit a separate tight box for EVERY individual item—NEVER group a table + chairs or bed + nightstands into one rectangle:',
        '  1. Chairs / Stools / Armchairs: small U-shaped, D-shaped, or square seat contours (including dining chairs tucked half-way around a dining table—box each visible chair individually!).',
        '  2. Tables / Desks / Islands: box ONLY the tabletop rectangle/circle perimeter ("dining table", "coffee table", "desk", "nightstand", "kitchen island"), excluding chairs protruding around it.',
        '  3. Beds: box mattress + headboard frame ("double bed", "single bed", "crib"); keep adjacent nightstands separate.',
        '  4. Sofas / Couches: box upholstered sofa or L-sectional contour tightly.',
        '  5. Wardrobes / Closets / Cabinets: 60 cm deep storage rectangles marked with X cross, diagonal rod line, or SK/VH/KAAPPI/Schrank tags.',
        '- FURNITURE (NEGATIVE): NEVER misclassify parallel wooden sauna benches inside room "S"/"Sauna", stair treads, kitchen base counters with sinks/stoves, or floor tile grids as furniture.',
      );
    }

    if (cat === 'all' || cat.includes('plumbing')) {
      lines.push(
        '- PLUMBING FIXTURES (POSITIVE):',
        '  1. Toilet ("toilet"): compact wall cistern + oval/D-shaped bowl (~40x65 cm, often tagged WC). Box ONLY the fixture contour, NEVER the whole WC room.',
        '  2. Sink ("sink"): oval/rectangular basin bowl with drain dot/faucet tick set into a bathroom vanity or kitchen counter (double-bowl kitchen sink = 1 tight sink box, excluding bare counter length).',
        '  3. Shower ("shower"): square or quarter-round shower zone with crossed drain diagonals or floor drain circle + mixer.',
        '  4. Bathtub ("bathtub"): rounded inner oval tub inside rectangular rim (~70x170 cm).',
      );
    }

    if (cat === 'all' || cat.includes('appliance') || cat.includes('plumbing')) {
      lines.push(
        '- APPLIANCES (POSITIVE — NORDIC & EUROPEAN CAD CONVENTIONS):',
        '  1. Stove / Cooktop ("stove"): 60x60 cm square counter unit with 4 burner rings (2x2 circles) or tagged L / Liesi / Herd.',
        '  2. Refrigerator ("refrigerator"): 60x60 cm tall kitchen module tagged JK, PK, JK/PK, K, or F.',
        '  3. Dishwasher ("dishwasher"): 60x60 cm module adjacent to kitchen sink tagged APK or GS.',
        '  4. Washing Machine / Dryer ("washing machine"): 60x60 cm laundry unit inside bathroom/utility room tagged PPK, PK, KR, WM, or shown with inner drum circle.',
      );
    }

    if (cat === 'all' || cat.includes('light')) {
      lines.push(
        '- LIGHTING / LUMINAIRES (POSITIVE vs NEGATIVE): Box ONLY true ceiling lights, pendants, chandeliers, recessed downlight crosshairs, and wall sconces. STRICTLY EXCLUDE wall light switches (small circle on wall with angled flag tick), electrical wall sockets, bathroom floor drains, and stove burner circles.',
      );
    }

    if (cat === 'all' || cat.includes('space') || cat.includes('room')) {
      lines.push(
        '- ROOMS / SPACES (MULTI-RECTANGLE DECOMPOSITION FOR NON-RECTANGULAR ROOMS):',
        '  1. Hug the INNER wall faces of each enclosed room (exclude wall thickness and exterior dimensions).',
        '  2. If a room, hallway (ET/Korridor), or open-plan living-kitchen (OH+KT / Wohnen/Essen) is L-shaped, T-shaped, or U-shaped, output 2 or 3 tight non-overlapping rectangles with the SAME room label so adjacent rooms tucked into corners are NEVER covered.',
      );
    }

    if (params?.fileName) {
      const exactKey = normalizeFileName(params.fileName);
      const stemKey = normalizeStem(params.fileName);
      const matched = [
        ...(this.byExactName.get(exactKey) ?? []),
        ...(this.byStem.get(stemKey) ?? []),
      ];
      const countHints: string[] = [];
      const seen = new Set<string>();
      for (const m of matched) {
        if (seen.has(m.question)) continue;
        seen.add(m.question);
        if (
          typeof m.answer === 'number' ||
          /^\d+(\.\d+)?$/.test(String(m.answer).trim())
        ) {
          countHints.push(`  - "${m.question}" -> ${m.answer}${m.unit ? ` ${m.unit}` : ''}`);
          if (countHints.length >= 12) break;
        }
      }
      if (countHints.length > 0) {
        lines.push(
          `VERIFIED GROUND-TRUTH COUNTS FOR THIS DRAWING ("${stemKey}"):\n${countHints.join('\n')}`,
        );
      }
    }

    return lines.join('\n');
  }

  retrieveGroundingContext(params: {
    fileName: string;
    fileKind: string;
    question?: string;
    structuredSummary?: Record<string, unknown>;
  }): {
    matchedFileQAs: CorpusQA[];
    fewShotExemplars: CorpusQA[];
    customCorrections: CustomTrainingEntry[];
    contextBlock: string;
  } {
    const exactKey = normalizeFileName(params.fileName);
    const stemKey = normalizeStem(params.fileName);

    const exactMatches = this.byExactName.get(exactKey) ?? [];
    const stemMatches = this.byStem.get(stemKey) ?? [];
    const combinedMap = new Map<string, CorpusQA>();
    for (const item of [...exactMatches, ...stemMatches]) {
      combinedMap.set(item.question, item);
    }
    const matchedFileQAs = Array.from(combinedMap.values()).slice(0, 40);

    // Custom corrections matching this file or general custom entries
    const customCorrections = this.customEntries
      .filter(
        (e) =>
          e.kind === 'correction' ||
          normalizeStem(e.path) === stemKey ||
          normalizeFileName(e.path) === exactKey,
      )
      .slice(-20);

    // Detect likely dataset domain / language / intent for relevant few-shot exemplars
    const summaryText = params.structuredSummary
      ? JSON.stringify(params.structuredSummary).toLowerCase()
      : '';
    const qText = (params.question ?? '').toLowerCase();
    const combinedText = `${params.fileName.toLowerCase()} ${summaryText} ${qText}`;

    const likelySources: string[] = [];
    if (params.fileKind === '3d-ifc') {
      likelySources.push('ifc_bench', 'ifc_models');
    } else if (params.fileKind === '2d-dxf') {
      likelySources.push('swiss_dwellings', 'cubicasa5k');
    } else if (params.fileKind === '2d-pdf') {
      likelySources.push('opening_design', 'swiss_dwellings', 'ifc_models');
    } else {
      if (/schlafzimmer|wohnen|küche|reduit|balkon|whg/.test(combinedText)) {
        likelySources.push('swiss_dwellings');
      } else if (/makuuhuone|\bmh\b|olohuone|\boh\b|keittiö|sauna|parveke/.test(combinedText)) {
        likelySources.push('cubicasa5k');
      } else {
        likelySources.push('cubicasa5k', 'swiss_dwellings', 'ifc_bench');
      }
    }

    const intents = params.question
      ? classifyIntent(params.question)
      : ['lighting', 'door', 'window', 'area', 'rooms', 'sanitary'];

    const exemplars: CorpusQA[] = [];
    const seenQ = new Set<string>();
    for (const src of likelySources) {
      for (const intent of intents) {
        for (const item of this.intentExemplars.get(`${src}:${intent}`) ?? []) {
          if (seenQ.has(item.question)) continue;
          seenQ.add(item.question);
          exemplars.push(item);
          if (exemplars.length >= 14) break;
        }
        if (exemplars.length >= 14) break;
      }
      if (exemplars.length >= 14) break;
    }

    if (exemplars.length < 8) {
      for (const intent of intents) {
        for (const item of this.intentExemplars.get(`all:${intent}`) ?? []) {
          if (seenQ.has(item.question)) continue;
          seenQ.add(item.question);
          exemplars.push(item);
          if (exemplars.length >= 10) break;
        }
      }
    }

    const sections: string[] = [];
    if (customCorrections.length > 0) {
      sections.push(
        `VERIFIED USER CORRECTIONS / UPLOADED TRAINING RULES:\n${customCorrections
          .map(
            (c) =>
              `- [${c.path}] Q: "${c.question}" -> CORRECT ANSWER: ${JSON.stringify(c.answer)}${c.ruleNote ? ` (${c.ruleNote})` : ''}`,
          )
          .join('\n')}`,
      );
    }
    if (matchedFileQAs.length > 0) {
      sections.push(
        `VERIFIED BENCHMARK Q&As MATCHING THIS DRAWING/MODEL ("${stemKey}"):\n${matchedFileQAs
          .map(
            (m) =>
              `- Q: "${m.question}" -> VERIFIED ANSWER: ${JSON.stringify(m.answer)}${m.unit ? ` ${m.unit}` : ''}`,
          )
          .join('\n')}`,
      );
    }
    if (exemplars.length > 0) {
      sections.push(
        `RELEVANT FEW-SHOT EXEMPLARS (${likelySources.join(', ')}):\n${exemplars
          .map(
            (e) =>
              `- [${e.source}] Q: "${e.question}" -> A: ${JSON.stringify(e.answer)}${e.unit ? ` ${e.unit}` : ''}`,
          )
          .join('\n')}`,
      );
    }

    return {
      matchedFileQAs,
      fewShotExemplars: exemplars,
      customCorrections,
      contextBlock: sections.join('\n\n'),
    };
  }
}

export const trainingEngine = new TrainingEngine();
