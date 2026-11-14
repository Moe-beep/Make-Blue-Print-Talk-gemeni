import * as THREE from 'three';
import type {
  ParsedIfcSpace,
  ParsedIfcStoreyBreakdown,
  ParsedIfcSummary,
} from '../types/analysis';

interface StepEntity {
  id: number;
  type: string;
  rawArgs: string;
  args: unknown[];
}

function splitStepArgs(raw: string): string[] {
  const result: string[] = [];
  let depth = 0;
  let inString = false;
  let current = '';

  for (let i = 0; i < raw.length; i++) {
    const ch = raw[i];
    if (ch === "'") {
      if (inString && raw[i + 1] === "'") {
        current += "''";
        i++;
        continue;
      }
      inString = !inString;
      current += ch;
    } else if (!inString) {
      if (ch === '(') {
        depth++;
        current += ch;
      } else if (ch === ')') {
        depth--;
        current += ch;
      } else if (ch === ',' && depth === 0) {
        result.push(current.trim());
        current = '';
      } else {
        current += ch;
      }
    } else {
      current += ch;
    }
  }
  if (current.trim().length > 0) {
    result.push(current.trim());
  }
  return result;
}

function decodeIfcString(str: string): string {
  return str.replace(/\\X2\\([0-9A-Fa-f]+)\\X0\\/g, (_, hex: string) => {
    let out = '';
    for (let i = 0; i + 3 < hex.length; i += 4) {
      out += String.fromCharCode(parseInt(hex.slice(i, i + 4), 16));
    }
    return out;
  });
}

function parseStepValue(token: string): unknown {
  if (!token || token === '$' || token === '*') return null;
  if (token.startsWith("'") && token.endsWith("'")) {
    return decodeIfcString(token.slice(1, -1).replace(/''/g, "'"));
  }
  if (token.startsWith('#')) {
    return { ref: parseInt(token.slice(1), 10) };
  }
  // Typed wrapper e.g. IFCAREAMEASURE(27.66) or IFCIDENTIFIER('A101')
  const typedMatch = token.match(/^[A-Z0-9_]+\(([\s\S]*)\)$/i);
  if (typedMatch && !token.startsWith('(')) {
    return parseStepValue(typedMatch[1].trim());
  }
  if (token.startsWith('(') && token.endsWith(')')) {
    const inner = token.slice(1, -1).trim();
    if (!inner) return [];
    return splitStepArgs(inner).map(parseStepValue);
  }
  if (token.startsWith('.') && token.endsWith('.')) {
    return token.slice(1, -1);
  }
  const num = Number(token);
  if (!Number.isNaN(num)) return num;
  return token;
}

function getRef(val: unknown): number | null {
  if (val && typeof val === 'object' && 'ref' in val) {
    return (val as { ref: number }).ref;
  }
  return null;
}

function getRefArray(val: unknown): number[] {
  if (!Array.isArray(val)) return [];
  const refs: number[] = [];
  for (const item of val) {
    const r = getRef(item);
    if (r !== null) refs.push(r);
  }
  return refs;
}

function getStr(val: unknown): string {
  return typeof val === 'string' ? val : '';
}

function getNum(val: unknown, fallback = 0): number {
  return typeof val === 'number' && Number.isFinite(val) ? val : fallback;
}

export interface ParsedIfcResult {
  summary: ParsedIfcSummary;
  group: THREE.Group;
}

// Separate true light fixtures (luminaires) from switches/dimmers/receptacles
const CONTROL_OR_RECEPTACLE_REGEX =
  /switch|dimmer|sensor|thermostat|panelboard|receptacle|outlet|breaker|telephone|board|socket|schakelaar/i;
const LUMINAIRE_REGEX =
  /lamp|luminaire|sconce|pendant|chandelier|downlight|spotlight|troffer|lantern|light\s*fixture|ceiling\s*light|wall\s*light|track\s*light|fluorescent|halogen|armatuur|\bled\b|light\s*-|fixture/i;
const SANITARY_OR_APPLIANCE_REGEX =
  /toilet|wc|sink|basin|lavatory|shower|bathtub|bath|urinal|range|stove|oven|fridge|refrigerator|microwave|dishwasher|washer|dryer|water\s*heater/i;

const CATEGORY_COLORS: Record<
  string,
  { color: number; opacity?: number; emissive?: number }
> = {
  LIGHTING: { color: 0xf59e0b, emissive: 0x78350f },
  IFCWALL: { color: 0xe2e8f0 },
  IFCWALLSTANDARDCASE: { color: 0xe2e8f0 },
  IFCCURTAINWALL: { color: 0xbae6fd, opacity: 0.45 },
  IFCSLAB: { color: 0xcbd5e1 },
  IFCROOF: { color: 0x94a3b8 },
  IFCDOOR: { color: 0x3b82f6 },
  IFCWINDOW: { color: 0x38bdf8, opacity: 0.42 },
  IFCFURNISHINGELEMENT: { color: 0xd97706 },
  IFCFLOWTERMINAL: { color: 0x0d9488 },
  IFCSANITARYTERMINAL: { color: 0x0ea5e9 },
  IFCSTAIR: { color: 0x64748b },
  IFCSTAIRFLIGHT: { color: 0x64748b },
  IFCRAMP: { color: 0x64748b },
  IFCCOLUMN: { color: 0x475569 },
  IFCBEAM: { color: 0x475569 },
  IFCRAILING: { color: 0x334155 },
  IFCSPACE: { color: 0x93c5fd, opacity: 0.12 },
  DEFAULT: { color: 0x94a3b8 },
};

const SI_PREFIX_MULT: Record<string, number> = {
  MILLI: 0.001,
  CENTI: 0.01,
  DECI: 0.1,
  '': 1,
  KILO: 1000,
};

export function parseIfcFile(text: string): ParsedIfcResult {
  const cleaned = text.replace(/\/\*[\s\S]*?\*\//g, '');

  const schemaMatch = cleaned.match(/FILE_SCHEMA\s*\(\s*\(\s*'([^']+)'/i);
  const schema = schemaMatch?.[1] ?? 'IFC';

  const appMatch = cleaned.match(/IFCAPPLICATION\([^,]+,'[^']*','([^']+)'/i);
  const authoringTool = appMatch?.[1] ?? 'Unknown BIM Tool';

  const dataStart = cleaned.indexOf('DATA;');
  const dataEnd = cleaned.lastIndexOf('ENDSEC;');
  const dataBody =
    dataStart !== -1
      ? cleaned.slice(dataStart + 5, dataEnd > dataStart ? dataEnd : undefined)
      : cleaned;

  const entities = new Map<number, StepEntity>();
  const statements = dataBody.split(';');
  const stmtRegex = /^\s*#(\d+)\s*=\s*([A-Z0-9_]+)\s*\(([\s\S]*)\)\s*$/;

  for (const stmt of statements) {
    const m = stmt.match(stmtRegex);
    if (!m) continue;
    const id = parseInt(m[1], 10);
    const type = m[2].toUpperCase();
    const rawArgs = m[3];
    entities.set(id, { id, type, rawArgs, args: [] });
  }

  const getArgs = (ent: StepEntity): unknown[] => {
    if (ent.args.length === 0 && ent.rawArgs.trim().length > 0) {
      ent.args = splitStepArgs(ent.rawArgs).map(parseStepValue);
    }
    return ent.args;
  };

  // Detect length, area, volume units accurately
  let lengthToMeters = 1;
  let lengthUnit = 'm';
  let areaToSqM = 1;
  let volumeToCuM = 1;
  let areaUnitExplicit = false;
  let volumeUnitExplicit = false;
  let projectName = 'Untitled BIM Project';

  for (const ent of entities.values()) {
    if (ent.type === 'IFCPROJECT') {
      const a = getArgs(ent);
      projectName = getStr(a[2]) || getStr(a[7]) || projectName;
    } else if (ent.type === 'IFCSIUNIT') {
      const a = getArgs(ent);
      const unitType = getStr(a[1]);
      const prefix = getStr(a[2]);
      const name = getStr(a[3]);
      const mult = SI_PREFIX_MULT[prefix] ?? 1;
      if (unitType === 'LENGTHUNIT' && name === 'METRE') {
        lengthToMeters = mult;
        lengthUnit =
          prefix === 'MILLI'
            ? 'mm'
            : prefix === 'CENTI'
              ? 'cm'
              : prefix === 'DECI'
                ? 'dm'
                : 'm';
      } else if (unitType === 'AREAUNIT' && name === 'SQUARE_METRE') {
        areaToSqM = mult * mult;
        areaUnitExplicit = true;
      } else if (unitType === 'VOLUMEUNIT' && name === 'CUBIC_METRE') {
        volumeToCuM = mult * mult * mult;
        volumeUnitExplicit = true;
      }
    } else if (ent.type === 'IFCCONVERSIONBASEDUNIT') {
      const a = getArgs(ent);
      const unitType = getStr(a[1]);
      const name = getStr(a[2]).toUpperCase();
      if (
        unitType === 'LENGTHUNIT' &&
        (name.includes('FOOT') || name.includes('FEET'))
      ) {
        lengthToMeters = 0.3048;
        lengthUnit = 'ft';
      } else if (unitType === 'LENGTHUNIT' && name.includes('INCH')) {
        lengthToMeters = 0.0254;
        lengthUnit = 'in';
      } else if (
        unitType === 'AREAUNIT' &&
        (name.includes('FOOT') || name.includes('FEET'))
      ) {
        areaToSqM = 0.09290304;
        areaUnitExplicit = true;
      } else if (
        unitType === 'VOLUMEUNIT' &&
        (name.includes('FOOT') || name.includes('FEET'))
      ) {
        volumeToCuM = 0.0283168466;
        volumeUnitExplicit = true;
      }
    }
  }

  if (!areaUnitExplicit && lengthUnit === 'ft') {
    areaToSqM = 0.09290304;
  }
  if (!volumeUnitExplicit && lengthUnit === 'ft') {
    volumeToCuM = 0.0283168466;
  }

  // 1. Spatial hierarchy: map BuildingStorey -> contained/aggregated element IDs
  const storeyInfoMap = new Map<number, { name: string; elevationM: number }>();
  for (const ent of entities.values()) {
    if (ent.type === 'IFCBUILDINGSTOREY') {
      const a = getArgs(ent);
      const name = getStr(a[2]) || getStr(a[7]) || `Storey #${ent.id}`;
      const elev = getNum(a[9], 0) * lengthToMeters;
      storeyInfoMap.set(ent.id, {
        name,
        elevationM: Number(elev.toFixed(3)),
      });
    }
  }

  const elementToStorey = new Map<number, string>();
  const aggChildren = new Map<number, number[]>();
  for (const ent of entities.values()) {
    if (ent.type === 'IFCRELCONTAINEDINSPATIALSTRUCTURE') {
      const a = getArgs(ent);
      const elems = getRefArray(a[4]);
      const structureRef = getRef(a[5]);
      if (structureRef !== null && storeyInfoMap.has(structureRef)) {
        const sName = storeyInfoMap.get(structureRef)!.name;
        for (const eid of elems) {
          elementToStorey.set(eid, sName);
        }
      }
    } else if (ent.type === 'IFCRELAGGREGATES') {
      const a = getArgs(ent);
      const parentRef = getRef(a[4]);
      const children = getRefArray(a[5]);
      if (parentRef !== null) {
        aggChildren.set(parentRef, [
          ...(aggChildren.get(parentRef) ?? []),
          ...children,
        ]);
      }
    }
  }
  // Propagate storey membership through IFCRELAGGREGATES (e.g. Storey -> Space, Stair -> StairFlight)
  const assignStoreyTree = (nodeId: number, storeyName: string, depth = 0) => {
    if (depth > 8) return;
    if (!elementToStorey.has(nodeId)) {
      elementToStorey.set(nodeId, storeyName);
    }
    for (const childId of aggChildren.get(nodeId) ?? []) {
      assignStoreyTree(childId, storeyName, depth + 1);
    }
  };
  for (const [sid, info] of storeyInfoMap.entries()) {
    for (const childId of aggChildren.get(sid) ?? []) {
      assignStoreyTree(childId, info.name, 0);
    }
  }
  for (const [eid, sName] of Array.from(elementToStorey.entries())) {
    for (const childId of aggChildren.get(eid) ?? []) {
      assignStoreyTree(childId, sName, 0);
    }
  }

  // 1b. Space boundary links (IFCRELSPACEBOUNDARY) so doors/windows know adjacent rooms
  const elemToSpaceIds = new Map<number, Set<number>>();
  for (const ent of entities.values()) {
    if (ent.type !== 'IFCRELSPACEBOUNDARY') continue;
    const a = getArgs(ent);
    const spaceRef = getRef(a[4]);
    const elemRef = getRef(a[5]);
    if (spaceRef !== null && elemRef !== null) {
      let set = elemToSpaceIds.get(elemRef);
      if (!set) {
        set = new Set<number>();
        elemToSpaceIds.set(elemRef, set);
      }
      set.add(spaceRef);
    }
  }
  const spaceLabelMap = new Map<number, string>();
  for (const ent of entities.values()) {
    if (ent.type === 'IFCSPACE') {
      const sa = getArgs(ent);
      const code = getStr(sa[2]);
      const longN = getStr(sa[7]);
      spaceLabelMap.set(ent.id, longN || code || `Space #${ent.id}`);
    }
  }

  // 2. Type mapping (IFCRELDEFINESBYTYPE) for Door/Window/Light/Furnishing type names
  const elementTypeName = new Map<number, string>();
  for (const ent of entities.values()) {
    if (ent.type !== 'IFCRELDEFINESBYTYPE') continue;
    const a = getArgs(ent);
    const related = getRefArray(a[4]);
    const typeRef = getRef(a[5]);
    if (typeRef === null) continue;
    const typeEnt = entities.get(typeRef);
    if (!typeEnt) continue;
    const ta = getArgs(typeEnt);
    const tName = getStr(ta[2]) || getStr(ta[6]) || '';
    if (tName) {
      for (const eid of related) {
        elementTypeName.set(eid, tName);
      }
    }
  }

  // 3. Extract IFCELEMENTQUANTITY & IFCPROPERTYSET quantities per element
  interface ElemQuantities {
    areaSqM: number | null;
    areaPriority: number;
    netSideAreaSqM: number | null;
    volumeCuM: number | null;
    lengthM: number | null;
    heightM: number | null;
    widthM: number | null;
  }
  const elemQuantities = new Map<number, ElemQuantities>();
  const ensureQ = (id: number): ElemQuantities => {
    let q = elemQuantities.get(id);
    if (!q) {
      q = {
        areaSqM: null,
        areaPriority: 0,
        netSideAreaSqM: null,
        volumeCuM: null,
        lengthM: null,
        heightM: null,
        widthM: null,
      };
      elemQuantities.set(id, q);
    }
    return q;
  };

  for (const ent of entities.values()) {
    if (ent.type !== 'IFCRELDEFINESBYPROPERTIES') continue;
    const a = getArgs(ent);
    const relatedObjects = getRefArray(a[4]);
    const propSetRef = getRef(a[5]);
    if (propSetRef === null) continue;
    const propSet = entities.get(propSetRef);
    if (!propSet) continue;

    if (propSet.type === 'IFCELEMENTQUANTITY') {
      const pa = getArgs(propSet);
      const qRefs = getRefArray(pa[5]);
      for (const qr of qRefs) {
        const qEnt = entities.get(qr);
        if (!qEnt) continue;
        const qa = getArgs(qEnt);
        const qName = getStr(qa[0]).toLowerCase();
        const val = getNum(qa[3], NaN);
        if (!Number.isFinite(val)) continue;

        if (qEnt.type === 'IFCQUANTITYAREA') {
          const sqM = val * areaToSqM;
          if (qName.includes('side')) {
            for (const oid of relatedObjects) ensureQ(oid).netSideAreaSqM = sqM;
          } else {
            const priority = qName.includes('net')
              ? 3
              : qName.includes('gsa') || qName.includes('floor')
                ? 2
                : 1;
            for (const oid of relatedObjects) {
              const q = ensureQ(oid);
              if (q.areaSqM === null || priority >= q.areaPriority) {
                q.areaSqM = sqM;
                q.areaPriority = priority;
              }
            }
          }
        } else if (qEnt.type === 'IFCQUANTITYVOLUME') {
          const cuM = val * volumeToCuM;
          for (const oid of relatedObjects) {
            const q = ensureQ(oid);
            if (q.volumeCuM === null || qName.includes('net')) {
              q.volumeCuM = cuM;
            }
          }
        } else if (qEnt.type === 'IFCQUANTITYLENGTH') {
          const mVal = val * lengthToMeters;
          for (const oid of relatedObjects) {
            const q = ensureQ(oid);
            if (qName.includes('height')) q.heightM = mVal;
            else if (qName.includes('width')) q.widthM = mVal;
            else if (qName.includes('length')) q.lengthM = mVal;
          }
        }
      }
    } else if (propSet.type === 'IFCPROPERTYSET') {
      const pa = getArgs(propSet);
      const pRefs = getRefArray(pa[4]);
      for (const pr of pRefs) {
        const pEnt = entities.get(pr);
        if (!pEnt || pEnt.type !== 'IFCPROPERTYSINGLEVALUE') continue;
        const pArgs = getArgs(pEnt);
        const pName = getStr(pArgs[0]).toLowerCase();
        const val = getNum(pArgs[2], NaN);
        if (!Number.isFinite(val) || val <= 0) continue;

        if (
          pName === 'area' ||
          pName === 'netfloorarea' ||
          pName === 'net floor area' ||
          pName === 'gsa bim area' ||
          pName === 'grossfloorarea'
        ) {
          const sqM = val * areaToSqM;
          const prio = pName.includes('net')
            ? 3
            : pName.includes('gsa')
              ? 2
              : 1;
          for (const oid of relatedObjects) {
            const q = ensureQ(oid);
            if (q.areaSqM === null || prio >= q.areaPriority) {
              q.areaSqM = sqM;
              q.areaPriority = prio;
            }
          }
        } else if (pName === 'volume' || pName === 'netvolume') {
          const cuM = val * volumeToCuM;
          for (const oid of relatedObjects) {
            const q = ensureQ(oid);
            if (q.volumeCuM === null) q.volumeCuM = cuM;
          }
        } else if (
          pName === 'unconnected height' ||
          pName === 'ceilingheight' ||
          pName === 'clear height' ||
          pName === 'limit offset'
        ) {
          const mVal = val * lengthToMeters;
          for (const oid of relatedObjects) {
            const q = ensureQ(oid);
            if (q.heightM === null) q.heightM = mVal;
          }
        } else if (pName === 'length') {
          const mVal = val * lengthToMeters;
          for (const oid of relatedObjects) {
            const q = ensureQ(oid);
            if (q.lengthM === null) q.lengthM = mVal;
          }
        }
      }
    }
  }

  // Geometry helpers
  const pointCache = new Map<number, THREE.Vector3>();
  const getPoint = (id: number): THREE.Vector3 => {
    const cached = pointCache.get(id);
    if (cached) return cached;
    const ent = entities.get(id);
    if (!ent || ent.type !== 'IFCCARTESIANPOINT') return new THREE.Vector3();
    const a = getArgs(ent);
    const coords = Array.isArray(a[0]) ? a[0] : [];
    const x = getNum(coords[0]) * lengthToMeters;
    const y = getNum(coords[1]) * lengthToMeters;
    const z = getNum(coords[2]) * lengthToMeters;
    const v = new THREE.Vector3(x, y, z);
    pointCache.set(id, v);
    return v;
  };

  const getDirection = (
    id: number | null,
    fallback: THREE.Vector3,
  ): THREE.Vector3 => {
    if (id === null) return fallback.clone();
    const ent = entities.get(id);
    if (!ent || ent.type !== 'IFCDIRECTION') return fallback.clone();
    const a = getArgs(ent);
    const ratios = Array.isArray(a[0]) ? a[0] : [];
    const v = new THREE.Vector3(
      getNum(ratios[0], fallback.x),
      getNum(ratios[1], fallback.y),
      getNum(ratios[2], fallback.z),
    );
    return v.lengthSq() > 0 ? v.normalize() : fallback.clone();
  };

  const getAxis2Placement3DMatrix = (id: number | null): THREE.Matrix4 => {
    const mat = new THREE.Matrix4();
    if (id === null) return mat;
    const ent = entities.get(id);
    if (!ent) return mat;
    const a = getArgs(ent);
    if (ent.type === 'IFCAXIS2PLACEMENT3D') {
      const locRef = getRef(a[0]);
      const loc = locRef !== null ? getPoint(locRef) : new THREE.Vector3();
      const zAxis = getDirection(getRef(a[1]), new THREE.Vector3(0, 0, 1));
      let xAxis = getDirection(getRef(a[2]), new THREE.Vector3(1, 0, 0));
      xAxis = xAxis
        .clone()
        .sub(zAxis.clone().multiplyScalar(xAxis.dot(zAxis)));
      if (xAxis.lengthSq() < 1e-8) {
        xAxis =
          Math.abs(zAxis.x) < 0.9
            ? new THREE.Vector3(1, 0, 0).cross(zAxis).normalize()
            : new THREE.Vector3(0, 1, 0).cross(zAxis).normalize();
      } else {
        xAxis.normalize();
      }
      const yAxis = new THREE.Vector3().crossVectors(zAxis, xAxis).normalize();
      mat.makeBasis(xAxis, yAxis, zAxis);
      mat.setPosition(loc);
    } else if (ent.type === 'IFCAXIS2PLACEMENT2D') {
      const locRef = getRef(a[0]);
      const loc = locRef !== null ? getPoint(locRef) : new THREE.Vector3();
      const xAxis2 = getDirection(getRef(a[1]), new THREE.Vector3(1, 0, 0));
      const xAxis = new THREE.Vector3(xAxis2.x, xAxis2.y, 0).normalize();
      const zAxis = new THREE.Vector3(0, 0, 1);
      const yAxis = new THREE.Vector3().crossVectors(zAxis, xAxis).normalize();
      mat.makeBasis(xAxis, yAxis, zAxis);
      mat.setPosition(loc.x, loc.y, 0);
    }
    return mat;
  };

  const placementCache = new Map<number, THREE.Matrix4>();
  const getLocalPlacementMatrix = (
    id: number | null,
    depth = 0,
  ): THREE.Matrix4 => {
    if (id === null || depth > 25) return new THREE.Matrix4();
    const cached = placementCache.get(id);
    if (cached) return cached;
    const ent = entities.get(id);
    if (!ent || ent.type !== 'IFCLOCALPLACEMENT') return new THREE.Matrix4();
    const a = getArgs(ent);
    const parentMat = getLocalPlacementMatrix(getRef(a[0]), depth + 1);
    const relMat = getAxis2Placement3DMatrix(getRef(a[1]));
    const worldMat = new THREE.Matrix4().multiplyMatrices(parentMat, relMat);
    placementCache.set(id, worldMat);
    return worldMat;
  };

  const computeProfileAreaSqM = (profileId: number | null): number | null => {
    if (profileId === null) return null;
    const ent = entities.get(profileId);
    if (!ent) return null;
    const a = getArgs(ent);
    if (ent.type === 'IFCRECTANGLEPROFILEDEF') {
      const xDim = getNum(a[3]) * lengthToMeters;
      const yDim = getNum(a[4]) * lengthToMeters;
      return xDim > 0 && yDim > 0 ? xDim * yDim : null;
    }
    if (ent.type === 'IFCCIRCLEPROFILEDEF') {
      const r = getNum(a[3]) * lengthToMeters;
      return r > 0 ? Math.PI * r * r : null;
    }
    if (ent.type === 'IFCARBITRARYCLOSEDPROFILEDEF') {
      const curveRef = getRef(a[2]);
      if (curveRef === null) return null;
      const curve = entities.get(curveRef);
      if (!curve || curve.type !== 'IFCPOLYLINE') return null;
      const ptRefs = getRefArray(getArgs(curve)[0]);
      if (ptRefs.length < 3) return null;
      const pts = ptRefs.map(getPoint);
      let area2 = 0;
      for (let i = 0; i < pts.length; i++) {
        const j = (i + 1) % pts.length;
        area2 += pts[i].x * pts[j].y - pts[j].x * pts[i].y;
      }
      const val = Math.abs(area2) / 2;
      return val > 0.001 ? val : null;
    }
    return null;
  };

  const buildProfileShape = (profileId: number | null): THREE.Shape | null => {
    if (profileId === null) return null;
    const ent = entities.get(profileId);
    if (!ent) return null;
    const a = getArgs(ent);

    if (ent.type === 'IFCRECTANGLEPROFILEDEF') {
      const posMat = getAxis2Placement3DMatrix(getRef(a[2]));
      const xDim = getNum(a[3]) * lengthToMeters;
      const yDim = getNum(a[4]) * lengthToMeters;
      if (xDim <= 0 || yDim <= 0) return null;
      const hw = xDim / 2;
      const hh = yDim / 2;
      const corners = [
        new THREE.Vector3(-hw, -hh, 0),
        new THREE.Vector3(hw, -hh, 0),
        new THREE.Vector3(hw, hh, 0),
        new THREE.Vector3(-hw, hh, 0),
      ].map((p) => p.applyMatrix4(posMat));
      const shape = new THREE.Shape();
      shape.moveTo(corners[0].x, corners[0].y);
      for (let i = 1; i < corners.length; i++)
        shape.lineTo(corners[i].x, corners[i].y);
      shape.closePath();
      return shape;
    }

    if (ent.type === 'IFCCIRCLEPROFILEDEF') {
      const posMat = getAxis2Placement3DMatrix(getRef(a[2]));
      const radius = getNum(a[3]) * lengthToMeters;
      if (radius <= 0) return null;
      const center = new THREE.Vector3().applyMatrix4(posMat);
      const shape = new THREE.Shape();
      shape.absarc(center.x, center.y, radius, 0, Math.PI * 2, false);
      return shape;
    }

    if (ent.type === 'IFCARBITRARYCLOSEDPROFILEDEF') {
      const curveRef = getRef(a[2]);
      if (curveRef === null) return null;
      const curve = entities.get(curveRef);
      if (!curve) return null;
      const ca = getArgs(curve);
      if (curve.type === 'IFCPOLYLINE') {
        const ptRefs = getRefArray(ca[0]);
        if (ptRefs.length < 3) return null;
        const pts = ptRefs.map(getPoint);
        const shape = new THREE.Shape();
        shape.moveTo(pts[0].x, pts[0].y);
        for (let i = 1; i < pts.length; i++) shape.lineTo(pts[i].x, pts[i].y);
        shape.closePath();
        return shape;
      }
    }

    return null;
  };

  const buildRepresentationGeometries = (
    repItemId: number,
    out: Array<{ geometry: THREE.BufferGeometry; matrix: THREE.Matrix4 }>,
    parentMat: THREE.Matrix4,
    depth = 0,
  ) => {
    if (depth > 8) return;
    const ent = entities.get(repItemId);
    if (!ent) return;
    const a = getArgs(ent);

    if (ent.type === 'IFCEXTRUDEDAREASOLID') {
      const shape = buildProfileShape(getRef(a[0]));
      if (!shape) return;
      const posMat = getAxis2Placement3DMatrix(getRef(a[1]));
      const dir = getDirection(getRef(a[2]), new THREE.Vector3(0, 0, 1));
      const depthVal = getNum(a[3]) * lengthToMeters;
      if (depthVal <= 0) return;
      try {
        const geo = new THREE.ExtrudeGeometry(shape, {
          depth: depthVal,
          bevelEnabled: false,
          steps: 1,
        });
        if (dir.z < -0.5) {
          geo.translate(0, 0, -depthVal);
        }
        const combined = new THREE.Matrix4().multiplyMatrices(parentMat, posMat);
        out.push({ geometry: geo, matrix: combined });
      } catch {
        // Ignore degenerate profile extrusions
      }
      return;
    }

    if (
      ent.type === 'IFCFACETEDBREP' ||
      ent.type === 'IFCFACETEDBREPWITHVOIDS'
    ) {
      const shellRef = getRef(a[0]);
      if (shellRef === null) return;
      const shell = entities.get(shellRef);
      if (!shell) return;
      const sa = getArgs(shell);
      const faceRefs = getRefArray(sa[0]);
      const positions: number[] = [];

      for (const fr of faceRefs) {
        const face = entities.get(fr);
        if (!face) continue;
        const fa = getArgs(face);
        const boundRefs = getRefArray(fa[0]);
        for (const br of boundRefs) {
          const bound = entities.get(br);
          if (!bound) continue;
          const ba = getArgs(bound);
          if (bound.type !== 'IFCFACEOUTERBOUND') continue;
          const loopRef = getRef(ba[0]);
          if (loopRef === null) continue;
          const loop = entities.get(loopRef);
          if (!loop || loop.type !== 'IFCPOLYLOOP') continue;
          const la = getArgs(loop);
          const ptRefs = getRefArray(la[0]);
          if (ptRefs.length < 3) continue;
          const pts = ptRefs.map(getPoint);
          for (let i = 1; i + 1 < pts.length; i++) {
            const p0 = pts[0];
            const p1 = pts[i];
            const p2 = pts[i + 1];
            positions.push(
              p0.x,
              p0.y,
              p0.z,
              p1.x,
              p1.y,
              p1.z,
              p2.x,
              p2.y,
              p2.z,
            );
          }
        }
      }

      if (positions.length >= 9) {
        const geo = new THREE.BufferGeometry();
        geo.setAttribute(
          'position',
          new THREE.Float32BufferAttribute(positions, 3),
        );
        geo.computeVertexNormals();
        out.push({ geometry: geo, matrix: parentMat.clone() });
      }
      return;
    }

    if (
      ent.type === 'IFCBOOLEANCLIPPINGRESULT' ||
      ent.type === 'IFCBOOLEANRESULT'
    ) {
      const firstOp = getRef(a[1]);
      if (firstOp !== null) {
        buildRepresentationGeometries(firstOp, out, parentMat, depth + 1);
      }
      return;
    }

    if (ent.type === 'IFCMAPPEDITEM') {
      const mapRef = getRef(a[0]);
      if (mapRef === null) return;
      const mapEnt = entities.get(mapRef);
      if (!mapEnt) return;
      const ma = getArgs(mapEnt);
      const mapOriginMat = getAxis2Placement3DMatrix(getRef(ma[0]));
      const mappedRepRef = getRef(ma[1]);
      if (mappedRepRef === null) return;
      const mappedRep = entities.get(mappedRepRef);
      if (!mappedRep) return;
      const mra = getArgs(mappedRep);
      const itemRefs = getRefArray(mra[3]);
      const nextMat = new THREE.Matrix4().multiplyMatrices(
        parentMat,
        mapOriginMat,
      );
      for (const ir of itemRefs) {
        buildRepresentationGeometries(ir, out, nextMat, depth + 1);
      }
    }
  };

  const zUpToYUp = new THREE.Matrix4().makeRotationX(-Math.PI / 2);

  const storeyAgg = new Map<
    string,
    {
      name: string;
      elevationM: number;
      spaceCount: number;
      netFloorAreaSqM: number;
      hasArea: boolean;
      doorCount: number;
      windowCount: number;
      wallCount: number;
      stairCount: number;
      lightFixtureCount: number;
      sanitaryCount: number;
    }
  >();
  for (const sInfo of storeyInfoMap.values()) {
    storeyAgg.set(sInfo.name, {
      name: sInfo.name,
      elevationM: sInfo.elevationM,
      spaceCount: 0,
      netFloorAreaSqM: 0,
      hasArea: false,
      doorCount: 0,
      windowCount: 0,
      wallCount: 0,
      stairCount: 0,
      lightFixtureCount: 0,
      sanitaryCount: 0,
    });
  }

  const spaces: ParsedIfcSpace[] = [];
  const elementCounts: Record<string, number> = {};
  const doorTypeMap = new Map<
    string,
    { count: number; widthM?: number; heightM?: number }
  >();
  const windowTypeMap = new Map<
    string,
    { count: number; widthM?: number; heightM?: number }
  >();
  const lightingMap = new Map<string, number>();
  const lightingControlsMap = new Map<string, number>();
  const electricalDevicesMap = new Map<string, number>();
  const sanitaryMap = new Map<string, number>();
  const furnitureMap = new Map<string, number>();
  const doorInstances: NonNullable<ParsedIfcSummary['doorInstances']> = [];
  const windowInstances: NonNullable<ParsedIfcSummary['windowInstances']> = [];
  const lightingInstances: NonNullable<ParsedIfcSummary['lightingInstances']> = [];

  let wallCount = 0;
  let wallTotalLength = 0;
  let wallHasLength = false;
  let wallTotalSideArea = 0;
  let wallHasSideArea = false;
  let wallTotalVolume = 0;
  let wallHasVolume = false;

  let slabCount = 0;
  let slabTotalArea = 0;
  let slabHasArea = false;
  let slabTotalVolume = 0;
  let slabHasVolume = false;

  const rootGroup = new THREE.Group();
  rootGroup.name = projectName;

  const materialCache = new Map<string, THREE.MeshStandardMaterial>();
  const getMaterial = (key: string): THREE.MeshStandardMaterial => {
    const existing = materialCache.get(key);
    if (existing) return existing;
    const cfg = CATEGORY_COLORS[key] ?? CATEGORY_COLORS.DEFAULT;
    const mat = new THREE.MeshStandardMaterial({
      color: cfg.color,
      roughness: key === 'LIGHTING' ? 0.25 : key === 'IFCWINDOW' ? 0.15 : 0.72,
      metalness: key === 'LIGHTING' ? 0.35 : 0.08,
      emissive: cfg.emissive ?? 0x000000,
      emissiveIntensity: cfg.emissive ? 0.45 : 0,
      transparent: cfg.opacity !== undefined,
      opacity: cfg.opacity ?? 1,
      side: THREE.DoubleSide,
      depthWrite: cfg.opacity === undefined || cfg.opacity > 0.6,
    });
    materialCache.set(key, mat);
    return mat;
  };

  const trackedTypes = new Set([
    'IFCWALL',
    'IFCWALLSTANDARDCASE',
    'IFCCURTAINWALL',
    'IFCSLAB',
    'IFCROOF',
    'IFCDOOR',
    'IFCWINDOW',
    'IFCLIGHTFIXTURE',
    'IFCFLOWTERMINAL',
    'IFCFLOWCONTROLLER',
    'IFCFURNISHINGELEMENT',
    'IFCSANITARYTERMINAL',
    'IFCSTAIR',
    'IFCSTAIRFLIGHT',
    'IFCCOLUMN',
    'IFCBEAM',
    'IFCRAILING',
    'IFCSPACE',
  ]);

  let totalMeshCount = 0;

  for (const ent of entities.values()) {
    if (!trackedTypes.has(ent.type)) continue;

    const a = getArgs(ent);
    const elemName = getStr(a[2]) || getStr(a[4]) || ent.type;
    const longName = getStr(a[7]);
    const typeName = elementTypeName.get(ent.id) || '';
    const displayName = longName || elemName;
    const baseLabel = elemName.includes(':') ? elemName : typeName || elemName;
    const cleanFamily =
      baseLabel.split(':').slice(0, 2).join(':').trim() || ent.type;
    const storeyName = elementToStorey.get(ent.id) ?? null;
    const storeyBucket = storeyName ? storeyAgg.get(storeyName) : undefined;
    const q = elemQuantities.get(ent.id);

    if (ent.type === 'IFCSPACE') {
      let areaSqM = q?.areaSqM ?? null;
      let areaSource: ParsedIfcSpace['areaSource'] =
        areaSqM !== null ? 'Explicit BIM Property' : 'Unknown';
      let heightM = q?.heightM ?? null;

      // Fallback: extract extrusion area & height from IFCSPACE shape if explicit property missing
      const shapeRef = getRef(a[6]);
      if ((areaSqM === null || heightM === null) && shapeRef !== null) {
        const prodShape = entities.get(shapeRef);
        if (prodShape && prodShape.type === 'IFCPRODUCTDEFINITIONSHAPE') {
          const repRefs = getRefArray(getArgs(prodShape)[2]);
          for (const rr of repRefs) {
            const rep = entities.get(rr);
            if (!rep) continue;
            const itemRefs = getRefArray(getArgs(rep)[3]);
            for (const ir of itemRefs) {
              const item = entities.get(ir);
              if (item && item.type === 'IFCEXTRUDEDAREASOLID') {
                const ia = getArgs(item);
                const extH = getNum(ia[3], 0) * lengthToMeters;
                if (heightM === null && extH > 0) heightM = extH;
                if (areaSqM === null) {
                  const extArea = computeProfileAreaSqM(getRef(ia[0]));
                  if (extArea !== null) {
                    areaSqM = extArea;
                    areaSource = 'Calculated Extrusion';
                  }
                }
              }
            }
          }
        }
      }

      const roundedArea = areaSqM !== null ? Number(areaSqM.toFixed(2)) : null;
      spaces.push({
        id: `#${ent.id}`,
        code: elemName,
        name: longName || elemName,
        storey: storeyName,
        areaSqM: roundedArea,
        areaSource,
        volumeCuM:
          q?.volumeCuM !== null && q?.volumeCuM !== undefined
            ? Number(q.volumeCuM.toFixed(2))
            : null,
        heightM: heightM !== null ? Number(heightM.toFixed(2)) : null,
      });

      if (storeyBucket) {
        storeyBucket.spaceCount += 1;
        if (roundedArea !== null) {
          storeyBucket.netFloorAreaSqM += roundedArea;
          storeyBucket.hasArea = true;
        }
      }
      elementCounts.IFCSPACE = (elementCounts.IFCSPACE ?? 0) + 1;
    } else {
      elementCounts[ent.type] = (elementCounts[ent.type] ?? 0) + 1;
    }

    if (ent.type === 'IFCDOOR') {
      if (storeyBucket) storeyBucket.doorCount += 1;
      // IFCDOOR attributes 8=OverallHeight, 9=OverallWidth
      const hRaw = getNum(a[8], 0);
      const wRaw = getNum(a[9], 0);
      const doorKey = typeName || elemName.split(':').slice(0, 2).join(':') || 'Door';
      const prev = doorTypeMap.get(doorKey) ?? { count: 0 };
      prev.count += 1;
      const wM = wRaw > 0 ? Number((wRaw * lengthToMeters).toFixed(3)) : undefined;
      const hM = hRaw > 0 ? Number((hRaw * lengthToMeters).toFixed(3)) : undefined;
      if (wM && !prev.widthM) prev.widthM = wM;
      if (hM && !prev.heightM) prev.heightM = hM;
      doorTypeMap.set(doorKey, prev);
      if (doorInstances.length < 150) {
        const spIds = elemToSpaceIds.get(ent.id);
        const spNames = spIds
          ? Array.from(spIds)
              .map((sid) => spaceLabelMap.get(sid))
              .filter((s): s is string => Boolean(s))
          : [];
        doorInstances.push({
          id: `#${ent.id}`,
          name: elemName,
          type: doorKey,
          storey: storeyName,
          ...(wM ? { widthM: wM } : {}),
          ...(hM ? { heightM: hM } : {}),
          ...(spNames.length > 0 ? { spaces: Array.from(new Set(spNames)) } : {}),
        });
      }
    } else if (ent.type === 'IFCWINDOW') {
      if (storeyBucket) storeyBucket.windowCount += 1;
      const hRaw = getNum(a[8], 0);
      const wRaw = getNum(a[9], 0);
      const winKey = typeName || elemName.split(':').slice(0, 2).join(':') || 'Window';
      const prev = windowTypeMap.get(winKey) ?? { count: 0 };
      prev.count += 1;
      const wM = wRaw > 0 ? Number((wRaw * lengthToMeters).toFixed(3)) : undefined;
      const hM = hRaw > 0 ? Number((hRaw * lengthToMeters).toFixed(3)) : undefined;
      if (wM && !prev.widthM) prev.widthM = wM;
      if (hM && !prev.heightM) prev.heightM = hM;
      windowTypeMap.set(winKey, prev);
      if (windowInstances.length < 150) {
        const spIds = elemToSpaceIds.get(ent.id);
        const spNames = spIds
          ? Array.from(spIds)
              .map((sid) => spaceLabelMap.get(sid))
              .filter((s): s is string => Boolean(s))
          : [];
        windowInstances.push({
          id: `#${ent.id}`,
          name: elemName,
          type: winKey,
          storey: storeyName,
          ...(wM ? { widthM: wM } : {}),
          ...(hM ? { heightM: hM } : {}),
          ...(spNames.length > 0 ? { spaces: Array.from(new Set(spNames)) } : {}),
        });
      }
    } else if (
      ent.type === 'IFCWALL' ||
      ent.type === 'IFCWALLSTANDARDCASE' ||
      ent.type === 'IFCCURTAINWALL'
    ) {
      wallCount += 1;
      if (storeyBucket) storeyBucket.wallCount += 1;
      if (q?.lengthM) {
        wallTotalLength += q.lengthM;
        wallHasLength = true;
      }
      if (q?.netSideAreaSqM) {
        wallTotalSideArea += q.netSideAreaSqM;
        wallHasSideArea = true;
      }
      if (q?.volumeCuM) {
        wallTotalVolume += q.volumeCuM;
        wallHasVolume = true;
      }
    } else if (ent.type === 'IFCSLAB') {
      slabCount += 1;
      if (q?.areaSqM) {
        slabTotalArea += q.areaSqM;
        slabHasArea = true;
      }
      if (q?.volumeCuM) {
        slabTotalVolume += q.volumeCuM;
        slabHasVolume = true;
      }
    } else if (ent.type === 'IFCSTAIR' && storeyBucket) {
      storeyBucket.stairCount += 1;
    }

    const searchLabel = `${elemName} ${longName} ${typeName}`;
    const isControlOrOutlet = CONTROL_OR_RECEPTACLE_REGEX.test(searchLabel);
    const isLight =
      ent.type === 'IFCLIGHTFIXTURE' ||
      (!isControlOrOutlet &&
        (ent.type === 'IFCFLOWTERMINAL' ||
          ent.type === 'IFCFURNISHINGELEMENT') &&
        LUMINAIRE_REGEX.test(searchLabel));
    const isControlSwitch = /switch|dimmer|sensor|schakelaar/i.test(searchLabel);
    const isElectricalDevice =
      !isControlSwitch &&
      (/receptacle|outlet|panelboard|breaker|telephone|board|socket/i.test(
        searchLabel,
      ) ||
        ent.type === 'IFCFLOWCONTROLLER');
    const isSanitary =
      ent.type === 'IFCSANITARYTERMINAL' ||
      ((ent.type === 'IFCFLOWTERMINAL' || ent.type === 'IFCFURNISHINGELEMENT') &&
        SANITARY_OR_APPLIANCE_REGEX.test(searchLabel));

    let semanticCategory: string = 'Structural';
    if (isLight) {
      semanticCategory = 'Lighting';
      lightingMap.set(cleanFamily, (lightingMap.get(cleanFamily) ?? 0) + 1);
      if (storeyBucket) storeyBucket.lightFixtureCount += 1;
      if (lightingInstances.length < 150) {
        lightingInstances.push({
          id: `#${ent.id}`,
          name: elemName,
          type: cleanFamily,
          storey: storeyName,
        });
      }
    } else if (isControlSwitch) {
      semanticCategory = 'Other';
      lightingControlsMap.set(
        cleanFamily,
        (lightingControlsMap.get(cleanFamily) ?? 0) + 1,
      );
    } else if (isElectricalDevice) {
      semanticCategory = 'Other';
      electricalDevicesMap.set(
        cleanFamily,
        (electricalDevicesMap.get(cleanFamily) ?? 0) + 1,
      );
    } else if (isSanitary) {
      semanticCategory = 'Plumbing & Appliances';
      sanitaryMap.set(cleanFamily, (sanitaryMap.get(cleanFamily) ?? 0) + 1);
      if (storeyBucket) storeyBucket.sanitaryCount += 1;
    } else if (
      ent.type === 'IFCFURNISHINGELEMENT' ||
      ent.type === 'IFCFLOWTERMINAL'
    ) {
      semanticCategory = 'Furniture';
      furnitureMap.set(cleanFamily, (furnitureMap.get(cleanFamily) ?? 0) + 1);
    } else if (ent.type === 'IFCDOOR' || ent.type === 'IFCWINDOW') {
      semanticCategory = 'Openings';
    } else if (ent.type === 'IFCSPACE') {
      semanticCategory = 'Space';
    }

    // Build 3D geometry
    const placementRef = getRef(a[5]);
    const shapeRef = getRef(a[6]);
    if (shapeRef === null) continue;

    const prodShape = entities.get(shapeRef);
    if (!prodShape || prodShape.type !== 'IFCPRODUCTDEFINITIONSHAPE') continue;
    const psa = getArgs(prodShape);
    const repRefs = getRefArray(psa[2]);

    const localMat = getLocalPlacementMatrix(placementRef);
    const worldMat = new THREE.Matrix4().multiplyMatrices(zUpToYUp, localMat);

    const pieces: Array<{
      geometry: THREE.BufferGeometry;
      matrix: THREE.Matrix4;
    }> = [];
    for (const rr of repRefs) {
      const rep = entities.get(rr);
      if (!rep || rep.type !== 'IFCSHAPEREPRESENTATION') continue;
      const ra = getArgs(rep);
      const repId = getStr(ra[1]).toLowerCase();
      if (repId === 'axis' || repId === 'box') continue;
      const itemRefs = getRefArray(ra[3]);
      for (const ir of itemRefs) {
        buildRepresentationGeometries(ir, pieces, worldMat);
      }
    }

    const matKey = isLight ? 'LIGHTING' : ent.type;
    const mat = getMaterial(matKey);

    for (const piece of pieces) {
      if (totalMeshCount > 3500) {
        piece.geometry.dispose();
        continue;
      }
      const mesh = new THREE.Mesh(piece.geometry, mat);
      mesh.applyMatrix4(piece.matrix);
      mesh.castShadow = ent.type !== 'IFCSPACE' && ent.type !== 'IFCWINDOW';
      mesh.receiveShadow = true;
      mesh.userData = {
        ifcId: `#${ent.id}`,
        ifcType: ent.type,
        name: displayName,
        familyName: cleanFamily,
        typeName,
        storey: storeyName ?? '',
        semanticCategory,
        isLight,
        isControlOrOutlet: isControlSwitch || isElectricalDevice,
        isSanitary,
      };
      rootGroup.add(mesh);
      totalMeshCount++;
    }
  }

  let boundingBoxMeters: ParsedIfcSummary['boundingBoxMeters'] = null;
  if (rootGroup.children.length > 0) {
    const box = new THREE.Box3().setFromObject(rootGroup);
    if (!box.isEmpty()) {
      const size = box.getSize(new THREE.Vector3());
      const center = box.getCenter(new THREE.Vector3());
      rootGroup.position.set(-center.x, -box.min.y, -center.z);
      boundingBoxMeters = {
        width: Number(size.x.toFixed(2)),
        depth: Number(size.z.toFixed(2)),
        height: Number(size.y.toFixed(2)),
      };
    }
  }

  const storeys: ParsedIfcStoreyBreakdown[] = Array.from(storeyAgg.values())
    .sort((a, b) => a.elevationM - b.elevationM)
    .map((s) => ({
      name: s.name,
      elevationM: s.elevationM,
      spaceCount: s.spaceCount,
      netFloorAreaSqM: s.hasArea ? Number(s.netFloorAreaSqM.toFixed(2)) : null,
      doorCount: s.doorCount,
      windowCount: s.windowCount,
      wallCount: s.wallCount,
      stairCount: s.stairCount,
      lightFixtureCount: s.lightFixtureCount,
      sanitaryCount: s.sanitaryCount,
    }));

  const spacesWithArea = spaces.filter((s) => s.areaSqM !== null);
  const totalSpaceAreaSqM =
    spacesWithArea.length > 0
      ? Number(
          spacesWithArea
            .reduce((acc, s) => acc + (s.areaSqM ?? 0), 0)
            .toFixed(2),
        )
      : null;

  const toSortedArray = (m: Map<string, number>) =>
    Array.from(m.entries())
      .map(([name, count]) => ({ name, count }))
      .sort((a, b) => b.count - a.count);

  return {
    summary: {
      schema,
      authoringTool,
      projectName,
      lengthUnit,
      lengthToMeters,
      storeys,
      spaces,
      totalSpaceAreaSqM,
      elementCounts,
      doorTypes: Array.from(doorTypeMap.entries())
        .map(([name, d]) => ({ name, ...d }))
        .sort((a, b) => b.count - a.count),
      windowTypes: Array.from(windowTypeMap.entries())
        .map(([name, w]) => ({ name, ...w }))
        .sort((a, b) => b.count - a.count),
      doorInstances,
      windowInstances,
      lightingInstances,
      lightingFixtures: toSortedArray(lightingMap),
      lightingControls: toSortedArray(lightingControlsMap),
      electricalDevices: toSortedArray(electricalDevicesMap),
      sanitaryAndAppliances: toSortedArray(sanitaryMap),
      furnitureAndFixtures: toSortedArray(furnitureMap),
      wallStats: {
        count: wallCount,
        totalLengthM: wallHasLength ? Number(wallTotalLength.toFixed(3)) : null,
        totalNetSideAreaSqM: wallHasSideArea
          ? Number(wallTotalSideArea.toFixed(3))
          : null,
        totalNetVolumeCuM: wallHasVolume
          ? Number(wallTotalVolume.toFixed(3))
          : null,
      },
      slabStats: {
        count: slabCount,
        totalNetAreaSqM: slabHasArea ? Number(slabTotalArea.toFixed(3)) : null,
        totalNetVolumeCuM: slabHasVolume
          ? Number(slabTotalVolume.toFixed(3))
          : null,
      },
      boundingBoxMeters,
      totalMeshCount,
    },
    group: rootGroup,
  };
}
