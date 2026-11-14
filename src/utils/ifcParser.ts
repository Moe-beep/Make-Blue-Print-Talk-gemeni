import * as THREE from 'three';
import type { ParsedIfcSummary } from '../types/analysis';

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

function parseStepValue(token: string): unknown {
  if (!token || token === '$' || token === '*') return null;
  if (token.startsWith("'") && token.endsWith("'")) {
    return decodeIfcString(token.slice(1, -1).replace(/''/g, "'"));
  }
  if (token.startsWith('#')) {
    return { ref: parseInt(token.slice(1), 10) };
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

function decodeIfcString(str: string): string {
  // Handle basic IFC \X2\...\X0\ UTF-16BE escapes if present
  return str.replace(/\\X2\\([0-9A-Fa-f]+)\\X0\\/g, (_, hex: string) => {
    let out = '';
    for (let i = 0; i + 3 < hex.length; i += 4) {
      out += String.fromCharCode(parseInt(hex.slice(i, i + 4), 16));
    }
    return out;
  });
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

const LIGHTING_REGEX =
  /lamp|light|luminaire|sconce|pendant|chandelier|downlight|spotlight|fixture|bulb|led|fluorescent|track\s*light/i;

const CATEGORY_COLORS: Record<string, { color: number; opacity?: number; emissive?: number }> = {
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

export function parseIfcFile(text: string): ParsedIfcResult {
  // Strip C-style STEP comments
  const cleaned = text.replace(/\/\*[\s\S]*?\*\//g, '');

  // Header metadata
  const schemaMatch = cleaned.match(/FILE_SCHEMA\s*\(\s*\(\s*'([^']+)'/i);
  const schema = schemaMatch?.[1] ?? 'IFC';

  const appMatch = cleaned.match(/IFCAPPLICATION\([^,]+,'[^']*','([^']+)'/i);
  const authoringTool = appMatch?.[1] ?? 'Unknown BIM Tool';

  // Parse DATA section statements
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
    entities.set(id, {
      id,
      type,
      rawArgs,
      args: [],
    });
  }

  const getArgs = (ent: StepEntity): unknown[] => {
    if (ent.args.length === 0 && ent.rawArgs.trim().length > 0) {
      ent.args = splitStepArgs(ent.rawArgs).map(parseStepValue);
    }
    return ent.args;
  };

  // Detect length & area units
  let lengthToMeters = 1;
  let lengthUnit = 'm';
  let areaToSqM = 1;
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
      if (unitType === 'LENGTHUNIT' && name === 'METRE') {
        if (prefix === 'MILLI') {
          lengthToMeters = 0.001;
          lengthUnit = 'mm';
        } else if (prefix === 'CENTI') {
          lengthToMeters = 0.01;
          lengthUnit = 'cm';
        } else {
          lengthToMeters = 1;
          lengthUnit = 'm';
        }
      }
    } else if (ent.type === 'IFCCONVERSIONBASEDUNIT') {
      const a = getArgs(ent);
      const unitType = getStr(a[1]);
      const name = getStr(a[2]).toUpperCase();
      if (unitType === 'LENGTHUNIT' && (name.includes('FOOT') || name.includes('FEET'))) {
        lengthToMeters = 0.3048;
        lengthUnit = 'ft';
      } else if (unitType === 'LENGTHUNIT' && name.includes('INCH')) {
        lengthToMeters = 0.0254;
        lengthUnit = 'in';
      } else if (unitType === 'AREAUNIT' && name.includes('FOOT')) {
        areaToSqM = 0.09290304;
      }
    }
  }

  // Fallback heuristic if areaToSqM was not explicitly set via IFCCONVERSIONBASEDUNIT
  if (areaToSqM === 1 && lengthUnit === 'ft') {
    areaToSqM = 0.09290304;
  }

  // Build property/quantity associations for IFCSPACE and elements
  const spaceQuantities = new Map<number, { areaSqM: number | null; volumeCuM: number | null }>();
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
      let foundArea: number | null = null;
      let foundVol: number | null = null;
      for (const qr of qRefs) {
        const qEnt = entities.get(qr);
        if (!qEnt) continue;
        const qa = getArgs(qEnt);
        const qName = getStr(qa[0]).toLowerCase();
        if (qEnt.type === 'IFCQUANTITYAREA') {
          const val = getNum(qa[3], NaN);
          if (Number.isFinite(val)) {
            const sqM = val * areaToSqM;
            if (
              foundArea === null ||
              qName.includes('net') ||
              qName.includes('gsa') ||
              qName.includes('floor')
            ) {
              foundArea = sqM;
            }
          }
        } else if (qEnt.type === 'IFCQUANTITYVOLUME') {
          const val = getNum(qa[3], NaN);
          if (Number.isFinite(val)) {
            foundVol = val * Math.pow(lengthToMeters, 3);
          }
        }
      }
      if (foundArea !== null || foundVol !== null) {
        for (const objId of relatedObjects) {
          const prev = spaceQuantities.get(objId) ?? { areaSqM: null, volumeCuM: null };
          spaceQuantities.set(objId, {
            areaSqM: foundArea ?? prev.areaSqM,
            volumeCuM: foundVol ?? prev.volumeCuM,
          });
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
    // IFC uses Z-up; Three.js uses Y-up -> map (x, y, z)_ifc -> (x, z, -y)_three
    const x = getNum(coords[0]) * lengthToMeters;
    const y = getNum(coords[1]) * lengthToMeters;
    const z = getNum(coords[2]) * lengthToMeters;
    const v = new THREE.Vector3(x, y, z);
    pointCache.set(id, v);
    return v;
  };

  const getDirection = (id: number | null, fallback: THREE.Vector3): THREE.Vector3 => {
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
      // Orthogonalize xAxis relative to zAxis
      xAxis = xAxis.clone().sub(zAxis.clone().multiplyScalar(xAxis.dot(zAxis)));
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
  const getLocalPlacementMatrix = (id: number | null, depth = 0): THREE.Matrix4 => {
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
      for (let i = 1; i < corners.length; i++) shape.lineTo(corners[i].x, corners[i].y);
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
        // If extrusion direction is inverted along Z, flip geometry
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

    if (ent.type === 'IFCFACETEDBREP' || ent.type === 'IFCFACETEDBREPWITHVOIDS') {
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
          const isOuter = bound.type === 'IFCFACEOUTERBOUND';
          if (!isOuter) continue;
          const loopRef = getRef(ba[0]);
          if (loopRef === null) continue;
          const loop = entities.get(loopRef);
          if (!loop || loop.type !== 'IFCPOLYLOOP') continue;
          const la = getArgs(loop);
          const ptRefs = getRefArray(la[0]);
          if (ptRefs.length < 3) continue;
          const pts = ptRefs.map(getPoint);
          // Fan triangulation from vertex 0
          for (let i = 1; i + 1 < pts.length; i++) {
            const p0 = pts[0];
            const p1 = pts[i];
            const p2 = pts[i + 1];
            positions.push(p0.x, p0.y, p0.z, p1.x, p1.y, p1.z, p2.x, p2.y, p2.z);
          }
        }
      }

      if (positions.length >= 9) {
        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geo.computeVertexNormals();
        out.push({ geometry: geo, matrix: parentMat.clone() });
      }
      return;
    }

    if (ent.type === 'IFCBOOLEANCLIPPINGRESULT' || ent.type === 'IFCBOOLEANRESULT') {
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
      // Apply simple cartesian transform if present
      const nextMat = new THREE.Matrix4().multiplyMatrices(parentMat, mapOriginMat);
      for (const ir of itemRefs) {
        buildRepresentationGeometries(ir, out, nextMat, depth + 1);
      }
    }
  };

  // Convert IFC Z-up coordinates to Three.js Y-up
  const zUpToYUp = new THREE.Matrix4().makeRotationX(-Math.PI / 2);

  // Extract storeys, spaces, and elements
  const storeys: Array<{ name: string; elevationM: number }> = [];
  const spaces: ParsedIfcSummary['spaces'] = [];
  const elementCounts: Record<string, number> = {};
  const lightingMap = new Map<string, number>();
  const furnitureMap = new Map<string, number>();

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
    if (ent.type === 'IFCBUILDINGSTOREY') {
      const a = getArgs(ent);
      const name = getStr(a[2]) || getStr(a[7]) || 'Storey';
      const elev = getNum(a[9], 0) * lengthToMeters;
      storeys.push({ name, elevationM: Number(elev.toFixed(2)) });
      continue;
    }

    if (!trackedTypes.has(ent.type)) continue;

    const a = getArgs(ent);
    const elemName = getStr(a[2]) || getStr(a[4]) || ent.type;
    const longName = getStr(a[7]);
    const displayName = longName || elemName;

    if (ent.type === 'IFCSPACE') {
      const q = spaceQuantities.get(ent.id);
      spaces.push({
        id: `#${ent.id}`,
        code: elemName,
        name: longName || elemName,
        areaSqM: q?.areaSqM !== null && q?.areaSqM !== undefined ? Number(q.areaSqM.toFixed(2)) : null,
        volumeCuM:
          q?.volumeCuM !== null && q?.volumeCuM !== undefined ? Number(q.volumeCuM.toFixed(2)) : null,
      });
    } else {
      elementCounts[ent.type] = (elementCounts[ent.type] ?? 0) + 1;
    }

    const isLight =
      ent.type === 'IFCLIGHTFIXTURE' ||
      ((ent.type === 'IFCFLOWTERMINAL' || ent.type === 'IFCFURNISHINGELEMENT') &&
        LIGHTING_REGEX.test(displayName));

    if (isLight) {
      const cleanFamily = displayName.split(':').slice(0, 2).join(':').trim() || 'Light Fixture';
      lightingMap.set(cleanFamily, (lightingMap.get(cleanFamily) ?? 0) + 1);
    } else if (
      ent.type === 'IFCFURNISHINGELEMENT' ||
      ent.type === 'IFCFLOWTERMINAL' ||
      ent.type === 'IFCSANITARYTERMINAL'
    ) {
      const cleanFamily = displayName.split(':').slice(0, 2).join(':').trim() || ent.type;
      furnitureMap.set(cleanFamily, (furnitureMap.get(cleanFamily) ?? 0) + 1);
    }

    // Build 3D geometry if representation exists
    const placementRef = getRef(a[5]);
    const shapeRef = getRef(a[6]);
    if (shapeRef === null) continue;

    const prodShape = entities.get(shapeRef);
    if (!prodShape || prodShape.type !== 'IFCPRODUCTDEFINITIONSHAPE') continue;
    const psa = getArgs(prodShape);
    const repRefs = getRefArray(psa[2]);

    const localMat = getLocalPlacementMatrix(placementRef);
    const worldMat = new THREE.Matrix4().multiplyMatrices(zUpToYUp, localMat);

    const pieces: Array<{ geometry: THREE.BufferGeometry; matrix: THREE.Matrix4 }> = [];
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

    // Cap overly dense single-element meshes to keep WebGL responsive
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
        isLight,
      };
      rootGroup.add(mesh);
      totalMeshCount++;
    }
  }

  // Center the group horizontally on X/Z and place bottom at Y = 0
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

  storeys.sort((a, b) => a.elevationM - b.elevationM);

  const lightingFixtures = Array.from(lightingMap.entries())
    .map(([name, count]) => ({ name, count }))
    .sort((a, b) => b.count - a.count);

  const furnitureAndFixtures = Array.from(furnitureMap.entries())
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
      elementCounts,
      lightingFixtures,
      furnitureAndFixtures,
      boundingBoxMeters,
      totalMeshCount,
    },
    group: rootGroup,
  };
}
