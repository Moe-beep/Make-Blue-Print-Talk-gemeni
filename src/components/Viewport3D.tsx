import {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';
import { parseIfcFile } from '../utils/ifcParser';
import type {
  HighlightTarget,
  LoadedAsset,
  ParsedIfcSummary,
} from '../types/analysis';
import {
  ChevronDown,
  ChevronUp,
  Crosshair,
  Eye,
  Focus,
  Grid3X3,
  Lightbulb,
  RotateCcw,
  X,
} from 'lucide-react';

export interface Viewport3DHandle {
  captureSnapshots: () => Promise<{
    isometricBase64: string;
    topPlanBase64: string;
  } | null>;
}

interface Viewport3DProps {
  asset: LoadedAsset;
  activeHighlight?: HighlightTarget | null;
  onClearHighlight?: () => void;
  onParsedSummary: (summary: Record<string, unknown>) => void;
  onSelectElement?: (
    info: { name: string; type: string; sizeText: string } | null,
  ) => void;
}

type CameraPreset = 'iso' | 'top' | 'front' | 'side';

function detectSoftwareWebGL(): boolean {
  try {
    const gl = document.createElement('canvas').getContext('webgl2');
    if (!gl) return true;
    const ext = gl.getExtension('WEBGL_debug_renderer_info');
    const param = ext ? ext.UNMASKED_RENDERER_WEBGL : gl.RENDERER;
    const gpu = String(gl.getParameter(param));
    gl.getExtension('WEBGL_lose_context')?.loseContext();
    return /SwiftShader|llvmpipe|softpipe|Software|Basic Render/i.test(gpu);
  } catch {
    return true;
  }
}

const SYNONYMS_3D: string[][] = [
  ['lamp', 'light', 'luminaire', 'pendant', 'sconce', 'chandelier', 'bulb', 'fixture'],
  ['switch', 'dimmer', 'sensor', 'schakelaar'],
  ['door', 'doors', 'tuer', 'tür', 'porte', 'ifcdoor'],
  ['window', 'windows', 'fenster', 'fenetre', 'ifcwindow'],
  ['toilet', 'wc', 'water closet', 'commode'],
  ['sink', 'basin', 'lavatory', 'vanity'],
  ['shower', 'bathtub', 'bath', 'tub'],
  ['wall', 'walls', 'ifcwall', 'ifcwallstandardcase', 'curtainwall'],
  ['slab', 'floor', 'deck', 'ifcslab'],
  ['roof', 'ifcroof'],
  ['stair', 'stairs', 'ifcstair', 'ifcstairflight'],
  ['column', 'columns', 'pillar', 'ifccolumn'],
  ['beam', 'beams', 'joist', 'ifcbeam'],
  ['chair', 'seating', 'stool'],
  ['table', 'desk'],
  ['sofa', 'couch'],
  ['kitchen', 'küche'],
  ['bedroom', 'schlafzimmer'],
  ['living', 'wohnzimmer'],
  ['bathroom', 'bath', 'bad'],
];

function buildSynonymGroup(term: string): string[] {
  const t = term.trim().toLowerCase();
  const group = new Set<string>([t]);
  for (const grp of SYNONYMS_3D) {
    if (grp.some((g) => g === t || (t.length >= 4 && (g.includes(t) || t.includes(g))))) {
      for (const g of grp) group.add(g);
    }
  }
  return Array.from(group);
}

function buildKeywordGroups3D(rawKeywords: string[]): string[][] {
  const cleaned = rawKeywords
    .map((s) => s.trim().toLowerCase())
    .filter((s) => s.length >= 2 && !/^\d+$/.test(s));
  const groups: string[][] = [];
  for (const term of cleaned) {
    const sg = buildSynonymGroup(term);
    // Merge into an existing group if synonyms overlap (prevents ["door", "ifcdoor"] requiring two different groups)
    const existing = groups.find((g) => g.some((item) => sg.includes(item)));
    if (existing) {
      for (const item of sg) {
        if (!existing.includes(item)) existing.push(item);
      }
    } else {
      groups.push(sg);
    }
  }
  return groups;
}

function doesMeshMatchHighlight(
  mesh: THREE.Mesh,
  target: HighlightTarget,
): boolean {
  if (target.subTargets && target.subTargets.length > 0) {
    return target.subTargets.some((sub) => doesMeshMatchHighlight(mesh, sub));
  }

  const ud = mesh.userData ?? {};
  const ifcId = String(ud.ifcId || '').trim();
  const name = String(ud.name || mesh.name || '').toLowerCase();
  const familyName = String(ud.familyName || '').toLowerCase();
  const typeName = String(ud.typeName || '').toLowerCase();
  const ifcType = String(ud.ifcType || '').toLowerCase();
  const storey = String(ud.storey || '').toLowerCase();
  const semCat = String(ud.semanticCategory || '').toLowerCase();
  const isLight = Boolean(ud.isLight);
  const isControlOrOutlet = Boolean(ud.isControlOrOutlet);
  const isSanitary = Boolean(ud.isSanitary);

  // 1. Exact IFC entity IDs (1-to-1 exact subset match, e.g. 6 specific doors)
  if (target.ifcIds && target.ifcIds.length > 0) {
    if (!ifcId) return false;
    return target.ifcIds.some((id) => id.trim() === ifcId);
  }

  // 2. Mandatory storey scope filter if specified
  if (target.storey && target.storey.trim()) {
    const sNeed = target.storey.trim().toLowerCase();
    if (!storey || !storey.includes(sNeed)) {
      return false;
    }
  }

  // 3. Mandatory IFC class filter if specified
  if (target.ifcTypes && target.ifcTypes.length > 0) {
    const allowed = target.ifcTypes.some((t) => ifcType === t.toLowerCase());
    if (!allowed) return false;
  }

  const haystack = `${name} ${familyName} ${typeName} ${ifcType} ${storey} ${semCat}`;

  // 4. Exclusion keywords filter
  if (target.excludeKeywords && target.excludeKeywords.length > 0) {
    const excluded = target.excludeKeywords.some((ex) => {
      const low = ex.trim().toLowerCase();
      return low.length >= 2 && haystack.includes(low);
    });
    if (excluded) return false;
  }

  // 5. Strict exactNames filter (never falls through to generic category/noun keywords!)
  if (target.exactNames && target.exactNames.length > 0) {
    return target.exactNames.some((en) => {
      const eLow = en.trim().toLowerCase();
      if (!eLow) return false;
      return (
        familyName === eLow ||
        name === eLow ||
        typeName === eLow ||
        familyName.includes(eLow) ||
        typeName.includes(eLow) ||
        name.includes(eLow)
      );
    });
  }

  const groups = buildKeywordGroups3D(target.keywords ?? []);
  const askingSwitches = groups.some((g) =>
    g.some((t) => /switch|dimmer|outlet|panel/i.test(t)),
  );
  const askingLights =
    target.category === 'Lighting' ||
    groups.some((g) =>
      g.some((t) => /lamp|light|luminaire|pendant|sconce|chandelier/i.test(t)),
    );

  if (askingLights && !askingSwitches) {
    if (isControlOrOutlet) return false;
    if (groups.length <= 1) {
      if (isLight) return true;
      return /lamp|light|luminaire|pendant|sconce|chandelier|bulb/i.test(name);
    }
    if (!isLight && !/lamp|light|luminaire|pendant|sconce|chandelier|bulb/i.test(name)) {
      return false;
    }
  }

  if (groups.length === 0) {
    if (target.ifcTypes && target.ifcTypes.length > 0) return true;
    if (target.category === 'Openings') {
      return ifcType === 'ifcdoor' || ifcType === 'ifcwindow';
    }
    if (target.category === 'Plumbing & Appliances') return isSanitary;
    if (target.category && target.category !== 'All') {
      return semCat === target.category.toLowerCase();
    }
    return false;
  }

  // Require ALL distinct keyword concept groups to match (e.g. "interior" AND "door")
  return groups.every((g) => g.some((t) => haystack.includes(t)));
}

export const Viewport3D = forwardRef<Viewport3DHandle, Viewport3DProps>(
  function Viewport3D(
    {
      asset,
      activeHighlight,
      onClearHighlight,
      onParsedSummary,
      onSelectElement,
    },
    ref,
  ) {
    const mountRef = useRef<HTMLDivElement | null>(null);
    const [dockCollapsed, setDockCollapsed] = useState(false);
    const [wireframe, setWireframe] = useState(false);
    const [highlightLights, setHighlightLights] = useState(false);
    const [showGrid, setShowGrid] = useState(true);
    const [activePreset, setActivePreset] = useState<CameraPreset>('iso');
    const [loadingMesh, setLoadingMesh] = useState(true);
    const [loadError, setLoadError] = useState<string | null>(null);
    const [highlightedCount, setHighlightedCount] = useState(0);
    const [selectedInfo, setSelectedInfo] = useState<{
      name: string;
      type: string;
      sizeText: string;
    } | null>(null);
    const [modelDims, setModelDims] = useState<{
      width: number;
      depth: number;
      height: number;
      meshes: number;
    } | null>(null);

    const onParsedRef = useRef(onParsedSummary);
    onParsedRef.current = onParsedSummary;
    const onSelectRef = useRef(onSelectElement);
    onSelectRef.current = onSelectElement;
    const highlightBoundsRef = useRef<{
      center: THREE.Vector3;
      size: THREE.Vector3;
    } | null>(null);

    const sceneApiRef = useRef<{
      renderer: THREE.WebGLRenderer;
      scene: THREE.Scene;
      camera: THREE.PerspectiveCamera;
      controls: OrbitControls;
      grid: THREE.GridHelper;
      modelRoot: THREE.Group | null;
      modelSize: THREE.Vector3;
      modelCenter: THREE.Vector3;
      highlightMaterial: THREE.MeshStandardMaterial;
      ghostMaterial: THREE.MeshStandardMaterial;
      setPreset: (preset: CameraPreset) => void;
      focusBox: (center: THREE.Vector3, size: THREE.Vector3) => void;
      requestRender: () => void;
    } | null>(null);

    useImperativeHandle(ref, () => ({
      captureSnapshots: async () => {
        const api = sceneApiRef.current;
        if (!api) return null;
        const { renderer, scene, camera, controls, modelSize, modelCenter } = api;

        const prevPos = camera.position.clone();
        const prevTarget = controls.target.clone();
        const maxDim = Math.max(modelSize.x, modelSize.y, modelSize.z, 4);

        // 1. Isometric perspective view
        camera.position.set(
          modelCenter.x + maxDim * 1.15,
          modelCenter.y + maxDim * 0.9,
          modelCenter.z + maxDim * 1.15,
        );
        controls.target.copy(modelCenter);
        camera.lookAt(modelCenter);
        camera.updateMatrixWorld();
        renderer.render(scene, camera);
        const isometricBase64 = renderer.domElement
          .toDataURL('image/png')
          .replace(/^data:image\/png;base64,/, '');

        // 2. Top-down architectural plan view
        camera.position.set(
          modelCenter.x,
          modelCenter.y + maxDim * 1.65,
          modelCenter.z + 0.01,
        );
        controls.target.copy(modelCenter);
        camera.lookAt(modelCenter);
        camera.updateMatrixWorld();
        renderer.render(scene, camera);
        const topPlanBase64 = renderer.domElement
          .toDataURL('image/png')
          .replace(/^data:image\/png;base64,/, '');

        // Restore user camera
        camera.position.copy(prevPos);
        controls.target.copy(prevTarget);
        camera.lookAt(prevTarget);
        camera.updateMatrixWorld();
        renderer.render(scene, camera);

        return { isometricBase64, topPlanBase64 };
      },
    }));

    useEffect(() => {
      const container = mountRef.current;
      if (!container) return;

      const isSoftwareGl = detectSoftwareWebGL();
      const width = Math.max(container.clientWidth, 320);
      const height = Math.max(container.clientHeight, 320);

      const renderer = new THREE.WebGLRenderer({
        antialias: !isSoftwareGl,
        powerPreference: 'high-performance',
        preserveDrawingBuffer: true,
      });
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      renderer.toneMapping = THREE.ACESFilmicToneMapping;
      renderer.toneMappingExposure = 1.08;
      renderer.setPixelRatio(
        isSoftwareGl ? 1 : Math.min(window.devicePixelRatio, 2),
      );
      renderer.setSize(width, height);
      renderer.shadowMap.enabled = !isSoftwareGl;
      renderer.shadowMap.type = THREE.PCFSoftShadowMap;
      container.appendChild(renderer.domElement);

      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0xedeae2);
      scene.fog = new THREE.Fog(0xedeae2, 45, 180);

      const pmremGenerator = new THREE.PMREMGenerator(renderer);
      const envTex = pmremGenerator.fromScene(new RoomEnvironment(), 0.04).texture;
      scene.environment = envTex;
      pmremGenerator.dispose();

      // Dedicated highlight & ghost materials (never mutates shared IFC materials)
      const highlightMaterial = new THREE.MeshStandardMaterial({
        color: 0xd97706,
        emissive: 0xc2410c,
        emissiveIntensity: 0.72,
        roughness: 0.35,
        metalness: 0.15,
      });
      const ghostMaterial = new THREE.MeshStandardMaterial({
        color: 0xb5b2a8,
        roughness: 0.85,
        metalness: 0.05,
        transparent: true,
        opacity: 0.14,
        depthWrite: false,
      });

      // Three-point architectural lighting rig
      const hemiLight = new THREE.HemisphereLight(0xfefcf8, 0xdad6cc, 0.95);
      scene.add(hemiLight);

      const keyLight = new THREE.DirectionalLight(0xfff7ed, 1.45);
      keyLight.position.set(18, 28, 16);
      keyLight.castShadow = !isSoftwareGl;
      keyLight.shadow.mapSize.set(1024, 1024);
      keyLight.shadow.bias = -0.0005;
      keyLight.shadow.normalBias = 0.02;
      scene.add(keyLight);

      const rimLight = new THREE.DirectionalLight(0xe0f2fe, 0.55);
      rimLight.position.set(-16, 12, -18);
      scene.add(rimLight);

      const grid = new THREE.GridHelper(60, 60, 0x9c988e, 0xd4d0c6);
      grid.position.y = -0.01;
      scene.add(grid);

      const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 600);
      camera.position.set(14, 11, 14);

      const controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = !isSoftwareGl;
      controls.dampingFactor = 0.08;
      controls.maxPolarAngle = Math.PI * 0.495;

      let frameId = 0;
      let renderBusy = false;
      let renderDirty = false;
      let disposed = false;

      let transitionActive = false;
      const targetCamPos = new THREE.Vector3();
      const targetLookAt = new THREE.Vector3();
      let lastTime = performance.now();

      function afterDraw(cb: () => void) {
        const gl = renderer.getContext() as WebGL2RenderingContext;
        if (!gl || typeof gl.fenceSync !== 'function') {
          frameId = window.setTimeout(cb, 80);
          return;
        }
        const sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
        gl.flush();
        const poll = () => {
          if (disposed) {
            if (sync) gl.deleteSync(sync);
            return;
          }
          const state = sync
            ? gl.getSyncParameter(sync, gl.SYNC_STATUS)
            : gl.SIGNALED;
          if (state === gl.UNSIGNALED) {
            frameId = window.setTimeout(poll, 5);
          } else {
            if (sync) gl.deleteSync(sync);
            frameId = window.setTimeout(cb, 80);
          }
        };
        frameId = window.setTimeout(poll, 5);
      }

      function requestRender() {
        if (disposed) return;
        if (renderBusy) {
          renderDirty = true;
          return;
        }
        renderBusy = true;
        frameId = requestAnimationFrame(() => {
          if (disposed) return;
          renderDirty = false;
          const now = performance.now();
          const dt = Math.min((now - lastTime) / 1000, 0.1);
          lastTime = now;

          if (transitionActive) {
            const lambda = 10;
            camera.position.x = THREE.MathUtils.damp(
              camera.position.x,
              targetCamPos.x,
              lambda,
              dt,
            );
            camera.position.y = THREE.MathUtils.damp(
              camera.position.y,
              targetCamPos.y,
              lambda,
              dt,
            );
            camera.position.z = THREE.MathUtils.damp(
              camera.position.z,
              targetCamPos.z,
              lambda,
              dt,
            );
            controls.target.x = THREE.MathUtils.damp(
              controls.target.x,
              targetLookAt.x,
              lambda,
              dt,
            );
            controls.target.y = THREE.MathUtils.damp(
              controls.target.y,
              targetLookAt.y,
              lambda,
              dt,
            );
            controls.target.z = THREE.MathUtils.damp(
              controls.target.z,
              targetLookAt.z,
              lambda,
              dt,
            );
            if (
              camera.position.distanceTo(targetCamPos) < 0.03 &&
              controls.target.distanceTo(targetLookAt) < 0.03
            ) {
              camera.position.copy(targetCamPos);
              controls.target.copy(targetLookAt);
              transitionActive = false;
            } else {
              renderDirty = true;
            }
          }

          controls.update();
          renderer.render(scene, camera);

          const done = () => {
            renderBusy = false;
            if (renderDirty && !disposed) requestRender();
          };
          if (isSoftwareGl) afterDraw(done);
          else done();
        });
      }

      const api = {
        renderer,
        scene,
        camera,
        controls,
        grid,
        modelRoot: null as THREE.Group | null,
        modelSize: new THREE.Vector3(10, 4, 10),
        modelCenter: new THREE.Vector3(0, 1.5, 0),
        highlightMaterial,
        ghostMaterial,
        setPreset: (preset: CameraPreset) => {
          const d = Math.max(api.modelSize.x, api.modelSize.y, api.modelSize.z, 4);
          const c = api.modelCenter;
          targetLookAt.copy(c);
          if (preset === 'iso') {
            targetCamPos.set(c.x + d * 1.15, c.y + d * 0.85, c.z + d * 1.15);
          } else if (preset === 'top') {
            targetCamPos.set(c.x, c.y + d * 1.65, c.z + 0.02);
          } else if (preset === 'front') {
            targetCamPos.set(c.x, c.y + d * 0.25, c.z + d * 1.45);
          } else {
            targetCamPos.set(c.x + d * 1.45, c.y + d * 0.25, c.z);
          }
          transitionActive = true;
          lastTime = performance.now();
          requestRender();
        },
        focusBox: (center: THREE.Vector3, size: THREE.Vector3) => {
          const d = Math.max(size.x, size.y, size.z, 2.5);
          targetLookAt.copy(center);
          targetCamPos.set(
            center.x + d * 1.1,
            center.y + d * 0.85,
            center.z + d * 1.1,
          );
          transitionActive = true;
          lastTime = performance.now();
          requestRender();
        },
        requestRender,
      };
      sceneApiRef.current = api;

      controls.addEventListener('change', () => {
        requestRender();
      });
      controls.addEventListener('start', () => {
        transitionActive = false;
      });

      const handleContextLost = (e: Event) => {
        e.preventDefault();
      };
      const handleContextRestored = () => {
        requestRender();
      };
      renderer.domElement.addEventListener('webglcontextlost', handleContextLost);
      renderer.domElement.addEventListener(
        'webglcontextrestored',
        handleContextRestored,
      );

      // Raycasting on click for inspecting individual BIM/mesh elements
      const raycaster = new THREE.Raycaster();
      const pointer = new THREE.Vector2();
      let pointerDownPos = { x: 0, y: 0 };

      const onPointerDown = (e: PointerEvent) => {
        pointerDownPos = { x: e.clientX, y: e.clientY };
      };
      const onPointerUp = (e: PointerEvent) => {
        const moveDist = Math.hypot(
          e.clientX - pointerDownPos.x,
          e.clientY - pointerDownPos.y,
        );
        if (moveDist > 5 || !api.modelRoot) return;
        const rect = renderer.domElement.getBoundingClientRect();
        pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
        raycaster.setFromCamera(pointer, camera);
        const hits = raycaster.intersectObjects(api.modelRoot.children, true);
        if (hits.length > 0) {
          const obj = hits[0].object as THREE.Mesh;
          const box = new THREE.Box3().setFromObject(obj);
          const sz = box.getSize(new THREE.Vector3());
          const info = {
            name: String(
              obj.userData?.familyName ||
                obj.userData?.name ||
                obj.name ||
                '3D Element',
            ),
            type: String(obj.userData?.ifcType || obj.type || 'Mesh'),
            sizeText: `${sz.x.toFixed(2)}m × ${sz.z.toFixed(2)}m × ${sz.y.toFixed(2)}m`,
          };
          setSelectedInfo(info);
          onSelectRef.current?.(info);
        } else {
          setSelectedInfo(null);
          onSelectRef.current?.(null);
        }
      };

      renderer.domElement.addEventListener('pointerdown', onPointerDown);
      renderer.domElement.addEventListener('pointerup', onPointerUp);

      const resizeObserver = new ResizeObserver(() => {
        if (disposed || !container) return;
        const w = Math.max(container.clientWidth, 200);
        const h = Math.max(container.clientHeight, 200);
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h);
        requestRender();
      });
      resizeObserver.observe(container);

      const loadModel = async () => {
        setLoadingMesh(true);
        setLoadError(null);
        setSelectedInfo(null);
        try {
          const lower = asset.name.toLowerCase();
          let group = new THREE.Group();
          let extraSummary: Record<string, unknown> = {};

          if (lower.endsWith('.ifc') && asset.rawText) {
            const parsed = parseIfcFile(asset.rawText);
            group = parsed.group;
            const s: ParsedIfcSummary = parsed.summary;
            extraSummary = {
              format: 'IFC',
              ...s,
            };
          } else if (
            (lower.endsWith('.glb') || lower.endsWith('.gltf')) &&
            asset.arrayBuffer
          ) {
            const loader = new GLTFLoader();
            const gltf = await new Promise<{ scene: THREE.Group }>(
              (resolve, reject) => {
                loader.parse(asset.arrayBuffer!, '', resolve, reject);
              },
            );
            group = gltf.scene;
          } else if (lower.endsWith('.obj') && asset.rawText) {
            const loader = new OBJLoader();
            group = loader.parse(asset.rawText);
          } else if (lower.endsWith('.stl') && asset.arrayBuffer) {
            const loader = new STLLoader();
            const geo = loader.parse(asset.arrayBuffer);
            const mat = new THREE.MeshStandardMaterial({
              color: 0xcbd5e1,
              roughness: 0.65,
              metalness: 0.1,
            });
            const mesh = new THREE.Mesh(geo, mat);
            mesh.name = asset.name;
            group.add(mesh);
          } else if (lower.endsWith('.ply') && asset.arrayBuffer) {
            const loader = new PLYLoader();
            const geo = loader.parse(asset.arrayBuffer);
            geo.computeVertexNormals();
            const mat = new THREE.MeshStandardMaterial({
              vertexColors: geo.hasAttribute('color'),
              color: geo.hasAttribute('color') ? 0xffffff : 0xcbd5e1,
              roughness: 0.65,
            });
            const mesh = new THREE.Mesh(geo, mat);
            mesh.name = asset.name;
            group.add(mesh);
          } else {
            throw new Error(`Unsupported 3D format for ${asset.name}`);
          }

          if (disposed) return;

          const meshNames: Record<string, number> = {};
          let meshCount = 0;
          group.traverse((child) => {
            if ((child as THREE.Mesh).isMesh) {
              const m = child as THREE.Mesh;
              meshCount++;
              const label = (m.name || m.userData?.name || 'Mesh').trim();
              meshNames[label] = (meshNames[label] ?? 0) + 1;
              if (!m.material) {
                m.material = new THREE.MeshStandardMaterial({
                  color: 0xcbd5e1,
                  roughness: 0.7,
                });
              }
              // Store original material cleanly so per-mesh highlights never affect shared templates
              m.userData.origMaterial = m.material;
              if (
                !lower.endsWith('.ifc') &&
                /lamp|light|sconce|pendant|chandelier|bulb|fixture/i.test(
                  label,
                ) &&
                !/switch|dimmer/i.test(label)
              ) {
                m.userData.isLight = true;
              }
            }
          });

          const box = new THREE.Box3().setFromObject(group);
          if (!box.isEmpty()) {
            const center = box.getCenter(new THREE.Vector3());
            if (!lower.endsWith('.ifc')) {
              group.position.set(-center.x, -box.min.y, -center.z);
            }
            const normBox = new THREE.Box3().setFromObject(group);
            const normSize = normBox.getSize(new THREE.Vector3());
            const normCenter = normBox.getCenter(new THREE.Vector3());
            api.modelSize.copy(normSize);
            api.modelCenter.copy(normCenter);

            setModelDims({
              width: Number(normSize.x.toFixed(2)),
              depth: Number(normSize.z.toFixed(2)),
              height: Number(normSize.y.toFixed(2)),
              meshes: meshCount,
            });

            if (!lower.endsWith('.ifc')) {
              const allMeshEntries = Object.entries(meshNames).map(
                ([name, count]) => ({ name, count }),
              );
              const lampMeshes = allMeshEntries.filter(
                (e) =>
                  /lamp|light|sconce|pendant|chandelier|bulb|fixture/i.test(
                    e.name,
                  ) && !/switch|dimmer/i.test(e.name),
              );
              extraSummary = {
                format: lower.split('.').pop()?.toUpperCase(),
                assumedUnit: 'Metres (standard glTF/CAD 3D units)',
                boundingBoxMeters: {
                  width: Number(normSize.x.toFixed(2)),
                  depth: Number(normSize.z.toFixed(2)),
                  height: Number(normSize.y.toFixed(2)),
                },
                boundingFootprintAreaSqM: Number(
                  (normSize.x * normSize.z).toFixed(2),
                ),
                totalMeshCount: meshCount,
                detectedLampMeshes: lampMeshes,
                meshNames: allMeshEntries.slice(0, 120),
              };
            }
          }

          api.modelRoot = group;
          scene.add(group);
          onParsedRef.current(extraSummary);

          const d = Math.max(api.modelSize.x, api.modelSize.y, api.modelSize.z, 4);
          camera.position.set(
            api.modelCenter.x + d * 1.15,
            api.modelCenter.y + d * 0.85,
            api.modelCenter.z + d * 1.15,
          );
          controls.target.copy(api.modelCenter);
          camera.lookAt(api.modelCenter);
          if (!isSoftwareGl) {
            renderer.compile(scene, camera);
          }
          requestRender();
          setLoadingMesh(false);
        } catch (err) {
          if (!disposed) {
            setLoadError(
              err instanceof Error ? err.message : 'Failed to parse 3D file',
            );
            setLoadingMesh(false);
          }
        }
      };

      void loadModel();

      return () => {
        disposed = true;
        cancelAnimationFrame(frameId);
        clearTimeout(frameId);
        resizeObserver.disconnect();
        renderer.domElement.removeEventListener('pointerdown', onPointerDown);
        renderer.domElement.removeEventListener('pointerup', onPointerUp);
        renderer.domElement.removeEventListener(
          'webglcontextlost',
          handleContextLost,
        );
        renderer.domElement.removeEventListener(
          'webglcontextrestored',
          handleContextRestored,
        );
        controls.dispose();
        envTex.dispose();
        highlightMaterial.dispose();
        ghostMaterial.dispose();
        scene.traverse((obj) => {
          if ((obj as THREE.Mesh).isMesh) {
            const m = obj as THREE.Mesh;
            m.geometry?.dispose();
            const origMat = m.userData?.origMaterial;
            if (origMat) {
              if (Array.isArray(origMat)) origMat.forEach((mt) => mt.dispose());
              else origMat.dispose();
            }
          }
        });
        renderer.dispose();
        if (renderer.domElement.parentElement === container) {
          container.removeChild(renderer.domElement);
        }
        sceneApiRef.current = null;
      };
    }, [asset]);

    // Sync wireframe, grid, highlightLights, and activeHighlight cleanly
    useEffect(() => {
      const api = sceneApiRef.current;
      if (!api) return;
      api.grid.visible = showGrid;

      const effectiveTarget: HighlightTarget | null =
        activeHighlight ??
        (highlightLights
          ? {
              id: 'lamps-toggle',
              label: 'Lamps & Lighting Fixtures',
              category: 'Lighting',
              keywords: ['lamp', 'light', 'luminaire'],
            }
          : null);

      api.highlightMaterial.wireframe = wireframe;
      api.ghostMaterial.wireframe = wireframe;

      let matchedCount = 0;
      const matchedIfcIds = new Set<string>();
      const highlightBox = new THREE.Box3();

      if (api.modelRoot) {
        api.modelRoot.updateMatrixWorld(true);
        api.modelRoot.traverse((child) => {
          if ((child as THREE.Mesh).isMesh) {
            const m = child as THREE.Mesh;
            const origMat = m.userData?.origMaterial;
            if (!origMat) return;

            if (effectiveTarget) {
              const matched = doesMeshMatchHighlight(m, effectiveTarget);
              if (matched) {
                const eid = m.userData?.ifcId ? String(m.userData.ifcId) : null;
                if (eid) {
                  if (!matchedIfcIds.has(eid)) {
                    matchedIfcIds.add(eid);
                    matchedCount += 1;
                  }
                } else {
                  matchedCount += 1;
                }
                m.material = api.highlightMaterial;
                m.renderOrder = 2;
                highlightBox.expandByObject(m);
              } else {
                m.material = api.ghostMaterial;
                m.renderOrder = 0;
              }
            } else {
              m.material = origMat;
              m.renderOrder = 0;
              const mats = Array.isArray(origMat) ? origMat : [origMat];
              for (const mat of mats) {
                if (mat && 'wireframe' in mat) {
                  (mat as THREE.MeshStandardMaterial).wireframe = wireframe;
                }
              }
            }
          }
        });
      }

      if (effectiveTarget && !highlightBox.isEmpty()) {
        highlightBoundsRef.current = {
          center: highlightBox.getCenter(new THREE.Vector3()),
          size: highlightBox.getSize(new THREE.Vector3()),
        };
      } else {
        highlightBoundsRef.current = null;
      }

      setHighlightedCount(matchedCount);
      api.requestRender();
    }, [wireframe, showGrid, highlightLights, activeHighlight, loadingMesh]);

    const handlePreset = (p: CameraPreset) => {
      setActivePreset(p);
      sceneApiRef.current?.setPreset(p);
    };

    const handleFocusHighlight = () => {
      if (!highlightBoundsRef.current || !sceneApiRef.current) return;
      sceneApiRef.current.focusBox(
        highlightBoundsRef.current.center,
        highlightBoundsRef.current.size,
      );
    };

    return (
      <div className="relative w-full h-full min-h-[480px] bg-[#EDEAE2] overflow-hidden select-none">
        {/* Isolated WebGL mount container */}
        <div ref={mountRef} className="w-full h-full" />

        {/* Top-left semantic geometry readout (solid backing, no backdrop-blur over WebGL) */}
        <div className="absolute top-3.5 left-3.5 z-10 pointer-events-none flex flex-col gap-1.5">
          <div className="bg-white/95 border border-[#E2DFD7] rounded-lg px-3 py-1.5 shadow-sm">
            <div className="text-xs font-medium text-[#161614] truncate max-w-[260px]">
              {asset.name}
            </div>
            {modelDims && (
              <div className="text-[11px] font-mono tabular-nums text-[#5C5A54] mt-0.5">
                {modelDims.width}m × {modelDims.depth}m × {modelDims.height}m ·{' '}
                {modelDims.meshes} meshes
              </div>
            )}
          </div>
          {selectedInfo && (
            <div className="bg-[#161614] text-white rounded-lg px-3 py-2 shadow-sm max-w-[280px] pointer-events-auto">
              <div className="text-[11px] font-mono text-[#F59E0B]">
                {selectedInfo.type}
              </div>
              <div className="text-xs font-medium mt-0.5 break-words">
                {selectedInfo.name}
              </div>
              <div className="text-[11px] font-mono tabular-nums text-[#D6D3CB] mt-0.5">
                {selectedInfo.sizeText}
              </div>
            </div>
          )}
        </div>

        {/* Top-right active highlight readout */}
        {activeHighlight && (
          <div className="absolute top-3.5 right-3.5 z-10 flex items-center gap-2 bg-[#161614] text-white rounded-lg px-3 py-1.5 shadow-md max-w-[340px]">
            <Crosshair className="w-3.5 h-3.5 text-[#F59E0B] shrink-0" />
            <div className="min-w-0">
              <div className="text-[11px] font-medium truncate">
                {activeHighlight.label}
              </div>
              <div className="text-[10px] font-mono text-[#D6D3CB] tabular-nums">
                {highlightedCount} 3D objects matched
              </div>
            </div>
            {highlightedCount > 0 && (
              <button
                type="button"
                onClick={handleFocusHighlight}
                title="Focus camera on highlighted objects"
                className="inline-flex items-center gap-1 px-2 py-0.5 text-[10px] font-medium bg-[#C26D2D] hover:bg-[#AD5E23] text-white rounded transition-colors cursor-pointer shrink-0"
              >
                <Focus className="w-3 h-3" />
                Focus
              </button>
            )}
            {onClearHighlight && (
              <button
                type="button"
                onClick={onClearHighlight}
                title="Clear highlight"
                className="p-1 rounded hover:bg-white/10 text-[#D6D3CB] hover:text-white transition-colors cursor-pointer shrink-0"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            )}
          </div>
        )}

        {/* Loading / error state */}
        {loadingMesh && (
          <div className="absolute inset-0 z-20 flex items-center justify-center bg-[#EDEAE2]/85">
            <div className="bg-white border border-[#E2DFD7] rounded-xl px-5 py-4 shadow-sm text-center">
              <div className="w-6 h-6 border-2 border-[#1F4E3D] border-t-transparent rounded-full animate-spin mx-auto" />
              <p className="text-xs font-medium text-[#3B3934] mt-2.5">
                Building 3D geometry &amp; extracting BIM entities…
              </p>
            </div>
          </div>
        )}

        {loadError && (
          <div className="absolute inset-0 z-20 flex items-center justify-center bg-[#EDEAE2]/95 p-6">
            <div className="bg-white border border-red-200 rounded-xl p-5 max-w-md text-center">
              <p className="text-sm font-semibold text-red-700">
                Could not render 3D file
              </p>
              <p className="text-xs text-[#5C5A54] mt-1">{loadError}</p>
            </div>
          </div>
        )}

        {/* Collapsible bottom control dock */}
        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-10 flex flex-col items-center max-w-[94%]">
          <button
            type="button"
            onClick={() => setDockCollapsed((v) => !v)}
            className="mb-1.5 inline-flex items-center gap-1.5 px-3 py-1 rounded-md bg-white/95 border border-[#E2DFD7] text-[11px] font-medium text-[#3B3934] hover:text-[#161614] shadow-sm transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            {dockCollapsed ? (
              <>
                <ChevronUp className="w-3.5 h-3.5" />
                Show 3D Controls
              </>
            ) : (
              <>
                <ChevronDown className="w-3.5 h-3.5" />
                Hide Controls
              </>
            )}
          </button>

          {!dockCollapsed && (
            <div className="flex flex-wrap items-center justify-center gap-2 px-3 py-2 rounded-xl bg-white/95 border border-[#E2DFD7] shadow-md">
              <div className="flex items-center gap-1 p-0.5 bg-[#F3F1EB] rounded-lg">
                {(
                  [
                    ['iso', 'Isometric'],
                    ['top', 'Top Plan'],
                    ['front', 'Front'],
                    ['side', 'Side'],
                  ] as const
                ).map(([key, label]) => (
                  <button
                    key={key}
                    type="button"
                    onClick={() => handlePreset(key)}
                    className={`px-2.5 py-1 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                      activePreset === key
                        ? 'bg-white text-[#161614] shadow-sm'
                        : 'text-[#5C5A54] hover:text-[#161614]'
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>

              <div className="h-5 w-px bg-[#E2DFD7]" />

              <button
                type="button"
                onClick={() => setWireframe((v) => !v)}
                className={`inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium rounded-md border transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                  wireframe
                    ? 'bg-[#1F4E3D] text-white border-[#1F4E3D]'
                    : 'bg-white text-[#3B3934] border-[#E2DFD7] hover:bg-[#F7F6F2]'
                }`}
              >
                <Eye className="w-3.5 h-3.5" />
                Wireframe
              </button>

              <button
                type="button"
                onClick={() => {
                  if (activeHighlight) onClearHighlight?.();
                  setHighlightLights((v) => !v);
                }}
                className={`inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium rounded-md border transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                  highlightLights && !activeHighlight
                    ? 'bg-[#C26D2D] text-white border-[#C26D2D]'
                    : 'bg-white text-[#3B3934] border-[#E2DFD7] hover:bg-[#F7F6F2]'
                }`}
              >
                <Lightbulb className="w-3.5 h-3.5" />
                Isolate Lamps
              </button>

              <button
                type="button"
                onClick={() => setShowGrid((v) => !v)}
                className={`inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium rounded-md border transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                  showGrid
                    ? 'bg-[#161614] text-white border-[#161614]'
                    : 'bg-white text-[#3B3934] border-[#E2DFD7] hover:bg-[#F7F6F2]'
                }`}
              >
                <Grid3X3 className="w-3.5 h-3.5" />
                Grid
              </button>

              <button
                type="button"
                onClick={() => handlePreset('iso')}
                title="Reset camera"
                className="inline-flex items-center gap-1 px-2 py-1 text-xs font-medium rounded-md border border-[#E2DFD7] bg-white text-[#3B3934] hover:bg-[#F7F6F2] transition-colors whitespace-nowrap shrink-0 cursor-pointer"
              >
                <RotateCcw className="w-3.5 h-3.5" />
              </button>
            </div>
          )}
        </div>
      </div>
    );
  },
);
