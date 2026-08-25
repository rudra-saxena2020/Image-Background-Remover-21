import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { apiFetch } from "@/lib/api";
import {
  Activity,
  AlertCircle,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Clock,
  Download,
  Image as ImageIcon,
  Info,
  Loader2,
  RefreshCw,
  Sparkles,
  Terminal,
  Zap,
} from "lucide-react";

// ─── Types ────────────────────────────────────────────────────────────────────

type LogStatus = "pending" | "success" | "error";

interface LogEntry {
  id: string;
  ts: number;
  prompt: string;
  status: LogStatus;
  durationMs?: number;
  bytes?: number;
  contentType?: string;
  error?: string;
}
interface GeneratedOutput {
  url: string;
  blob: Blob;
  index: number;
}

const STUDIO_ANGLE_DIRECTIONS = [
  "Luxury handbag catalog front: straight-on, centered, upright, complete product fully visible, clean retail product-page presentation.",
  "Luxury handbag catalog front-left three-quarter: camera approximately 45 degrees from the front-left, same upright styling and scale, clearly showing the front and left depth.",
  "Luxury handbag catalog front-right three-quarter: camera approximately 45 degrees from the front-right, same upright styling and scale, clearly showing the front and right depth.",
  "Luxury handbag catalog left profile: true full left-side view, same height and scale, showing the complete side construction and depth.",
  "Luxury handbag catalog right profile: true full right-side view, same height and scale, showing the complete side construction and depth.",
  "Luxury handbag catalog back: straight-on centered rear view, same upright styling and scale, showing the complete back construction.",
  "Luxury handbag catalog rear-left three-quarter: camera approximately 45 degrees from the rear-left, same upright styling and scale, clearly showing the back and left depth.",
  "Luxury handbag catalog rear-right three-quarter: camera approximately 45 degrees from the rear-right, same upright styling and scale, clearly showing the back and right depth.",
  "Luxury handbag catalog exterior detail: close framing of one real supported feature such as material, stitching, hardware, logo, or craftsmanship, with the product still clearly identifiable and no invented detail.",
  "Luxury handbag catalog top: camera directly above, preserve the exact real top geometry, handle/strap placement, and closed product state.",
  "Luxury handbag catalog base: camera from a low underside angle, preserve the exact real base construction and proportions without inventing feet or hardware.",
];
const FOUR_ANGLE_INDICES = [0, 1, 3, 5];
const NINE_ANGLE_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8];
const COMPLETE_STUDIO_SET_PROMPT = `Generate a complete premium e-commerce product photography set using one original product photograph as the exact product identity source. Each requested output must be one separate image containing exactly one photograph of exactly one complete product from exactly one camera angle. The camera may move around the product, but the product itself must not be restyled, reposed, rotated into a new pose, or physically rearranged. Use the reference only to identify the physical product; if it contains multiple views or a layout, do not use it as a source.

PRODUCT IDENTITY LOCK: The reference product is the absolute source of truth. Preserve its exact shape, silhouette, proportions, dimensions, material, texture, color, color calibration, stitching, seams, piping, edges, construction, logos, branding, graphics, text placement, hardware, buckles, rings, chains, zippers, metal finish, handles, straps, attachment points, pockets, panels, closures, and exterior details. Every attached component is geometry that must remain fixed: preserve the exact chain or strap type, count, length, thickness, attachment points, hardware, orientation, resting position, and visible drape. Do not lengthen, shorten, loosen, tighten, swing, drape, coil, cross, move, duplicate, replace, or invent a chain, strap, handle, charm, buckle, ring, or accessory. If a component is not clearly visible in the identity master, do not add it. Do not redesign, reinterpret, simplify, beautify, invent, replace, remove, duplicate, merge, distort, recolor, tint, brighten, or modify any feature.

SINGLE PRODUCT AND NORMAL STATE: Show exactly one complete physical product. Do not add a second product, duplicate, color variant, accessory, prop, hand, model, floating component, or extra object. Keep the product in its normal realistic CLOSED state and preserve the same neutral catalog pose in every image. Do not open, unzip, unfold, disassemble, explode, cut away, expose the interior, or show an interior view.

PHOTOGRAPHY: Create a genuinely different useful e-commerce viewpoint for each output, not a near-duplicate with only a tiny camera movement. The result must be a photorealistic premium commercial studio photograph, not CGI, a 3D render, illustration, AI-art styling, or artificially smooth plastic. Use physically accurate geometry, realistic material texture, authentic construction, believable metal reflections, natural imperfections where appropriate, realistic shadows, accurate perspective, sharp high-resolution details, and consistent color.

COMPOSITION AND STUDIO: Show the entire product unless the requested direction is an exterior detail close-up where the complete product remains clearly identifiable. Make it occupy approximately 70–85% of the frame with comfortable margins; do not crop handles, straps, chains, edges, or important components. Use a centered, balanced composition in the same plain warm-white or light-gray studio with soft diffused professional lighting, subtle contact shadow, consistent camera height, focal length, exposure, scale, white balance, and color calibration. The only meaningful difference between outputs must be the camera viewpoint and naturally changing visibility of the same fixed product features. Do not include angle numbers, labels, captions, UI elements, borders, frames, watermarks, decorative text, collage, contact sheet, grid, split screen, or multiple angles in one image.

FINAL ERROR-PREVENTION CONTRACT: First isolate only the intended physical product from the identity master and ignore screenshots, thumbnails, layouts, packaging, boxes, shopping bags, props, people, hands, clothing, mannequins, and background objects. The final image must contain ONLY one exact reference product, one angle, and an empty studio. ZERO HUMAN ELEMENTS: no person, hand, finger, arm, leg, face, body, model, mannequin, clothing, or product being held, carried, or worn. Never add a second bag, duplicate, miniature, alternate product, colorway, generic luxury item, box, packaging, shopping bag, prop, or non-permanently-attached accessory. Never invent or remove features; unclear details must remain conservative and unsupported details must remain absent. Reject the visual concept and return to the exact reference identity if silhouette, proportions, color, material, artwork, logo placement, hardware, chain, strap, attachment points, zipper, closure, or component shape changes. Reject melted geometry, warped edges, asymmetry, impossible stitching, duplicated hardware, malformed chains, broken straps, floating attachments, false reflections, CGI, or plastic-looking material. ABSOLUTE RULE: exactly one isolated, complete, closed, photorealistic physical product; no person, no hand, no second product, no packaging, no prop, no duplicate, no redesign, no color change, and no invented product.`;

interface RunPodStatus {
  configured: boolean;
  ready: boolean;
  reason: string;
}

interface ShopifyProduct {
  id: string;
  title: string;
  handle: string;
  product_type: string;
  vendor: string;
  images: { url: string; alt: string }[];
}
interface ShopifyImageChoice {
  url: string;
  productHandle: string;
  productTitle: string;
  index: number;
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function fmt(ms: number) {
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function timeLabel(ts: number) {
  return new Date(ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function shortPrompt(p: string, n = 60) {
  return p.length <= n ? p : p.slice(0, n) + "…";
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function StatusDot({ ready, loading }: { ready?: boolean; loading?: boolean }) {
  if (loading) return <span className="inline-block h-2 w-2 rounded-full bg-amber-400 animate-pulse" />;
  if (ready) return <span className="inline-block h-2 w-2 rounded-full bg-emerald-400" />;
  return <span className="inline-block h-2 w-2 rounded-full bg-red-400" />;
}

function LogRow({ entry }: { entry: LogEntry }) {
  return (
    <div className={cn(
      "grid grid-cols-[auto_1fr_auto] gap-3 items-start py-2.5 px-3 border-b border-border/40 last:border-b-0 text-xs font-mono",
      entry.status === "error" && "bg-destructive/5",
    )}>
      <div className="mt-0.5">
        {entry.status === "pending" && <Loader2 className="h-3.5 w-3.5 animate-spin text-amber-500" />}
        {entry.status === "success" && <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />}
        {entry.status === "error" && <AlertCircle className="h-3.5 w-3.5 text-destructive" />}
      </div>
      <div className="min-w-0">
        <p className="truncate text-foreground/80">{shortPrompt(entry.prompt)}</p>
        {entry.error && <p className="text-destructive/80 mt-0.5 truncate">{entry.error}</p>}
        {entry.status === "success" && (
          <p className="text-muted-foreground mt-0.5">
            {entry.bytes ? `${(entry.bytes / 1024).toFixed(1)} KB` : ""}{entry.contentType ? ` · ${entry.contentType}` : ""}
          </p>
        )}
      </div>
      <div className="text-right text-muted-foreground whitespace-nowrap">
        <p>{timeLabel(entry.ts)}</p>
        {entry.durationMs !== undefined && <p className="text-[10px] mt-0.5">{fmt(entry.durationMs)}</p>}
      </div>
    </div>
  );
}

function PropRow({ label, value, mono = false, badge }: { label: string; value: string; mono?: boolean; badge?: string }) {
  return (
    <div className="flex items-start gap-3 py-2 border-b border-border/40 last:border-b-0">
      <span className="text-xs text-muted-foreground w-32 shrink-0 pt-0.5">{label}</span>
      <span className={cn("text-xs flex-1", mono ? "font-mono" : "")}>
        {value}
        {badge && (
          <span className="ml-2 inline-block text-[10px] font-medium px-1.5 py-0.5 rounded-sm bg-primary/10 text-primary">{badge}</span>
        )}
      </span>
    </div>
  );
}

function StatusCodeRow({ code, meaning }: { code: string; meaning: string }) {
  const color =
    code === "200" ? "text-emerald-600" :
    code === "400" ? "text-amber-600" :
    "text-destructive";
  return (
    <div className="flex items-center gap-3 py-1.5 border-b border-border/30 last:border-b-0 text-xs">
      <span className={cn("font-mono font-semibold w-10 shrink-0", color)}>{code}</span>
      <span className="text-muted-foreground">{meaning}</span>
    </div>
  );
}

// ─── Main page ────────────────────────────────────────────────────────────────

export function Generate() {
  const [prompt, setPrompt] = useState("");
  const [references, setReferences] = useState<{ file: File; url: string }[]>([]);
  const [shopifyOpen, setShopifyOpen] = useState(false);
  const [shopifyQuery, setShopifyQuery] = useState("");
  const [shopifyProducts, setShopifyProducts] = useState<ShopifyProduct[]>([]);
  const [selectedShopifyImages, setSelectedShopifyImages] = useState<ShopifyImageChoice[]>([]);
  const [shopifyLoading, setShopifyLoading] = useState(false);
  const [shopifyError, setShopifyError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [outputs, setOutputs] = useState<GeneratedOutput[]>([]);
  const [outputCount, setOutputCount] = useState<1 | 4 | 8 | 9 | 11>(1);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [logsOpen, setLogsOpen] = useState(true);
  const [propsOpen, setPropsOpen] = useState(true);
  const [runpodStatus, setRunpodStatus] = useState<RunPodStatus | null>(null);
  const [statusLoading, setStatusLoading] = useState(true);
  const referenceInput = useRef<HTMLInputElement>(null);

  // ── Poll RunPod status ────────────────────────────────────────────────────

  const fetchStatus = async () => {
    try {
      const res = await apiFetch("/api/generate/status");
      if (res.ok) setRunpodStatus(await res.json());
    } catch {
      setRunpodStatus({ configured: false, ready: false, reason: "API server unreachable." });
    } finally {
      setStatusLoading(false);
    }
  };

  useEffect(() => {
    fetchStatus();
    const id = setInterval(fetchStatus, 30_000);
    return () => clearInterval(id);
  }, []);

  // ── Generate ──────────────────────────────────────────────────────────────

  const handleGenerate = async () => {
    const trimmed = prompt.trim();
    if (!trimmed || !references.length || loading) return;

    // The complete-studio preset describes the whole campaign, but Qwen
    // receives one request per output. Remove the set-level language so it
    // cannot resolve the request by drawing a contact sheet/collage.
    const singleViewPrompt = trimmed
      .replace(/generate a complete premium e-commerce product photography set/gi, "generate one premium e-commerce product photograph")
      .replace(/generate (separate|multiple|all) (supported )?(front|product|camera|individual)[^.]*views?/gi, "")
      .replace(/output requirement:[\s\S]*$/i, "")
      .split("\n")
      .filter((line) => !/^\s*[-•]?\s*(front|back|left side|right side|front three-quarter|rear three-quarter|top|bottom|close-up|macro|product in use|lifestyle)\s*(view|shot)?\s*$/i.test(line))
      .join("\n")
      .trim();
    // Qwen can merge multiple references into a new colorway or accessory
    // variant. The first image is the identity master for this batch.
    const identityMaster = references[0];

    outputs.forEach((output) => URL.revokeObjectURL(output.url));
    setOutputs([]);
    setLoading(true);

    try {
      for (let index = 0; index < outputCount; index += 1) {
        const id = crypto.randomUUID();
        const start = Date.now();
        setLogs((prev) => [{ id, ts: start, prompt: `${trimmed} · output ${index + 1}/${outputCount}`, status: "pending" }, ...prev]);
        const form = new FormData();
        const directionIndex = outputCount === 4
          ? FOUR_ANGLE_INDICES[index]
          : outputCount === 9
            ? NINE_ANGLE_INDICES[index]
            : index;
        const direction = STUDIO_ANGLE_DIRECTIONS[directionIndex];
         form.append("prompt", `Photograph one premium product as a clean standalone luxury retail catalog image. Do not design a product board or presentation. The visual language should match a consistent high-end handbag product page: restrained, polished, quiet, premium, and product-first, with no brand reinterpretation or fashion editorial styling.\n\nPRODUCT IDENTITY AND STYLE:\n${singleViewPrompt}\n\nCAMERA DIRECTION — THIS IS THE ONLY VIEW:\n${direction}\n\nCROSS-ANGLE STYLE AND HARDWARE LOCK:\nUse the exact same product styling, neutral upright/closed pose, scale in frame, camera height, focal length, perspective, warm-white or light-gray background, soft diffused lighting, shadow softness, exposure, white balance, and color calibration as the other images in this set. Change only the camera position required by the requested angle. Do not restyle, reshape, rotate or reposition the product, handles, straps, chains, charms, buckles, or hardware. The chain/strap must stay the exact same length, attachment points, thickness, count, orientation, and resting position; never create a new drape or duplicate loop. If a feature is not visible in the identity master, leave it absent rather than inventing it.\n\nFINAL PHOTOGRAPH:\nShow exactly one physical product, complete and CLOSED in its normal state, centered in one continuous warm-white or light-gray studio background, fully visible with breathing room. The product is the only subject: no second product, color variant, prop, accessory, open section, interior, unfolded part, or extra object. Use the supplied image only to identify the physical item; disregard its arrangement and recreate a clean studio photograph of the same item. Preserve exact shape, proportions, color, material, branding, hardware, stitching, construction, and attached-component placement. Use a clean image with no typography, graphic design elements, borders, decorative marks, or additional objects.`);
        form.append("image", identityMaster.file, identityMaster.file.name);
        const res = await apiFetch("/api/generate", { method: "POST", body: form, signal: AbortSignal.timeout(135_000) });
        const durationMs = Date.now() - start;
        if (!res.ok) {
          let errorMsg = `HTTP ${res.status}`;
          try {
            const json = await res.json();
            if (json?.detail || json?.error) errorMsg = json.detail || json.error;
          } catch { /* non-JSON error body */ }
          setLogs((prev) => prev.map((e) => e.id === id ? { ...e, status: "error", durationMs, error: errorMsg } : e));
          continue;
        }
        const contentType = res.headers.get("content-type") || "image/png";
        const blob = await res.blob();
        if (!blob.type.startsWith("image/") && !contentType.startsWith("image/")) throw new Error("Server returned a non-image response.");
        const url = URL.createObjectURL(blob);
        setOutputs((prev) => [...prev, { url, blob, index }]);
        setLogs((prev) => prev.map((e) => e.id === id ? { ...e, status: "success", durationMs, bytes: blob.size, contentType: blob.type || contentType } : e));
      }
    } catch (err) {
      const errorMsg =
        err instanceof Error
          ? err.name === "TimeoutError" ? "Request timed out (135 s)" : err.message
          : "Unknown error";
      setLogs((prev) => [{ id: crypto.randomUUID(), ts: Date.now(), prompt: trimmed, status: "error", error: errorMsg }, ...prev]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) handleGenerate();
  };

  const handleDownload = (output: GeneratedOutput) => {
    const ext = output.blob.type === "image/jpeg" ? "jpg" : "png";
    const a = document.createElement("a");
    a.href = output.url;
    a.download = `runpod-${output.index + 1}-${Date.now()}.${ext}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };
  const handleDownloadAll = async () => {
    if (!outputs.length) return;
    outputs.forEach((output, downloadIndex) => {
      window.setTimeout(() => handleDownload(output), downloadIndex * 180);
    });
  };

  const clearLogs = () => setLogs([]);
  const chars = prompt.length;
  const chooseReferences = (files: FileList | File[]) => {
    const next = Array.from(files)
      .filter((file) => file.type.startsWith("image/"))
      .slice(0, 3 - references.length)
      .map((file) => ({ file, url: URL.createObjectURL(file) }));
    setReferences((current) => [...current, ...next]);
  };
  const removeReference = (url: string) => {
    URL.revokeObjectURL(url);
    setReferences((current) => current.filter((item) => item.url !== url));
  };
  const loadShopifyProducts = async () => {
    setShopifyLoading(true);
    setShopifyError(null);
    try {
      const response = await apiFetch(`/api/shopify/products?query=${encodeURIComponent(shopifyQuery)}`);
      const body = await response.json();
      if (!response.ok) throw new Error(body?.detail || "Shopify products could not be loaded.");
      setShopifyProducts(Array.isArray(body) ? body : []);
    } catch (error) {
      setShopifyError(error instanceof Error ? error.message : "Shopify products could not be loaded.");
    } finally {
      setShopifyLoading(false);
    }
  };
  const toggleShopifyImage = (choice: ShopifyImageChoice) => {
    setSelectedShopifyImages((current) => {
      const exists = current.some((item) => item.url === choice.url);
      if (exists) return current.filter((item) => item.url !== choice.url);
      const selectedProductHandle = current[0]?.productHandle;
      if (selectedProductHandle && selectedProductHandle !== choice.productHandle) return current;
      return current.length < 3 - references.length ? [...current, choice] : current;
    });
  };
  const importSelectedShopifyImages = async () => {
    const imported = await Promise.all(selectedShopifyImages.map(async (item, index) => {
      const response = await fetch(item.url);
      if (!response.ok) throw new Error(`Could not load ${item.productTitle} image.`);
      const blob = await response.blob();
      const file = new File([blob], `${item.productHandle}-${references.length + index + 1}.jpg`, { type: blob.type || "image/jpeg" });
      return { file, url: URL.createObjectURL(file) };
    }));
    setReferences((current) => [...current, ...imported]);
    setSelectedShopifyImages([]);
    setShopifyOpen(false);
  };

  return (
    <div className="min-h-screen bg-background p-6 md:p-10">
      {/* ── Header ── */}
      <div className="mb-8 flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight mb-1 flex items-center gap-2">
            <Zap className="h-5 w-5 text-primary" />
            RunPod Image Generator
          </h1>
          <p className="text-sm text-muted-foreground">
            Image-to-image via RunPod · all traffic stays server-side
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs font-mono bg-card border border-border rounded-sm px-3 py-2">
          <StatusDot ready={runpodStatus?.ready} loading={statusLoading} />
          <span className="text-muted-foreground">
            {statusLoading
              ? "Checking…"
              : runpodStatus?.ready
              ? "RunPod ready"
              : "RunPod offline"}
          </span>
          <button
            onClick={fetchStatus}
            className="ml-1 text-muted-foreground hover:text-foreground transition-colors"
            title="Refresh status"
          >
            <RefreshCw className="h-3 w-3" />
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-[1fr_380px] gap-6">
        {/* ── Left column: prompt + output ── */}
        <div className="space-y-5">
          {/* Prompt */}
          <div className="bg-card border border-border rounded-sm p-5">
            <label className="block text-[10px] tracking-[.18em] uppercase text-muted-foreground mb-3">
              Source image
            </label>
            <input
              ref={referenceInput}
              type="file"
              multiple
              accept="image/png,image/jpeg,image/webp"
              className="hidden"
              onChange={(e) => {
                if (e.target.files) chooseReferences(e.target.files);
                e.currentTarget.value = "";
              }}
            />
            <button
              type="button"
              onClick={() => referenceInput.current?.click()}
              className={cn(
                "w-full min-h-28 border border-dashed rounded-sm p-3 flex items-center gap-3 text-left transition-colors hover:border-primary",
                references.length ? "border-primary/50 bg-primary/5" : "border-border",
              )}
              disabled={loading}
              data-testid="button-upload-reference"
            >
              {references.length ? (
                <div className="flex -space-x-3 shrink-0">
                  {references.map(({ url }, index) => (
                    <img key={url} src={url} alt={`Source ${index + 1}`} className="h-16 w-16 object-contain bg-background border-2 border-card rounded-sm" />
                  ))}
                </div>
              ) : (
                <ImageIcon className="h-8 w-8 text-muted-foreground/50 ml-2 shrink-0" />
              )}
              <span className="min-w-0">
                <span className="block text-sm">{references.length ? `${references.length} source image${references.length === 1 ? "" : "s"} selected` : "Choose product/source images"}</span>
                <span className="block text-xs text-muted-foreground mt-1">
                  {references.length ? "Identity locked to the first master image" : "Use one original product photo · collage boards are rejected · first image becomes the identity master"}
                </span>
              </span>
            </button>
            {references.length > 0 && (
              <div className="mt-2 flex flex-wrap gap-2">
                {references.map(({ file, url }, index) => (
                  <button key={url} type="button" onClick={() => removeReference(url)} className="text-[10px] text-muted-foreground hover:text-destructive" disabled={loading} data-testid={`button-remove-reference-${index + 1}`}>
                    {file.name} ×
                  </button>
                ))}
              </div>
            )}
            <div className="mt-3 flex items-center justify-between gap-3">
              <button type="button" onClick={() => { setShopifyOpen((value) => !value); if (!shopifyProducts.length) void loadShopifyProducts(); }} className="text-[10px] uppercase tracking-wider text-primary hover:underline" disabled={loading} data-testid="button-generate-shopify-import">
                Import from Shopify
              </button>
              <span className="text-[10px] text-muted-foreground">{references.length} / 3 references</span>
            </div>
            {shopifyOpen && (
              <div className="mt-3 border border-primary/30 bg-primary/5 p-3">
                <div className="flex gap-2">
                  <input value={shopifyQuery} onChange={(event) => setShopifyQuery(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter") void loadShopifyProducts(); }} placeholder="Search Shopify products" className="min-w-0 flex-1 border border-border bg-background px-3 py-2 text-sm outline-none" data-testid="input-generate-shopify-search" />
                  <button type="button" onClick={() => void loadShopifyProducts()} disabled={shopifyLoading} className="border border-primary px-3 text-xs uppercase tracking-wider">{shopifyLoading ? "Loading…" : "Search"}</button>
                </div>
                {shopifyError && <p className="mt-2 text-xs text-destructive">{shopifyError}</p>}
                <div className="mt-3 max-h-72 overflow-y-auto space-y-3">
                  {shopifyProducts.map((product) => (
                    <div key={product.id} className="border border-border bg-background p-2">
                      <div className="mb-2 flex items-center justify-between gap-2">
                        <strong className="truncate text-xs">{product.title}</strong>
                        <span className="shrink-0 text-[10px] text-muted-foreground">{product.images.length} images</span>
                      </div>
                      <div className="grid grid-cols-4 gap-2">
                        {product.images.map((image, index) => {
                          const choice = { url: image.url, productHandle: product.handle, productTitle: product.title, index };
                          const selected = selectedShopifyImages.some((item) => item.url === image.url);
                          return (
                            <button type="button" key={image.url} onClick={() => toggleShopifyImage(choice)} disabled={loading || (!selected && selectedShopifyImages.length >= 3 - references.length)} className={cn("relative aspect-square overflow-hidden border-2 bg-secondary", selected ? "border-primary" : "border-transparent hover:border-primary/50")} data-testid={`button-select-shopify-image-${product.handle}-${index + 1}`}>
                              <img src={image.url} alt={image.alt || `${product.title} image ${index + 1}`} className="h-full w-full object-contain" />
                              <span className={cn("absolute right-1 top-1 flex h-4 w-4 items-center justify-center text-[10px]", selected ? "bg-primary text-primary-foreground" : "bg-background/80 text-muted-foreground")}>{selected ? "✓" : index + 1}</span>
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  ))}
                  {!shopifyLoading && !shopifyError && !shopifyProducts.length && <p className="text-xs text-muted-foreground">No Shopify products found.</p>}
                </div>
                {selectedShopifyImages.length > 0 && (
                  <div className="mt-3 flex items-center justify-between gap-2 border-t border-border pt-3">
                    <span className="text-xs text-muted-foreground">{selectedShopifyImages.length} selected · same product only</span>
                    <button type="button" onClick={() => void importSelectedShopifyImages()} disabled={loading} className="border border-primary bg-primary px-3 py-1.5 text-[10px] uppercase tracking-wider text-primary-foreground" data-testid="button-import-selected-shopify-images">Import selected</button>
                  </div>
                )}
              </div>
            )}
            <label className="block text-[10px] tracking-[.18em] uppercase text-muted-foreground mb-3">
              Prompt
            </label>
            <div className="mb-3 flex flex-wrap gap-2">
              {[
                ["Complete studio set", COMPLETE_STUDIO_SET_PROMPT],
                ["Catalog", "Create a clean premium catalog image with the product centered, fully visible, accurate to every reference detail, soft studio lighting, and a warm neutral background."],
                ["Editorial", "Create a refined editorial product image with a distinctive camera angle, considered composition, natural shadows, and luxury art direction while preserving the exact product identity from the references."],
                ["Detail", "Create a close product detail image highlighting material, stitching, hardware, and construction visible in the references. Do not invent details or alter the product."],
              ].map(([label, value]) => (
                <button key={label} type="button" onClick={() => setPrompt(value)} disabled={loading} className="border border-border px-2.5 py-1 text-[10px] uppercase tracking-wider hover:border-primary" data-testid={`button-prompt-preset-${label.toLowerCase().replaceAll(" ", "-")}`}>
                  {label}
                </button>
              ))}
            </div>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              onKeyDown={handleKeyDown}
              rows={5}
              placeholder="Describe how to transform the source image…"
              disabled={loading}
              className={cn(
                "w-full resize-none bg-background border rounded-sm px-3 py-2.5 text-sm focus:outline-none focus:ring-1 focus:ring-primary transition-colors font-mono placeholder:font-sans placeholder:text-muted-foreground",
                "border-input",
                loading && "opacity-60 cursor-not-allowed",
              )}
            />
            <div className="mt-2 flex items-center justify-between">
              <p className="text-[11px] text-muted-foreground">
                ⌘ Enter to generate · source image + prompt required
              </p>
              <span className="text-[11px] font-mono tabular-nums text-muted-foreground">
                {chars} characters
              </span>
            </div>
            <div className="mt-4 flex items-center justify-between border-t border-border pt-3">
              <span className="text-[10px] tracking-[.18em] uppercase text-muted-foreground">Outputs</span>
              <div className="flex gap-1">
                {([1, 4, 8, 9, 11] as const).map((count) => (
                  <button key={count} type="button" onClick={() => setOutputCount(count)} disabled={loading} aria-pressed={outputCount === count} className={cn("border px-3 py-1.5 text-xs", outputCount === count ? "border-primary bg-primary text-primary-foreground" : "border-border hover:border-primary")} data-testid={`button-output-count-${count}`}>
                    {count}
                  </button>
                ))}
              </div>
              <span className="text-[10px] text-muted-foreground">{outputCount === 1 ? "single image" : `${outputCount}-image batch`}</span>
            </div>
            <div className="mt-4 flex gap-2">
              <Button
                onClick={handleGenerate}
                disabled={loading || !prompt.trim() || !references.length}
                className="gap-2"
              >
                {loading ? (
                  <><Loader2 className="h-4 w-4 animate-spin" /> Generating…</>
                ) : (
                  <><Sparkles className="h-4 w-4" /> Generate image</>
                )}
              </Button>
              {loading && (
                <p className="text-xs text-muted-foreground self-center animate-pulse">
                  GPU generation · up to 120 s
                </p>
              )}
            </div>
          </div>

          {/* Output */}
          <div className="bg-card border border-border rounded-sm overflow-hidden">
            <div className="flex items-center justify-between px-5 py-3 border-b border-border">
              <span className="text-[10px] tracking-[.18em] uppercase text-muted-foreground flex items-center gap-2">
                <ImageIcon className="h-3.5 w-3.5" /> Output
              </span>
              {outputs.length > 0 && (
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-muted-foreground">{outputs.length} ready</span>
                  {outputs.length > 1 && (
                    <Button size="sm" variant="default" className="h-7 gap-1.5 text-xs" onClick={() => void handleDownloadAll()} data-testid="button-download-all">
                      <Download className="h-3.5 w-3.5" /> Download all images
                    </Button>
                  )}
                  {outputs.map((output) => (
                    <Button key={output.url} size="sm" variant="ghost" className="h-7 gap-1.5 text-xs" onClick={() => handleDownload(output)}>
                      <Download className="h-3.5 w-3.5" /> {output.index + 1}
                    </Button>
                  ))}
                </div>
              )}
            </div>
            <div className={cn(
              "flex items-center justify-center min-h-[340px] bg-muted/30",
              loading && "relative",
            )}>
              {loading && (
                <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-background/60 backdrop-blur-sm z-10">
                  <Loader2 className="h-8 w-8 animate-spin text-primary" />
                  <p className="text-sm text-muted-foreground font-mono">Waiting for RunPod…</p>
                </div>
              )}
              {outputs.length ? (
                <div className={cn("grid w-full gap-4 p-4", outputs.length > 1 ? "grid-cols-2 md:grid-cols-4" : "grid-cols-1")}>
                  {outputs.map((output) => (
                    <div key={output.url} className="relative aspect-square bg-background border border-border">
                      <img src={output.url} alt={`${prompt} output ${output.index + 1}`} className="h-full w-full object-contain" />
                    </div>
                  ))}
                </div>
              ) : !loading ? (
                <div className="flex flex-col items-center gap-2 text-muted-foreground py-16">
                  <ImageIcon className="h-10 w-10 opacity-20" />
                  <p className="text-sm">Your generated image appears here</p>
                </div>
              ) : null}
            </div>
          </div>

          {/* Request Logs */}
          <div className="bg-card border border-border rounded-sm overflow-hidden">
            <button
              className="w-full flex items-center justify-between px-5 py-3 border-b border-border hover:bg-muted/30 transition-colors"
              onClick={() => setLogsOpen((v) => !v)}
            >
              <span className="text-[10px] tracking-[.18em] uppercase text-muted-foreground flex items-center gap-2">
                <Terminal className="h-3.5 w-3.5" />
                Request logs
                {logs.length > 0 && (
                  <span className="ml-1 text-[10px] font-mono bg-muted px-1.5 py-0.5 rounded-sm">
                    {logs.length}
                  </span>
                )}
              </span>
              <div className="flex items-center gap-3">
                {logs.length > 0 && (
                  <button
                    onClick={(e) => { e.stopPropagation(); clearLogs(); }}
                    className="text-[10px] text-muted-foreground hover:text-foreground transition-colors font-mono"
                  >
                    clear
                  </button>
                )}
                {logsOpen ? <ChevronUp className="h-4 w-4 text-muted-foreground" /> : <ChevronDown className="h-4 w-4 text-muted-foreground" />}
              </div>
            </button>
            {logsOpen && (
              <div className="max-h-64 overflow-y-auto">
                {logs.length === 0 ? (
                  <div className="flex items-center gap-2 px-5 py-6 text-xs text-muted-foreground">
                    <Clock className="h-3.5 w-3.5" />
                    <span>No requests yet — generate an image to see logs.</span>
                  </div>
                ) : (
                  logs.map((entry) => <LogRow key={entry.id} entry={entry} />)
                )}
              </div>
            )}
          </div>
        </div>

        {/* ── Right column: properties panel ── */}
        <div className="space-y-5">
          {/* RunPod connection */}
          <div className="bg-card border border-border rounded-sm p-5">
            <div className="text-[10px] tracking-[.18em] uppercase text-muted-foreground mb-3 flex items-center gap-2">
              <Activity className="h-3.5 w-3.5" />
              RunPod connection
            </div>
            <div className="space-y-1">
              <div className="flex items-center gap-2 text-sm">
                <StatusDot ready={runpodStatus?.ready} loading={statusLoading} />
                <span className={runpodStatus?.ready ? "text-emerald-600 font-medium" : "text-muted-foreground"}>
                  {statusLoading ? "Checking…" : runpodStatus?.ready ? "Connected & ready" : "Not connected"}
                </span>
              </div>
              {runpodStatus && (
                <p className="text-xs text-muted-foreground mt-2 font-mono leading-relaxed">{runpodStatus.reason}</p>
              )}
            </div>
          </div>

          {/* API properties */}
          <div className="bg-card border border-border rounded-sm overflow-hidden">
            <button
              className="w-full flex items-center justify-between px-5 py-3 border-b border-border hover:bg-muted/30 transition-colors"
              onClick={() => setPropsOpen((v) => !v)}
            >
              <span className="text-[10px] tracking-[.18em] uppercase text-muted-foreground flex items-center gap-2">
                <Info className="h-3.5 w-3.5" />
                API properties
              </span>
              {propsOpen ? <ChevronUp className="h-4 w-4 text-muted-foreground" /> : <ChevronDown className="h-4 w-4 text-muted-foreground" />}
            </button>

            {propsOpen && (
              <div className="px-5 py-4 space-y-5 text-xs">
                {/* Request */}
                <div>
                  <p className="text-[10px] uppercase tracking-[.15em] text-muted-foreground mb-2">Request</p>
                  <div className="divide-y divide-border/40">
                    <PropRow label="Method" value="POST" mono />
                    <PropRow label="Endpoint" value="/api/generate" mono />
                    <PropRow label="Content-Type" value="multipart/form-data" mono />
                    <PropRow label="Body" value="prompt + 1–3 image files" mono />
                    <PropRow label="Prompt min" value="1 character" />
                    <PropRow label="Prompt max" value="No client-side limit" />
                    <PropRow label="Timeout" value="130 seconds" badge="GPU" />
                  </div>
                </div>

                {/* Response */}
                <div>
                  <p className="text-[10px] uppercase tracking-[.15em] text-muted-foreground mb-2">Response</p>
                  <div className="divide-y divide-border/40">
                    <PropRow label="Content-Type" value="image/png · image/jpeg" mono />
                    <PropRow label="Format" value="Binary image bytes" />
                    <PropRow label="Not JSON" value="Read with response.blob()" mono />
                    <PropRow label="Max size" value="32 MB" />
                  </div>
                </div>

                {/* Status codes */}
                <div>
                  <p className="text-[10px] uppercase tracking-[.15em] text-muted-foreground mb-2">Status codes</p>
                  <div className="divide-y divide-border/30">
                    <StatusCodeRow code="200" meaning="Image generated successfully" />
                    <StatusCodeRow code="400" meaning="Invalid prompt or source image" />
                    <StatusCodeRow code="502" meaning="RunPod / image server failure" />
                    <StatusCodeRow code="503" meaning="RUNPOD_URL not configured" />
                    <StatusCodeRow code="504" meaning="Server unreachable or timed out" />
                  </div>
                </div>

                {/* Architecture */}
                <div>
                  <p className="text-[10px] uppercase tracking-[.15em] text-muted-foreground mb-2">Architecture</p>
                  <div className="font-mono text-[11px] text-muted-foreground bg-muted/40 rounded-sm p-3 leading-6 space-y-0.5">
                    <p>Browser</p>
                    <p className="pl-3">↓ POST /api/generate</p>
                    <p>Atelier API server</p>
                    <p className="pl-3">↓ POST {"{RUNPOD_URL}"} (multipart)</p>
                    <p>RunPod FLUX pod</p>
                    <p className="pl-3">↓ image/png bytes</p>
                    <p>Browser receives image</p>
                  </div>
                  <p className="text-[10px] text-muted-foreground mt-2">
                    RUNPOD_URL stays server-side · never exposed to the browser
                  </p>
                </div>

                {/* Status endpoint */}
                <div>
                  <p className="text-[10px] uppercase tracking-[.15em] text-muted-foreground mb-2">Status check</p>
                  <div className="divide-y divide-border/40">
                    <PropRow label="Method" value="GET" mono />
                    <PropRow label="Endpoint" value="/api/generate/status" mono />
                    <PropRow label="Returns" value='{ "configured": bool, "ready": bool }' mono />
                  </div>
                </div>

                {/* cURL */}
                <div>
                  <p className="text-[10px] uppercase tracking-[.15em] text-muted-foreground mb-2">cURL test</p>
                  <pre className="text-[10px] font-mono bg-muted/40 rounded-sm p-3 leading-5 overflow-x-auto whitespace-pre-wrap break-all text-muted-foreground">
{`curl -X POST /api/generate \\
  -F "prompt=Place the product in soft studio light" \\
  -F "image=@reference-1.png" \\
  -F "image=@reference-2.png" \\
  --output out.png`}
                  </pre>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
