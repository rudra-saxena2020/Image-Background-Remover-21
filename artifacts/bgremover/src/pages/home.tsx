import { useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import { Link } from "wouter";
import { useCreateShoot, useGetShoot, useCancelShoot, getGetShootQueryKey } from "@workspace/api-client-react";
import { apiFetch, apiUrl } from "@/lib/api";
import type { Shoot, ShootShot, CreateShootInput } from "@workspace/api-client-react";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { useToast } from "@/hooks/use-toast";
import { cn } from "@/lib/utils";
import { ArrowUpRight, Check, ChevronRight, CircleAlert, Download, FileArchive, ImagePlus, LockKeyhole, MoreHorizontal, Pause, Play, RefreshCw, RotateCcw, Sparkles, X, Zap } from "lucide-react";
import refHero from "@assets/P01187465_1785974470587.webp";
import refOne from "@assets/P01187465_b1_1785974470642.webp";
import refTwo from "@assets/P01187465_b2_1785974470662.webp";
import refDetail from "@assets/P01187465_d4_1785974470729.webp";

type LocalReference = { id: string; file?: File; url: string; name: string };
type ShopifyProduct = {
  id: string;
  title: string;
  handle: string;
  product_type: string;
  vendor: string;
  images: { url: string; alt: string }[];
};
type ShootStatus = "idle" | "creating" | "running" | "done" | "error";
type CampaignFormat = "flexible-8" | "front-back-7" | "compact-6";
type GenerationMode = "product-only" | "human-model";
type GenerationStatus = {
  id?: string;
  name?: string;
  mode?: string;
  cpu_available?: boolean;
  ready?: boolean;
  configured?: boolean;
  repository_present?: boolean;
  runner_present?: boolean;
  model_present?: boolean;
  cuda_available?: boolean;
  device?: string;
  ae_present?: boolean;
  checkpoint_present?: boolean;
  quantized?: boolean;
  runtime_importable?: boolean;
  runtime_supported?: boolean;
  worker_ready?: boolean;
  runtime_ready?: boolean;
  human_product_verified?: boolean;
  verification_state?: "unavailable" | "unverified" | "failed" | "verified";
  registry_status?: "unavailable" | "requires_larger_gpu" | "failed" | "unverified" | "online" | "verified";
  capabilities?: string[];
  installed?: boolean;
  runtime_reachable?: boolean;
  worker_state?: "reachable-empty" | "provider-failed" | "inference-ready-unverified" | "verified" | "unavailable";
  provider?: string | null;
  gpu_available?: boolean;
  model_loaded?: boolean;
  inference_passed?: boolean;
  product_validation_passed?: boolean;
  human_model_passed?: boolean;
  verified?: boolean;
  last_test_time?: string | null;
  last_error?: string | null;
  next_action?: string;
  verification?: {
    status?: "not-run" | "blocked" | "failed" | "passed" | "stale";
    passed?: boolean;
    current?: boolean;
    reason?: string;
    checked_at?: string | null;
    expires_at?: string | null;
    latency_ms?: number | null;
  };
  reason?: string;
};
type EngineId = "colab" | "flux2-pro" | "bfl-flux2" | "qwen-runpod" | "flux1-runpod" | "gemini-image" | "cpu";
type AvailableEngine = {
  id: "colab" | "flux2-pro" | "bfl-flux2" | "qwen-runpod" | "flux1-runpod" | "gemini-image";
  label: string;
  provider?: string | null;
  status?: GenerationStatus;
};
type GenerationOptions = {
  remote_worker?: GenerationStatus;
  qwen?: GenerationStatus;
  flux_schnell?: GenerationStatus;
  fooocus?: GenerationStatus;
  hidream?: GenerationStatus;
  flux2?: GenerationStatus;
  flux2_pro?: GenerationStatus;
  black_forest_flux2?: GenerationStatus;
  qwen_runpod?: GenerationStatus;
  flux1_dev_runpod?: GenerationStatus;
  gemini_image?: GenerationStatus;
  flux2_klein?: GenerationStatus;
  sdxl?: GenerationStatus;
  reference_preview?: GenerationStatus;
  available_engines?: AvailableEngine[];
};
const starterReferences: LocalReference[] = [
  { id: "ref-hero", url: refHero, name: "P01187465_hero.webp" },
  { id: "ref-b1", url: refOne, name: "P01187465_b1.webp" },
  { id: "ref-b2", url: refTwo, name: "P01187465_b2.webp" },
  { id: "ref-d4", url: refDetail, name: "P01187465_detail.webp" },
];
const shotBlueprint = [
  ["01", "Studio product", "Shopify primary image"],
  ["02", "Model carrying", "Product in natural use"],
  ["03", "Editorial campaign", "Movement and fashion story"],
  ["04", "Craftsmanship macro", "Leather, hardware and texture"],
  ["05", "Alternative perspective", "Rear, side or 45° view"],
  ["06", "Lifestyle image", "Natural movement in luxury setting"],
  ["07", "Luxury detail", "Handle, logo, interior or hardware"],
  ["08", "Hero campaign", "Homepage banner and brand story"],
];

function readableError(error: unknown) {
  const candidate = error as { response?: { status?: number; data?: { detail?: string; error?: string } }; message?: string };
  const status = candidate?.response?.status;
  const detail = candidate?.response?.data?.detail || candidate?.response?.data?.error || candidate?.message;
  return status ? `${status} — ${detail || "The studio could not complete this request."}` : detail || "The studio could not complete this request.";
}

function generationMessage(generation: GenerationStatus | null) {
  if (!generation) return "Checking the authenticated Colab worker.";
  return generation.reason || generation.next_action || (
    generation.ready
      ? "Connected to a freshly verified Colab GPU worker."
      : "The Colab worker is unavailable or needs a fresh human-with-product verification."
  );
}

function registryStatusLabel(status?: GenerationStatus) {
  if (!status) return "Checking";
  if (status.mode === "hosted-paid" && status.ready) return "Available · paid";
  if (status.registry_status === "requires_larger_gpu") return "Requires larger GPU";
  if (status.registry_status === "verified" || status.ready) return "Verified";
  if (status.registry_status === "online") return "Online · audit needed";
  if (status.registry_status === "failed" || status.verification_state === "failed") return "Failed";
  if (status.registry_status === "unavailable" || status.verification_state === "unavailable") return "Unavailable";
  return "Unverified";
}

function registryStatusClass(status?: GenerationStatus) {
  return status?.ready || status?.registry_status === "verified"
    ? "text-accent-foreground"
    : status?.registry_status === "failed"
      ? "text-destructive"
      : "text-muted-foreground";
}

export function Home() {
  const [productName, setProductName] = useState("P01187465");
  const [category, setCategory] = useState("Handbags");
  const [atmosphere, setAtmosphere] = useState("Quiet afternoon light");
  const [background, setBackground] = useState("Warm mineral plaster");
  const [output, setOutput] = useState<"png" | "jpeg" | "webp">("jpeg");
  const [speedMode, setSpeedMode] = useState<"fast" | "campaign">("fast");
  const [campaignFormat, setCampaignFormat] = useState<CampaignFormat>("flexible-8");
  const [generationMode, setGenerationMode] = useState<GenerationMode>("product-only");
  const [engine, setEngine] = useState<EngineId>("qwen-runpod");
  const [references, setReferences] = useState<LocalReference[]>(starterReferences);
  const [modelMaster, setModelMaster] = useState<LocalReference | null>(null);
  const [shopifyOpen, setShopifyOpen] = useState(false);
  const [shopifyQuery, setShopifyQuery] = useState("");
  const [shopifyProducts, setShopifyProducts] = useState<ShopifyProduct[]>([]);
  const [selectedShopifyProducts, setSelectedShopifyProducts] = useState<string[]>([]);
  const [shopifyLoading, setShopifyLoading] = useState(false);
  const [shopifyError, setShopifyError] = useState<string | null>(null);
  const [locked, setLocked] = useState(true);
  const [shootId, setShootId] = useState<string | null>(null);
  const [shootStatus, setShootStatus] = useState<ShootStatus>("idle");
  const [localError, setLocalError] = useState<string | null>(null);
  const [selectedShot, setSelectedShot] = useState<string | null>(null);
  const [generation, setGeneration] = useState<GenerationOptions | null>(null);
  const [generationLoading, setGenerationLoading] = useState(true);
  const [generationRefreshKey, setGenerationRefreshKey] = useState(0);
  const [progressNow, setProgressNow] = useState(() => Date.now());
  const inputRef = useRef<HTMLInputElement>(null);
  const modelInputRef = useRef<HTMLInputElement>(null);
  const shotProgressStartedAt = useRef<Record<string, { progress: number; at: number }>>({});
  const { toast } = useToast();

  const loadShopifyProducts = async () => {
    setShopifyLoading(true);
    setShopifyError(null);
    try {
      const response = await apiFetch(`/api/shopify/products?query=${encodeURIComponent(shopifyQuery)}`);
      const body = await response.json().catch(() => null);
      if (!response.ok) throw new Error(body?.detail || "Shopify products could not be loaded.");
      setShopifyProducts(Array.isArray(body) ? body as ShopifyProduct[] : []);
    } catch (error) {
      setShopifyError(error instanceof Error ? error.message : "Shopify products could not be loaded.");
    } finally {
      setShopifyLoading(false);
    }
  };

  const toggleShopifyProduct = (productId: string) => {
    setSelectedShopifyProducts((current) => current.includes(productId)
      ? current.filter((id) => id !== productId)
      : [...current, productId]);
  };

  const importSelectedShopifyProducts = () => {
    const selected = shopifyProducts.filter((product) => selectedShopifyProducts.includes(product.id));
    const seen = new Set<string>();
    const imported = selected.flatMap((product) => product.images.map((image, index) => ({
      id: `shopify-${product.id}-${index}`,
      url: image.url,
      name: image.alt || `${product.handle || product.title}-${index + 1}.jpg`,
    }))).filter((image) => {
      if (seen.has(image.url)) return false;
      seen.add(image.url);
      return true;
    }).slice(0, 6);
    if (!imported.length) {
      setShopifyError("Select at least one Shopify product with usable product images.");
      return;
    }
    const first = selected[0];
    setProductName(selected.length === 1 ? first.title : `${first.title} + ${selected.length - 1} more`);
    if (first.product_type) setCategory(first.product_type);
    setReferences(imported);
    setShopifyOpen(false);
    setSelectedShopifyProducts([]);
    toast({ title: `${selected.length} Shopify product${selected.length === 1 ? "" : "s"} imported`, description: `${imported.length} reference images ready.` });
  };

  const createShoot = useCreateShoot();
  const cancelShoot = useCancelShoot();
  const isLive = shootId && (shootStatus === "creating" || shootStatus === "running");
  const shootQuery = useGetShoot(shootId || "", {
    query: { enabled: Boolean(shootId), queryKey: getGetShootQueryKey(shootId || ""), refetchInterval: isLive ? 1000 : false },
  });
  const shoot = shootQuery.data as Shoot | undefined;
  const status = shoot?.status?.toLowerCase() ?? "";
  const terminal = ["completed", "done", "failed", "cancelled", "canceled", "error"].includes(status);
  const activeGeneration = generation?.remote_worker;
  const selectedProvider = engine === "flux2-pro"
    ? generation?.flux2_pro
    : engine === "bfl-flux2"
      ? generation?.black_forest_flux2
      : engine === "qwen-runpod"
        ? generation?.qwen_runpod
        : engine === "flux1-runpod"
          ? generation?.flux1_dev_runpod
          : engine === "gemini-image"
            ? generation?.gemini_image
      : activeGeneration;
  const availableEngine = generation?.available_engines?.find((choice) => choice.id === engine);
  const engineReady = engine === "cpu" || availableEngine?.status?.ready === true;
  const cpuCampaignBlocked = generationMode === "human-model" && engine === "cpu" && speedMode === "campaign";
  const strictFrontBack = campaignFormat === "front-back-7";
  const activeReferences = strictFrontBack ? references.slice(0, 2) : references;
  const runpodReferenceLimitExceeded = engine === "qwen-runpod" && activeReferences.length > 3;
  const canStartShoot = engineReady && !cpuCampaignBlocked && !runpodReferenceLimitExceeded;
  const selectedEngineLabel = engine === "cpu"
      ? "Local reference-locked preview"
      : availableEngine?.label || "Selected provider";
  const plannedFrameCount = strictFrontBack ? 7 : campaignFormat === "compact-6" ? 6 : speedMode === "fast" ? 1 : 8;
  const estimatedProviderCost = engine === "qwen-runpod"
    ? plannedFrameCount * 0.02
    : engine === "flux1-runpod"
      ? plannedFrameCount * 0.02097152
      : null;
  const unavailableMessage = runpodReferenceLimitExceeded
      ? "RunPod Qwen Image Edit accepts up to three source references. Remove extra images before generating."
      : engine === "cpu"
      ? "Uses an uploaded model-carrying reference and preserves the exact product locally."
        : generationMessage(selectedProvider ?? null);
  const engineChoices: Array<{ id: EngineId; label: string; ready: boolean }> = [
    ...(generation?.available_engines || []).map((choice) => ({
      id: choice.id as EngineId,
      label: choice.label,
      ready: choice.status?.ready === true,
    })),
    ...(!generation?.available_engines?.some((choice) => choice.id === "qwen-runpod")
      ? [{
          id: "qwen-runpod" as const,
          label: "RunPod image generator · Qwen Edit",
          ready: generation?.qwen_runpod?.ready === true,
        }]
      : []),
    { id: "cpu", label: "Reference-locked preview", ready: true },
  ];
  const backendAudit = [
    generation?.remote_worker,
    generation?.flux2_pro,
    generation?.black_forest_flux2,
    generation?.qwen_runpod,
    generation?.flux1_dev_runpod,
    generation?.gemini_image,
    generation?.reference_preview,
  ].filter(Boolean) as GenerationStatus[];

  const statusForEngine = (id: string) => {
    if (id === "cpu") return generation?.reference_preview;
    if (id === "flux2-pro") return generation?.flux2_pro;
    if (id === "bfl-flux2") return generation?.black_forest_flux2;
    if (id === "qwen-runpod") return generation?.qwen_runpod;
    if (id === "flux1-runpod") return generation?.flux1_dev_runpod;
    if (id === "gemini-image") return generation?.gemini_image;
    return generation?.remote_worker;
  };

  useEffect(() => {
    const hasPersistentProviderOption = engine === "qwen-runpod";
    if (!generationLoading && engine !== "cpu" && !hasPersistentProviderOption && !generation?.available_engines?.some((choice) => choice.id === engine)) {
      setEngine("cpu");
    }
  }, [generationLoading, generation?.available_engines, engine]);

  useEffect(() => {
    if (!shoot) return;
    if (terminal) {
      setShootStatus(status === "completed" || status === "done" ? "done" : "error");
      if (shoot.error) setLocalError(shoot.error);
    } else setShootStatus("running");
  }, [shoot, terminal, status]);

  const shots = useMemo(() => {
    const source = shoot?.shots?.length ? shoot.shots : [];
    if (source.length) return source;
    return shotBlueprint.map(([number, title, purpose], index) => source[index] ?? ({ id: `planned-${index}`, number: Number(number), title, purpose, kind: title, status: "planned", verification: "pending" } as ShootShot));
  }, [shoot]);

  useEffect(() => {
    const now = Date.now();
    for (const shot of shots) {
      const backendProgress = shot.progress ?? (shot.image_url ? 100 : 0);
      const tracked = shotProgressStartedAt.current[shot.id];
      if (!tracked || tracked.progress !== backendProgress) {
        shotProgressStartedAt.current[shot.id] = { progress: backendProgress, at: now };
      }
    }
  }, [shots]);

  useEffect(() => {
    if (!isLive) return;
    const timer = window.setInterval(() => setProgressNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [isLive]);

  const getShotDisplayProgress = (shot: ShootShot) => {
    const backendProgress = Math.max(0, Math.min(100, shot.progress ?? (shot.image_url ? 100 : 0)));
    if (!isLive || shot.status !== "processing" || backendProgress >= 100) {
      return { value: backendProgress, estimated: false };
    }
    const tracked = shotProgressStartedAt.current[shot.id];
    const stage = (shoot?.stage || "").toLowerCase();
    const longGenerationStage = stage.includes("quality pass")
      && !stage.includes("generation complete")
      && !stage.includes("removing background")
      && !stage.includes("validating output");
    if (!tracked || !longGenerationStage) {
      return { value: backendProgress, estimated: false };
    }
    const target = Math.min(95, backendProgress + 15);
    const estimatedDurationMs = shoot?.remote_model === "sdxl" ? 180_000 : 60_000;
    const elapsed = Math.max(0, progressNow - tracked.at);
    const fraction = Math.min(1, elapsed / estimatedDurationMs);
    return {
      value: Math.round(backendProgress + ((target - backendProgress) * fraction)),
      estimated: true,
    };
  };

  useEffect(() => {
    let mounted = true;
    const loadGenerationStatus = async () => {
      try {
        const response = await apiFetch("/api/health");
        if (!response.ok) throw new Error("Generation status unavailable");
        const payload = await response.json() as { generation?: GenerationOptions };
        if (mounted) setGeneration(payload.generation ?? {});
      } catch {
        if (mounted) setGeneration((current) => current ?? {});
      } finally {
        if (mounted) setGenerationLoading(false);
      }
    };
    void loadGenerationStatus();
     const interval = window.setInterval(() => void loadGenerationStatus(), 3000);
    return () => {
      mounted = false;
      window.clearInterval(interval);
    };
  }, [generationRefreshKey]);

  const addFiles = (files: FileList | File[]) => {
    const referenceLimit = strictFrontBack ? 2 : 6;
    const accepted = Array.from(files).slice(0, referenceLimit - references.length).filter((file) => file.type.startsWith("image/"));
    setReferences((current) => [...current, ...accepted.map((file) => ({ id: `${file.name}-${file.lastModified}`, file, url: URL.createObjectURL(file), name: file.name }))]);
    setLocalError(null);
  };
  const removeReference = (id: string) => {
    setReferences((current) => {
      const reference = current.find((item) => item.id === id);
      if (reference?.file) URL.revokeObjectURL(reference.url);
      return current.filter((ref) => ref.id !== id);
    });
  };
  const setModelMasterFile = (file: File | undefined) => {
    if (!file || !file.type.startsWith("image/")) return;
    setModelMaster((current) => {
      if (current?.file) URL.revokeObjectURL(current.url);
      return {
        id: `model-master-${file.name}-${file.lastModified}`,
        file,
        url: URL.createObjectURL(file),
        name: file.name,
      };
    });
    setLocalError(null);
  };
  const removeModelMaster = () => {
    setModelMaster((current) => {
      if (current?.file) URL.revokeObjectURL(current.url);
      return null;
    });
  };
  useEffect(() => {
    return () => references.forEach((reference) => {
      if (reference.file) URL.revokeObjectURL(reference.url);
    });
  }, []);
  const startShoot = async () => {
    if (!activeReferences.length || !productName.trim() || !locked) return;
    if (generationMode === "human-model" && !modelMaster) {
      setLocalError("Human-model generation requires a dedicated Model Master image.");
      return;
    }
    if (strictFrontBack && activeReferences.length !== 2) {
      setLocalError("Front/back campaign needs exactly two references: one front view and one back view.");
      return;
    }
    if (!canStartShoot) {
      if (cpuCampaignBlocked) {
        setLocalError("Full campaign mode needs a verified Colab human-generation model. Switch to Fast preview for the source-preserved path.");
      }
      else if (!engineReady) {
      setLocalError(`${selectedEngineLabel} is not ready on this machine.`);
      }
      return;
    }
    setShootStatus("creating"); setLocalError(null);
    try {
      const files = await Promise.all(activeReferences.map(async (reference) => {
        if (reference.file) return reference.file;
        const response = await fetch(reference.url);
        if (!response.ok) throw new Error(`Could not load ${reference.name}`);
        const blob = await response.blob();
        return new File([blob], reference.name, { type: blob.type || "image/webp" });
      }));
       const input: CreateShootInput = {
        product_name: productName, category, atmosphere, background, output_format: output,
        speed_mode: speedMode,
         campaign_format: campaignFormat,
         generation_mode: generationMode,
        engine,
          model_reference: generationMode === "human-model" && modelMaster?.file ? modelMaster.file : undefined,
         references: files,
      };
      createShoot.mutate({ data: input }, { onSuccess: (created) => { setShootId(created.id); setShootStatus("running"); }, onError: (error) => { setShootStatus("error"); setLocalError(readableError(error)); } });
    } catch (error) {
      setShootStatus("error");
      setLocalError(error instanceof Error ? error.message : "The studio could not prepare the references.");
    }
  };
  const resetStudio = () => { setShootId(null); setShootStatus("idle"); setLocalError(null); };
  const retryShoot = () => {
    setShootId(null);
    setShootStatus("idle");
    setLocalError(null);
    window.setTimeout(() => void startShoot(), 0);
  };
  const downloadFrame = async (shot: ShootShot) => {
    if (!shot.image_url) return;
    const a = document.createElement("a"); a.href = apiUrl(shot.image_url); a.download = `${productName}_${String(shot.number).padStart(2, "0")}.${output}`; a.target = "_blank"; a.click();
  };
  const downloadAll = async () => {
    if (!shootId) return;
    try {
      const response = await apiFetch(`/api/shoots/${shootId}/export`);
      if (!response.ok) {
        const body = await response.json().catch(() => null);
        throw new Error(body?.detail || `Export failed (${response.status})`);
      }
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `${productName}_atelier_export.zip`;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(url);
      toast({ title: "Export ready", description: "The validated frames and Shopify manifest are downloading." });
    } catch (error) {
      setLocalError(error instanceof Error ? error.message : "The export could not be prepared.");
    }
  };

  if (isLive || shootStatus === "done" || (shootStatus === "error" && shoot)) {
    const completed = shots.filter((shot) => Boolean(shot.image_url)).length;
    const usedReferenceLock = Boolean(shoot?.shots?.some((shot) => shot.verification?.startsWith("reference-locked")));
    return <StudioShell eyebrow={shoot?.product_name ?? productName} onReset={resetStudio}>
      <div className="animate-rise max-w-[1500px] mx-auto px-5 py-8 md:px-10 md:py-12">
        <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-6 mb-10">
          <div><div className="font-mono text-[10px] tracking-[.2em] uppercase text-muted-foreground mb-4">Shoot / {shootId?.slice(0, 8)}</div>
            <h1 className="font-display text-5xl md:text-7xl tracking-tight leading-[.85]">The quiet<br /><i>campaign.</i></h1>
            <p className="mt-5 text-sm text-muted-foreground max-w-md">{shoot?.atmosphere || atmosphere} · {shoot?.background || background}</p>
          </div>
          <div className="flex items-center gap-3">
            {shootStatus === "done" && <Button variant="outline" onClick={downloadAll} disabled={!completed} data-testid="button-download-all"><FileArchive className="h-4 w-4" /> Download all</Button>}
            {shootStatus === "error" && <Button variant="outline" onClick={retryShoot} disabled={createShoot.isPending} data-testid="button-retry-shoot"><RefreshCw className="h-4 w-4" /> Retry shoot</Button>}
            {isLive && <Button variant="outline" onClick={() => shootId && cancelShoot.mutate({ shootId }, { onSuccess: () => setShootStatus("error"), onError: (error) => setLocalError(readableError(error)) })} data-testid="button-cancel-shoot"><Pause className="h-4 w-4" /> Stop shoot</Button>}
            <Button onClick={resetStudio} variant="secondary" data-testid="button-new-shoot"><RotateCcw className="h-4 w-4" /> New shoot</Button>
          </div>
        </div>
        {localError && <ErrorBanner message={localError} onRetry={shootStatus === "error" && shoot ? retryShoot : resetStudio} />}
        <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_350px] gap-10">
          <div>
            <div className="flex items-center justify-between mb-3"><div className="font-mono text-[10px] tracking-[.18em] uppercase text-muted-foreground">{shootStatus === "done" ? "Review / validated output" : `${shoot?.stage || "Building output"} / ${shoot?.progress ?? 0}%`}</div><span className="text-sm">{completed} / {shoot?.frame_count ?? 8} ready</span></div>
            <Progress value={shoot?.progress ?? (shootStatus === "done" ? 100 : 8)} className="h-1 bg-secondary mb-7" />
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {shots.map((shot, index) => {
                const displayProgress = getShotDisplayProgress(shot);
                return <ShotCard key={shot.id} shot={shot} index={index} progress={displayProgress.value} progressEstimated={displayProgress.estimated} selected={selectedShot === shot.id} onSelect={() => setSelectedShot(shot.id)} onDownload={() => downloadFrame(shot)} />;
              })}
            </div>
          </div>
          <aside className="xl:border-l xl:pl-8 border-border">
            <div className="border-t border-foreground/15 pt-4 mb-8"><div className="font-mono text-[10px] tracking-[.18em] uppercase text-muted-foreground mb-3">Direction lock</div>
              <div className="flex items-start gap-3"><div className="h-9 w-9 bg-primary text-primary-foreground rounded-sm grid place-items-center"><LockKeyhole className="h-4 w-4" /></div><div><p className="font-medium">Product identity held</p><p className="text-xs text-muted-foreground mt-1 leading-relaxed">Product details stay constant while each frame changes purpose, composition and story.</p></div></div>
            </div>
             <Meta label="Product" value={shoot?.product_name || productName} /><Meta label="Category" value={shoot?.category || category} /><Meta label="Campaign" value={shoot?.campaign_format === "front-back-7" ? "Front / back · 7 images" : shoot?.campaign_format === "compact-6" ? "Model lookbook · 6 images" : shoot?.speed_mode === "fast" ? "Fast preview" : "Flexible · 8 images"} /><Meta label="Output" value={(shoot?.output_format || output).toUpperCase()} /><Meta label="References" value={`${shoot?.reference_count || activeReferences.length} source views`} />
              <Meta label="Generation" value={shoot?.generation_mode === "human-model" ? "Human model · category-aware" : "Product only · catalog"} />
              {shoot?.estimated_provider_cost_usd != null && <Meta label="Estimated credit use" value={`$${shoot.estimated_provider_cost_usd.toFixed(4)} base`} />}
              {shoot?.provider_cost_usd != null && shoot.provider_cost_usd > 0 && <Meta label="Actual credit use" value={`$${shoot.provider_cost_usd.toFixed(4)} · ${shoot.provider_request_count} requests`} />}
              {shoot?.identity_profile && <div className="mt-5 border border-foreground/15 bg-card p-4"><div className="font-mono text-[10px] tracking-[.16em] uppercase text-muted-foreground">Immutable source layer</div><div className="mt-2 text-sm font-medium">Product identity profiled</div><p className="mt-1 text-xs text-muted-foreground leading-relaxed">{String(shoot.identity_profile.reference_count ?? "—")} masked views · {String((shoot.identity_profile.identity_evidence as { confidence?: string } | undefined)?.confidence || "profile")} confidence · exact source pixels reserved for composite.</p></div>}
              {shootStatus === "done" && <div className="mt-8 p-4 bg-accent/25 border border-accent/50"><div className="text-sm font-medium">{usedReferenceLock ? "Reference identity preserved." : "Quality checks passed."}</div><div className="text-xs text-muted-foreground mt-1">{usedReferenceLock ? "The CPU generator could not guarantee exact hardware identity, so Atelier preserved the supplied model-carrying reference instead of showing a different product." : shoot?.speed_mode === "fast" ? "The fast preview passed identity, integrity, background and repeat-image validation." : shoot?.campaign_format === "compact-6" ? "Six curated frames passed product identity, same-model, human interaction, anatomy and repeat-image validation." : shoot?.campaign_format === "front-back-7" ? "Seven distinct front/back campaign frames passed identity, integrity, human interaction and repeat-image validation." : "Eight distinct frames passed identity, integrity, background and repeat-image validation."}</div></div>}
          </aside>
        </div>
      </div>
    </StudioShell>;
  }

  return <StudioShell eyebrow="New studio">
    <div className="animate-rise max-w-[1500px] mx-auto px-5 py-8 md:px-10 md:py-12">
      <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_390px] gap-12 xl:gap-20">
        <section>
          <div className="flex items-center gap-2 text-[10px] font-mono tracking-[.2em] uppercase text-muted-foreground mb-6"><span className="h-1.5 w-1.5 rounded-full bg-accent" /> Reference intake <span className="text-foreground/20">/</span> 01</div>
          <h1 className="font-display text-[clamp(3.8rem,8vw,8rem)] leading-[.8] tracking-[-.04em] max-w-3xl">Make the<br /><i>reference</i><br />the source.</h1>
            <p className="mt-8 text-sm md:text-base leading-relaxed text-muted-foreground max-w-lg">Start with a validated fast preview, or choose the campaign contract that fits the product brief.</p>
           <div className="mt-10">
             <div className="font-mono text-[10px] tracking-[.16em] uppercase text-muted-foreground mb-3">Campaign format</div>
             <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
               {([
                  ["compact-6", "Model lookbook", "6 images · 2 realistic model frames + product sides"],
                  ["flexible-8", "Flexible campaign", "1–6 references · 8 distinct images"],
                  ["front-back-7", "Front / back campaign", "Exactly 2 views · exactly 7 images"],
               ] as const).map(([value, label, description]) => (
                  <button key={value} onClick={() => { setCampaignFormat(value); if (value === "front-back-7") { setSpeedMode("campaign"); setReferences((current) => current.slice(0, 2)); } if (value === "compact-6") setSpeedMode("campaign"); }} aria-pressed={campaignFormat === value} data-testid={`button-campaign-format-${value}`} className={cn("border p-3 text-left transition-colors", campaignFormat === value ? "border-primary bg-primary text-primary-foreground" : "border-border hover:border-primary")}>
                   <div className="text-sm">{label}</div><div className={cn("text-[11px] mt-1", campaignFormat === value ? "text-primary-foreground/70" : "text-muted-foreground")}>{description}</div>
                 </button>
               ))}
             </div>
           </div>
           <div className="mt-8">
             <div className="font-mono text-[10px] tracking-[.16em] uppercase text-muted-foreground mb-3">Generation mode</div>
             <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
               {([
                 ["product-only", "Product only", "No person, hands, props or environment · Coach-style catalog angles"],
                 ["human-model", "Human model", "Category-aware real model or smallest useful human context"],
               ] as const).map(([value, label, description]) => (
                 <button key={value} onClick={() => setGenerationMode(value)} aria-pressed={generationMode === value} data-testid={`button-generation-mode-${value}`} className={cn("border p-3 text-left transition-colors", generationMode === value ? "border-primary bg-primary text-primary-foreground" : "border-border hover:border-primary")}>
                   <div className="text-sm">{label}</div><div className={cn("text-[11px] mt-1", generationMode === value ? "text-primary-foreground/70" : "text-muted-foreground")}>{description}</div>
                 </button>
               ))}
             </div>
             <p className="mt-2 text-xs text-muted-foreground">{generationMode === "product-only" ? "Product-only mode keeps every frame focused on the supplied product and does not invoke human-scene validation." : "Human-model mode requires genuine product interaction when the selected category calls for a person; failed anatomy or contact checks are rejected."}</p>
           </div>
            <div className={cn("mt-5 border p-4", generationMode === "human-model" ? "border-primary/30 bg-primary/5" : "border-border bg-card")} data-testid="model-master-panel">
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <div className={cn("font-mono text-[10px] tracking-[.16em] uppercase", generationMode === "human-model" ? "text-primary" : "text-muted-foreground")}>{generationMode === "human-model" ? "Required model master" : "Model master · human mode only"}</div>
                    <p className="mt-1 text-xs leading-relaxed text-muted-foreground">{generationMode === "human-model" ? "Upload one dedicated image of the exact person to keep across every human-model frame. Product references do not define model identity." : "Optional in Product only mode. Select Human model to use a dedicated person identity; product references will never be used as the model master."}</p>
                  </div>
                  <button type="button" onClick={() => modelInputRef.current?.click()} className="shrink-0 border border-primary px-3 py-2 text-[10px] uppercase tracking-wider text-primary hover:bg-primary hover:text-primary-foreground" data-testid="button-add-model-master">
                    {modelMaster ? "Replace" : "Upload"}
                  </button>
                </div>
                {modelMaster ? (
                  <div className="mt-4 flex items-center gap-3 border border-border bg-background p-2">
                    <img src={modelMaster.url} alt="Model master reference" className="h-16 w-16 object-cover" />
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-xs font-medium">{modelMaster.name}</p>
                      <p className="mt-1 text-[10px] text-muted-foreground">Used as the immutable model identity source.</p>
                    </div>
                    <button type="button" onClick={removeModelMaster} className="text-[10px] uppercase tracking-wider text-muted-foreground hover:text-destructive" data-testid="button-remove-model-master">Remove</button>
                  </div>
                ) : (
                  <p className="mt-3 text-[10px] uppercase tracking-wider text-destructive">No model master selected</p>
                )}
                <input ref={modelInputRef} type="file" accept="image/*" className="hidden" onChange={(event) => setModelMasterFile(event.target.files?.[0])} data-testid="input-model-master" />
              </div>
          <div className="mt-12">
              <div className="flex items-center justify-between mb-3"><label className="font-mono text-[10px] tracking-[.16em] uppercase text-muted-foreground">{strictFrontBack ? "Front / back references" : "Source references"}</label><div className="flex items-center gap-3"><button type="button" onClick={() => { setShopifyOpen(true); void loadShopifyProducts(); }} className="text-[10px] uppercase tracking-wider text-primary hover:underline" data-testid="button-import-shopify">Import from Shopify</button><span className="font-mono text-[10px] text-muted-foreground">{activeReferences.length} / {strictFrontBack ? 2 : 6}</span></div></div>
             <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-6 gap-2">
               {activeReferences.map((reference, index) => <div className="aspect-[3/4] relative bg-muted overflow-hidden group" key={reference.id}><img src={reference.url} alt={strictFrontBack ? `${index === 0 ? "Front" : "Back"} reference` : `Reference ${index + 1}`} className="w-full h-full object-cover" /><button aria-label={`Remove reference ${index + 1}`} onClick={() => removeReference(reference.id)} data-testid={`button-remove-reference-${index + 1}`} className="absolute top-1 right-1 h-6 w-6 bg-background/85 grid place-items-center opacity-0 group-hover:opacity-100 focus:opacity-100 transition-opacity"><X className="h-3 w-3" /></button><div className="absolute bottom-0 left-0 right-0 px-2 py-1 bg-gradient-to-t from-foreground/60 text-[9px] text-white">{strictFrontBack ? (index === 0 ? "FRONT" : "BACK") : String(index + 1).padStart(2, "0")}</div></div>)}
               {activeReferences.length < (strictFrontBack ? 2 : 6) && <button onClick={() => inputRef.current?.click()} data-testid="button-add-reference" className="aspect-[3/4] border border-dashed border-foreground/20 hover:border-primary hover:bg-accent/20 transition-colors grid place-items-center text-muted-foreground"><ImagePlus className="h-5 w-5" /><span className="sr-only">Add reference</span></button>}
            </div>
            <input ref={inputRef} type="file" multiple accept="image/*" className="hidden" onChange={(event) => event.target.files && addFiles(event.target.files)} data-testid="input-references" />
              <p className="mt-3 text-xs text-muted-foreground">{strictFrontBack ? "Upload exactly two views in order: front first, back second. Atelier profiles them as one product while preserving the two perspectives." : "Upload 1–6 views. One clear product image is enough to generate; extra front, scale, detail and interior views improve identity control."}</p>
          </div>
              <div className="mt-14 border-t border-foreground/15 pt-5"><div className="flex items-center justify-between mb-5"><div><div className="font-mono text-[10px] tracking-[.16em] uppercase text-muted-foreground">Output mode</div><p className="text-xs text-muted-foreground mt-1">{strictFrontBack ? "The front/back contract always builds all seven images." : campaignFormat === "compact-6" ? "The model lookbook contract builds six curated images." : "Fast validates one primary frame first; campaign builds all eight."}</p></div><span className="font-display italic text-xl text-primary">{strictFrontBack ? "7 frames" : campaignFormat === "compact-6" ? "6 frames" : speedMode === "fast" ? "1 frame" : "8 frames"}</span></div><div className="grid grid-cols-2 gap-2">{([["fast", "Fast preview", `One validated ${generationMode === "human-model" ? "product or model" : "product"} frame`], ["campaign", campaignFormat === "compact-6" ? "Model lookbook" : strictFrontBack ? "Front / back set" : "Full campaign", campaignFormat === "compact-6" ? "Six distinct frames with two realistic model images" : strictFrontBack ? "Seven distinct commercial frames" : "Eight distinct commercial frames"]] as const).map(([value, label, description]) => <button key={value} onClick={() => !strictFrontBack && campaignFormat !== "compact-6" && setSpeedMode(value)} aria-pressed={speedMode === value} disabled={strictFrontBack && value === "fast" || campaignFormat === "compact-6" && value === "fast"} className={cn("border p-3 text-left transition-colors", speedMode === value ? "border-primary bg-primary text-primary-foreground" : "border-border hover:border-primary", (strictFrontBack || campaignFormat === "compact-6") && value === "fast" && "opacity-50 cursor-not-allowed")}><div className="text-sm">{label}</div><div className={cn("text-[11px] mt-1", speedMode === value ? "text-primary-foreground/70" : "text-muted-foreground")}>{description}</div></button>)}</div></div>
        </section>
        <section className="xl:pt-16">
          <div className="border-t-2 border-primary pt-4"><div className="flex items-center justify-between"><div className="font-mono text-[10px] tracking-[.18em] uppercase">Art direction</div><span className="font-mono text-[10px] text-muted-foreground">SETTINGS</span></div></div>
          <div className="space-y-6 mt-6">
            <Field label="Product name"><input value={productName} onChange={(e) => setProductName(e.target.value)} data-testid="input-product-name" /></Field>
             <Field label="Product category"><select value={category} onChange={(e) => setCategory(e.target.value)} data-testid="select-category"><option>Automatic detection</option><option>Handbags</option><option>Wallets</option><option>Watches</option><option>Jewellery</option><option>Footwear</option><option>Ready-to-wear</option><option>Fragrance & beauty</option><option>Furniture & home</option><option>Electronics</option><option>Accessories</option></select></Field>
            <Field label="Atmosphere"><input value={atmosphere} onChange={(e) => setAtmosphere(e.target.value)} data-testid="input-atmosphere" /></Field>
            <Field label="Background"><input value={background} onChange={(e) => setBackground(e.target.value)} data-testid="input-background" /></Field>
            <div><div className="font-mono text-[10px] tracking-[.16em] uppercase text-muted-foreground mb-2">Output format</div><div className="grid grid-cols-3 gap-2">{(["jpeg", "png", "webp"] as const).map((format) => <button key={format} onClick={() => setOutput(format)} data-testid={`button-format-${format}`} className={cn("py-2.5 border text-xs uppercase tracking-wider transition-colors", output === format ? "border-primary bg-primary text-primary-foreground" : "border-border hover:border-primary")}>{format}</button>)}</div></div>
          </div>
           <div className="mt-9 border border-foreground/15 bg-card p-4"><button onClick={() => setLocked(!locked)} data-testid="button-identity-lock" className="w-full flex items-start gap-3 text-left"><div className={cn("mt-0.5 h-5 w-5 grid place-items-center border", locked ? "bg-primary text-primary-foreground border-primary" : "border-foreground/30")}>{locked && <Check className="h-3.5 w-3.5" />}</div><div><div className="text-sm font-medium flex items-center gap-2">Product identity lock <LockKeyhole className="h-3.5 w-3.5 text-accent-foreground" /></div><p className="text-xs text-muted-foreground mt-1 leading-relaxed">Keep material, hardware and silhouette faithful to your references.</p></div></button></div>
            <div className={cn("mt-4 border p-4", engineReady ? "border-accent/50 bg-accent/10" : "border-destructive/30 bg-destructive/5")} data-testid="status-generation-engine">
             <div className="flex items-start gap-3">
               <span className={cn("mt-1 h-2 w-2 rounded-full shrink-0", engineReady ? "bg-accent" : "bg-destructive")} />
               <div className="min-w-0">
                    <div className="text-sm font-medium">{generationLoading ? "Checking generation options…" : engine === "cpu" ? `${selectedEngineLabel} · source preserved` : engineReady ? `${selectedEngineLabel} · human + product verified` : "Selected engine not verified"}</div>
                 <p className="text-xs text-muted-foreground mt-1 leading-relaxed">
                             {generationLoading ? "Checking the worker and current provider availability." : engineReady && !runpodReferenceLimitExceeded ? engine === "cpu" ? "Uses a validated uploaded model reference with the exact BiRefNet product layer; this is source-preserved, not AI human generation." : engine === "flux2-pro" ? "Uses the connected fal.ai API. Every output must pass Atelier’s product identity, human interaction, anatomy, composition, and quality checks before delivery." : engine === "bfl-flux2" ? "Uses the server-routed Black Forest Labs API. Product references, provider request IDs, and output downloads stay on the server; every frame must pass Atelier’s validation gates." : engine === "qwen-runpod" ? "Uses the server-routed RunPod Qwen Image Edit endpoint; provide one to three source images." : engine === "flux1-runpod" ? "Uses the server-routed RunPod FLUX.1 Dev text-to-image endpoint; source images are not sent to this provider." : engine === "gemini-image" ? "Uses server-side Gemini image generation with up to three product references. Every frame must pass Atelier’s identity and distinctness gates." : `${activeGeneration?.provider || "Verified Colab provider"} · verified generation and quality checks` : unavailableMessage}
                 </p>
               </div>
             </div>
             {shopifyOpen && <div className="mt-4 border border-primary/30 bg-primary/5 p-4" data-testid="shopify-import-panel">
               <div className="flex items-center justify-between gap-3"><div><div className="font-medium text-sm">Import product media</div><p className="text-xs text-muted-foreground mt-1">Choose a Shopify product to use its title and images as references.</p></div><button type="button" onClick={() => setShopifyOpen(false)} className="text-xs underline">Close</button></div>
               <div className="flex gap-2 mt-4"><input value={shopifyQuery} onChange={(event) => setShopifyQuery(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter") void loadShopifyProducts(); }} placeholder="Search products" className="min-w-0 flex-1 border border-border bg-background px-3 py-2 text-sm outline-none" data-testid="input-shopify-search" /><button type="button" onClick={() => void loadShopifyProducts()} className="border border-primary px-3 text-xs uppercase tracking-wider" disabled={shopifyLoading}>{shopifyLoading ? "Loading…" : "Search"}</button></div>
               {shopifyError && <p className="mt-3 text-xs text-destructive">{shopifyError}</p>}
               {!shopifyLoading && shopifyProducts.length > 0 && <div className="mt-4 max-h-80 overflow-y-auto overscroll-contain pr-1" data-testid="shopify-product-results"><div className="grid grid-cols-1 sm:grid-cols-2 gap-2">{shopifyProducts.map((product) => { const selected = selectedShopifyProducts.includes(product.id); return <button type="button" key={product.id} onClick={() => toggleShopifyProduct(product.id)} aria-pressed={selected} className={cn("flex items-center gap-3 border bg-background p-2 text-left hover:border-primary", selected ? "border-primary ring-1 ring-primary" : "border-border")} data-testid={`button-shopify-product-${product.handle}`}><div className={cn("h-12 w-12 shrink-0 bg-secondary overflow-hidden", selected && "opacity-80")}>{product.images[0] && <img src={product.images[0].url} alt="" className="h-full w-full object-contain" />}</div><span className="min-w-0 flex-1"><strong className="block truncate text-xs">{product.title}</strong><small className="block truncate text-[10px] text-muted-foreground">{product.product_type || product.vendor || "Shopify product"} · {product.images.length} images</small></span><span className={cn("h-5 w-5 shrink-0 border grid place-items-center text-xs", selected ? "bg-primary text-primary-foreground border-primary" : "border-border")}>{selected ? "✓" : ""}</span></button> })}</div></div>}
               {selectedShopifyProducts.length > 0 && <button type="button" onClick={importSelectedShopifyProducts} className="mt-4 w-full border border-primary bg-primary px-3 py-2 text-xs uppercase tracking-wider text-primary-foreground hover:opacity-90" data-testid="button-import-selected-shopify">{`Import ${selectedShopifyProducts.length} selected product${selectedShopifyProducts.length === 1 ? "" : "s"} · combine photos`}</button>}
               {!shopifyLoading && !shopifyError && shopifyProducts.length === 0 && <p className="mt-4 text-xs text-muted-foreground">No Shopify products found.</p>}
             </div>}
           </div>
            <div className="mt-4">
              <div className="flex items-center justify-between gap-3 mb-2">
                <div className="font-mono text-[10px] tracking-[.16em] uppercase text-muted-foreground">Generation engine</div>
                <Link
                  href="/generate"
                  className="inline-flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-primary hover:underline"
                  data-testid="link-runpod-generator"
                >
                   <Zap className="h-3 w-3" /> RunPod generator
                </Link>
              </div>
                 <div className="grid grid-cols-3 sm:grid-cols-4 gap-2">
                  {engineChoices.map((choice) => (
                    <button
                      key={choice.id}
                      onClick={() => setEngine(choice.id)}
                      aria-pressed={engine === choice.id}
                      data-testid={`button-engine-${choice.id}`}
                      className={cn(
                        "border px-2 py-2.5 text-xs transition-colors text-left",
                        engine === choice.id
                          ? "border-primary bg-primary text-primary-foreground"
                          : "border-border hover:border-primary",
                      )}
                    >
                      <span className="block truncate">{choice.label}</span>
                      <span className={cn(
                        "mt-1 block font-mono text-[9px] uppercase tracking-wider",
                        engine === choice.id ? "text-primary-foreground/70" : choice.ready ? "text-accent-foreground" : "text-muted-foreground",
                      )}>
                         {choice.id === "cpu" ? "Source preserved" : registryStatusLabel(statusForEngine(choice.id))}
                      </span>
                    </button>
                  ))}
                    {!generation?.available_engines?.some((choice) => choice.id === "bfl-flux2") && (
                      <button
                        type="button"
                        disabled
                        data-testid="button-engine-bfl-flux2"
                        className="border border-dashed border-border px-2 py-2.5 text-xs text-left opacity-60 cursor-not-allowed"
                        title={generation?.black_forest_flux2?.reason || "Black Forest Labs FLUX.2 is unavailable."}
                      >
                        <span className="block truncate">FLUX.2 Pro · Black Forest</span>
                        <span className="mt-1 block font-mono text-[9px] uppercase tracking-wider text-muted-foreground">{generationLoading ? "Checking connection" : registryStatusLabel(generation?.black_forest_flux2)}</span>
                      </button>
                    )}
              </div>
                      {!generationLoading && !generation?.available_engines?.some((choice) => choice.id === "bfl-flux2") && generation?.black_forest_flux2?.reason && (
                        <p className="mt-2 text-[10px] leading-relaxed text-muted-foreground" data-testid="status-bfl-flux2-unavailable">
                          Black Forest FLUX.2: {generation.black_forest_flux2.reason}
                        </p>
                      )}
                       <p className="text-[10px] text-muted-foreground mt-2">{engine === "cpu" ? "Fast mode can create a source-preserved human preview when one uploaded reference already shows the model carrying the product. Campaign mode requires a verified Colab human-generation model." : engine === "flux2-pro" ? "Uploaded references are sent to the connected fal.ai FLUX.2 Pro image-editing API. Atelier does not mark output ready until the provider and quality gates pass." : engine === "bfl-flux2" ? "Uploaded references are sent as binary multipart files to Atelier, then forwarded server-side to Black Forest Labs. No provider upload URL, token, or MCP callback reaches the browser." : engine === "qwen-runpod" ? "New Studio will send up to three uploaded references to the server-routed RunPod Qwen Image Edit endpoint. The option stays visible while RunPod is offline, but generation remains blocked until the endpoint is verified." : "Uploaded references and the verified provider model are sent only to the authenticated Colab worker; no unverified model fallback is used."}</p>
               <div className="mt-4 border border-border bg-background/60 p-3" data-testid="status-backend-audit">
                 <div className="flex items-center justify-between mb-3"><div className="font-mono text-[9px] uppercase tracking-[.16em] text-muted-foreground">Backend audit</div><button type="button" onClick={() => { setGenerationLoading(true); setGenerationRefreshKey((value) => value + 1); }} className="inline-flex items-center gap-1 text-[10px] uppercase tracking-wider text-muted-foreground hover:text-foreground" data-testid="button-refresh-generation-status"><RefreshCw className={cn("h-3 w-3", generationLoading && "animate-spin")} /> Refresh</button></div>
                 <div className="mt-2 space-y-2">
                   {backendAudit.map((backend) => (
                      <div key={backend.id || backend.name}>
                        <div className="flex items-start justify-between gap-3 text-[11px]">
                          <span className="truncate">{backend.name}</span>
                          <span className={cn("shrink-0 font-mono text-[9px] uppercase tracking-wider", registryStatusClass(backend))}>
                            {registryStatusLabel(backend)}
                          </span>
                        </div>
                        {backend.next_action && backend.registry_status !== "verified" && (
                          <p className="ml-1 -mt-1 text-[10px] leading-relaxed text-muted-foreground">{backend.next_action}</p>
                        )}
                        {backend.id === "colab-worker" && (
                          <p className="ml-1 -mt-1 text-[10px] leading-relaxed text-muted-foreground">
                            {backend.worker_state === "verified" ? "Provider inference and fresh human/product verification passed." : `Colab state: ${backend.worker_state || "unavailable"} · ${backend.provider || "no provider"}${backend.gpu_available ? " · CUDA" : ""}`}
                          </p>
                        )}
                      </div>
                   ))}
                 </div>
                  {!generationLoading && backendAudit.length > 0 && !backendAudit.some((backend) => backend.ready) && (
                     <p className="mt-3 text-[10px] leading-relaxed text-muted-foreground">
                       A verified Colab provider requires fresh human-with-product verification. Until then, Fast preview uses the available source-preserved reference path.
                   </p>
                 )}
               </div>
            </div>
          {localError && <div className="mt-5"><ErrorBanner message={localError} onRetry={() => setLocalError(null)} /></div>}
                     {estimatedProviderCost != null && (
                       <div className="mt-4 border border-primary/30 bg-primary/5 p-3" data-testid="estimate-provider-cost">
                         <div className="flex items-center justify-between gap-3">
                           <span className="font-mono text-[10px] uppercase tracking-[.16em] text-muted-foreground">Estimated credit use</span>
                           <strong className="text-lg">${estimatedProviderCost.toFixed(4)}</strong>
                         </div>
                         <p className="mt-1 text-[10px] leading-relaxed text-muted-foreground">
                           {plannedFrameCount} {plannedFrameCount === 1 ? "image" : "images"} at {engine === "qwen-runpod" ? "$0.02/image" : "$0.02/megapixel × 1.048576 MP"}. Validation retries can increase the final total; the actual provider total is saved in History.
                         </p>
                       </div>
                     )}
                      <Button onClick={startShoot} disabled={!canStartShoot || !locked || !productName.trim() || createShoot.isPending || generationLoading || (strictFrontBack && activeReferences.length !== 2) || (generationMode === "human-model" && !modelMaster)} className="w-full mt-6 h-12 rounded-sm text-sm" data-testid="button-start-shoot">{createShoot.isPending ? <><RefreshCw className="h-4 w-4 animate-spin" /> Opening studio…</> : generationMode === "human-model" && !modelMaster ? <><CircleAlert className="h-4 w-4" /> Upload Model Master <ArrowUpRight className="h-4 w-4 ml-auto" /></> : runpodReferenceLimitExceeded ? <><CircleAlert className="h-4 w-4" /> Remove extra references <ArrowUpRight className="h-4 w-4 ml-auto" /></> : cpuCampaignBlocked ? <><CircleAlert className="h-4 w-4" /> Fast preview required <ArrowUpRight className="h-4 w-4 ml-auto" /></> : !engineReady && !generationLoading ? <><CircleAlert className="h-4 w-4" /> {engine === "flux2-pro" || engine === "bfl-flux2" || engine === "qwen-runpod" || engine === "flux1-runpod" ? "Provider unavailable" : "Colab verification required"} <ArrowUpRight className="h-4 w-4 ml-auto" /></> : <><Sparkles className="h-4 w-4" /> {speedMode === "fast" ? "Generate fast preview" : strictFrontBack ? "Start seven-image campaign" : "Start eight-frame shoot"} <ArrowUpRight className="h-4 w-4 ml-auto" /></>}</Button>
                <p className="text-center text-[10px] text-muted-foreground mt-3">{engine === "flux2-pro" ? "Paid fal.ai FLUX.2 Pro · server-routed references · validation required before delivery" : engine === "bfl-flux2" ? "Paid Black Forest FLUX.2 Pro · server-routed binary references · validation required before delivery" : engine === "qwen-runpod" ? "Paid RunPod Qwen Image Edit · $0.02/image · up to 3 server-routed references" : engine === "flux1-runpod" ? "Paid RunPod FLUX.1 Dev · $0.02/megapixel · usage is tracked in Operations" : `${selectedEngineLabel} · no hosted API or per-image billing`}</p>
        </section>
      </div>
    </div>
  </StudioShell>;
}

function StudioShell({ children, eyebrow, onReset }: { children: ReactNode; eyebrow: string; onReset?: () => void }) {
  return <div className="min-h-[100dvh]"><header className="h-16 border-b border-border flex items-center justify-between px-5 md:px-10"><div className="flex items-center gap-2 text-xs"><span className="text-muted-foreground">Atelier</span><ChevronRight className="h-3 w-3 text-muted-foreground/50" /><span>{eyebrow}</span></div><div className="flex items-center gap-3">{onReset && <button onClick={onReset} className="text-xs text-muted-foreground hover:text-foreground transition-colors" data-testid="button-reset-header">Close shoot</button>}<div className="h-7 w-7 rounded-full bg-primary text-primary-foreground grid place-items-center font-mono text-[10px]">AM</div></div></header>{children}</div>;
}
function Field({ label, children }: { label: string; children: ReactNode }) { return <label className="block"><span className="font-mono text-[10px] tracking-[.16em] uppercase text-muted-foreground">{label}</span><div className="mt-2 [&_input]:w-full [&_input]:bg-transparent [&_input]:border-0 [&_input]:border-b [&_input]:border-foreground/20 [&_input]:py-2 [&_input]:text-sm [&_input]:outline-none [&_input]:focus:border-primary [&_select]:w-full [&_select]:bg-transparent [&_select]:border-0 [&_select]:border-b [&_select]:border-foreground/20 [&_select]:py-2 [&_select]:text-sm [&_select]:outline-none [&_select]:focus:border-primary">{children}</div></label>; }
function Meta({ label, value }: { label: string; value: string }) { return <div className="flex items-center justify-between py-3 border-b border-foreground/10 text-sm"><span className="text-muted-foreground">{label}</span><span>{value}</span></div>; }
function ErrorBanner({ message, onRetry }: { message: string; onRetry: () => void }) { return <div role="alert" data-testid="status-shoot-error" className="flex items-start gap-3 border border-destructive/30 bg-destructive/5 p-3 text-sm"><CircleAlert className="h-4 w-4 text-destructive mt-0.5 shrink-0" /><div className="flex-1"><div className="font-medium">The studio paused</div><p className="text-xs text-muted-foreground mt-1">{message}</p></div><button onClick={onRetry} className="text-xs underline underline-offset-2" data-testid="button-error-retry">Dismiss</button></div>; }
function ShotCard({ shot, index, progress, progressEstimated, selected, onSelect, onDownload }: { shot: ShootShot; index: number; progress: number; progressEstimated: boolean; selected: boolean; onSelect: () => void; onDownload: () => void }) {
  const ready = Boolean(shot.image_url);
  const failed = ["failed", "error"].includes(shot.status?.toLowerCase());
  const referenceLocked = shot.verification?.startsWith("reference-locked");
  const statusLabel = shot.status === "processing" ? "Processing" : shot.status === "queued" ? "Queued" : shot.status || "Waiting";
  return <article className={cn("group border bg-card overflow-hidden transition-all", selected && "border-primary ring-1 ring-primary", !selected && "border-border hover:border-foreground/35")}>
    <button className="w-full text-left" onClick={onSelect} data-testid={`button-shot-${index + 1}`}>
      <div className="aspect-[4/5] bg-muted relative overflow-hidden">
        {ready ? <img src={shot.image_url!} alt={shot.title} className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-[1.02]" /> : <div className="absolute inset-0 p-3"><div className="h-full skeleton opacity-60" /></div>}
        {ready && <div className="absolute top-2 left-2 bg-background/85 backdrop-blur px-2 py-1 font-mono text-[9px] uppercase tracking-wider">{referenceLocked ? "Source preserved · 100%" : "Ready · 100%"}</div>}
        {failed && <div className="absolute inset-0 bg-destructive/10 flex items-center justify-center text-destructive"><CircleAlert className="h-5 w-5" /></div>}
        {!ready && !failed && <div className="absolute inset-x-3 bottom-3 bg-background/90 backdrop-blur border border-border/80 p-2.5">
          <div className="flex items-center justify-between gap-2 font-mono text-[10px] uppercase tracking-wider">
            <span className="flex items-center gap-2"><span className="h-1.5 w-1.5 bg-accent rounded-full animate-pulse" />{statusLabel}</span>
            <span className="text-primary">{progressEstimated ? "~" : ""}{progress}%</span>
          </div>
          <Progress value={progress} className="h-1 mt-2 bg-secondary" />
        </div>}
      </div>
      <div className="p-3"><div className="flex justify-between gap-2"><span className="font-mono text-[10px] text-muted-foreground">{String(shot.number || index + 1).padStart(2, "0")}</span><MoreHorizontal className="h-3.5 w-3.5 text-muted-foreground" /></div><div className="text-sm mt-2">{shot.title}</div><div className="text-[11px] text-muted-foreground mt-1">{shot.purpose}</div></div>
    </button>
    {ready && <button onClick={onDownload} className="border-t border-border w-full py-2 text-[10px] uppercase tracking-wider text-muted-foreground hover:text-primary transition-colors flex items-center justify-center gap-2" data-testid={`button-download-shot-${index + 1}`}><Download className="h-3 w-3" /> Download frame</button>}
  </article>;
}