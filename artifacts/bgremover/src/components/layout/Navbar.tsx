import { Link, useLocation } from "wouter";
import { cn } from "@/lib/utils";
import { useGetModelHealth, getGetModelHealthQueryKey } from "@workspace/api-client-react";
import { Aperture, Clock3, Layers3, Sparkles, Activity, ShieldCheck, Scissors, Gauge, Zap } from "lucide-react";

type GenerationBackend = {
  ready?: boolean;
  name?: string;
  worker_state?: string;
  next_action?: string;
};

type GenerationHealth = {
  remote_worker?: GenerationBackend;
  qwen?: GenerationBackend;
  flux_schnell?: GenerationBackend;
  fooocus?: GenerationBackend;
  hidream?: GenerationBackend;
  flux2?: GenerationBackend;
  flux2_klein?: GenerationBackend;
  sdxl?: GenerationBackend;
};

export function Navbar() {
  const [location] = useLocation();
  const { data: health, isLoading } = useGetModelHealth({ query: { refetchInterval: 10000, queryKey: getGetModelHealthQueryKey() } });
  const generation = health?.generation as GenerationHealth | undefined;
  const backends = generation ? [generation.remote_worker, generation.qwen, generation.flux_schnell, generation.fooocus, generation.hidream, generation.flux2, generation.flux2_klein, generation.sdxl] : [];
  const readyEngine = backends.find((backend) => backend?.ready === true);
  const generationReady = Boolean(readyEngine);
  const colab = generation?.remote_worker;
  const links = [
    { href: "/", label: "New studio", icon: Sparkles },
    { href: "/generate", label: "Generate image", icon: Zap },
    { href: "/batch", label: "Batch remover", icon: Layers3 },
    { href: "/history", label: "History", icon: Clock3 },
    { href: "/remove-background", label: "Cutout utility", icon: Scissors },
    { href: "/admin", label: "Operations", icon: Gauge },
  ];
  return (
    <aside className="w-full md:w-[250px] md:fixed md:inset-y-0 md:left-0 z-40 bg-sidebar text-sidebar-foreground border-b md:border-b-0 md:border-r border-sidebar-border flex md:flex-col">
      <div className="px-5 py-5 md:px-6 md:py-7">
        <Link href="/" className="flex items-center gap-3" data-testid="link-brand">
          <div className="h-9 w-9 rounded-sm bg-sidebar-primary text-sidebar-primary-foreground grid place-items-center">
            <Aperture className="h-4 w-4" />
          </div>
          <div>
            <div className="font-display text-[25px] leading-[.8] tracking-tight">atelier</div>
            <div className="font-mono text-[9px] tracking-[.22em] uppercase mt-2 opacity-60">image direction</div>
          </div>
        </Link>
      </div>
      <nav className="flex md:flex-col gap-1 px-3 pb-3 md:px-4 md:pt-8 overflow-x-auto" aria-label="Main navigation">
        <div className="hidden md:block px-3 pb-3 text-[10px] uppercase tracking-[.2em] text-sidebar-foreground/45">Workspace</div>
        {links.map(({ href, label, icon: Icon }) => (
          <Link key={href} href={href} data-testid={`link-nav-${label.toLowerCase().replace(" ", "-")}`}
            className={cn("shrink-0 flex items-center gap-3 rounded-sm px-3 py-2.5 text-sm transition-colors", location === href ? "bg-sidebar-accent text-sidebar-accent-foreground" : "text-sidebar-foreground/65 hover:bg-sidebar-accent/60 hover:text-sidebar-foreground")}>
            <Icon className="h-4 w-4" /><span>{label}</span>
          </Link>
        ))}
      </nav>
      <div className="hidden md:block mt-auto p-4">
        <div className="border border-sidebar-border rounded-sm p-3 bg-sidebar-accent/30">
          <div className="flex items-center gap-2 text-[11px] font-medium">
             {isLoading || !health ? <Activity className="h-3.5 w-3.5 animate-pulse text-sidebar-primary" /> : generationReady ? <ShieldCheck className="h-3.5 w-3.5 text-sidebar-primary" /> : <Activity className="h-3.5 w-3.5 text-destructive" />}
             {isLoading || !health ? "Checking engine" : generationReady ? "Human + product verified" : colab?.worker_state === "inference-ready-unverified" ? "Colab inference ready · verify product" : "No verified generator"}
          </div>
           <div className="mt-2 font-mono text-[9px] tracking-widest uppercase text-sidebar-foreground/45">{isLoading || !health ? "secure session" : generationReady ? readyEngine?.name : colab?.next_action || "Run CUDA smoke audit"}</div>
        </div>
        <p className="font-mono text-[9px] tracking-[.15em] uppercase text-sidebar-foreground/35 mt-5 px-1">AT / 2025.04</p>
      </div>
    </aside>
  );
}