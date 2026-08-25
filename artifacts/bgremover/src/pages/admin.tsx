import { useEffect, useState } from "react";
import { Activity, CheckCircle2, CircleAlert, DollarSign, Gauge, Layers3, RefreshCw } from "lucide-react";

type Metrics = {
  total_shoots: number;
  active_shoots: number;
  completed_shoots: number;
  failed_shoots: number;
  cancelled_shoots: number;
  frames_generated: number;
  frames_failed: number;
  provider: string;
  preprocessor: string;
  provider_cost_usd: number;
  provider_usage: Array<{
    provider: string;
    model: string;
    price: string;
    requests: number;
    cost_usd: number;
  }>;
};

const initialMetrics: Metrics = {
  total_shoots: 0,
  active_shoots: 0,
  completed_shoots: 0,
  failed_shoots: 0,
  cancelled_shoots: 0,
  frames_generated: 0,
  frames_failed: 0,
  provider: "Local GPU / HiDream-O1 Image, FLUX.2 Klein, FLUX.2 Dev, or SDXL",
  preprocessor: "BiRefNet",
  provider_cost_usd: 0,
  provider_usage: [],
};

const usd = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  minimumFractionDigits: 2,
  maximumFractionDigits: 4,
});

export function Admin() {
  const [metrics, setMetrics] = useState<Metrics>(initialMetrics);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    try {
      const response = await fetch("/api/admin/metrics");
      if (!response.ok) throw new Error(`Metrics unavailable (${response.status})`);
      setMetrics(await response.json());
      setError(null);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Metrics unavailable");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void load();
    const interval = window.setInterval(() => void load(), 10000);
    return () => window.clearInterval(interval);
  }, []);

  const qualityRate = metrics.frames_generated + metrics.frames_failed
    ? Math.round((metrics.frames_generated / (metrics.frames_generated + metrics.frames_failed)) * 100)
    : 0;

  return (
    <div className="animate-rise max-w-[1500px] mx-auto px-5 py-8 md:px-10 md:py-12">
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-5 mb-10">
        <div>
          <div className="flex items-center gap-2 font-mono text-[10px] tracking-[.2em] uppercase text-muted-foreground mb-5">
            <span className="h-1.5 w-1.5 rounded-full bg-accent" /> Operations / live
          </div>
          <h1 className="font-display text-6xl md:text-8xl leading-[.8] tracking-tight">The <i>desk.</i></h1>
          <p className="mt-6 text-sm text-muted-foreground max-w-md">A quiet view of queue pressure, validated output, and the engines behind each campaign.</p>
        </div>
        <button onClick={() => void load()} className="flex items-center gap-2 border border-border px-3 py-2 text-xs hover:border-primary transition-colors" data-testid="button-refresh-metrics">
          <RefreshCw className={loading ? "h-3.5 w-3.5 animate-spin" : "h-3.5 w-3.5"} /> Refresh
        </button>
      </div>
      {error && <div role="alert" className="mb-6 border border-destructive/30 bg-destructive/5 p-3 text-sm flex items-center gap-2"><CircleAlert className="h-4 w-4 text-destructive" />{error}</div>}
      <div className="grid grid-cols-2 lg:grid-cols-5 gap-3 mb-10">
        <Metric label="Active shoots" value={metrics.active_shoots} icon={Activity} accent />
        <Metric label="Completed shoots" value={metrics.completed_shoots} icon={CheckCircle2} />
        <Metric label="Validated frames" value={metrics.frames_generated} icon={Layers3} />
        <Metric label="Quality pass rate" value={`${qualityRate}%`} icon={Gauge} />
        <Metric label="RunPod spend · session" value={usd.format(metrics.provider_cost_usd)} icon={DollarSign} />
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_360px] gap-10">
        <section className="border-t-2 border-primary pt-4">
          <div className="font-mono text-[10px] tracking-[.18em] uppercase mb-6">Queue health</div>
          <div className="space-y-0 border-t border-border">
            <Row label="Total shoots this session" value={metrics.total_shoots} />
            <Row label="Failed shoots" value={metrics.failed_shoots} danger={metrics.failed_shoots > 0} />
            <Row label="Cancelled shoots" value={metrics.cancelled_shoots} />
            <Row label="Failed frame validations" value={metrics.frames_failed} danger={metrics.frames_failed > 0} />
          </div>
        </section>
        <aside className="border border-border bg-card p-5 h-fit">
          <div className="font-mono text-[10px] tracking-[.18em] uppercase text-muted-foreground mb-5">Pipeline</div>
          <div className="flex items-center gap-3 pb-4 border-b border-border"><div className="h-8 w-8 bg-primary text-primary-foreground grid place-items-center"><Layers3 className="h-4 w-4" /></div><div><div className="text-sm">Reference preprocessing</div><div className="text-xs text-muted-foreground mt-1">{metrics.preprocessor}</div></div></div>
          <div className="flex items-center gap-3 pt-4"><div className="h-8 w-8 bg-accent text-accent-foreground grid place-items-center"><Gauge className="h-4 w-4" /></div><div><div className="text-sm">Image generation</div><div className="text-xs text-muted-foreground mt-1">{metrics.provider}</div><div className="text-[10px] text-accent-foreground mt-1">RunPod costs are recorded from each completed provider response.</div></div></div>
        </aside>
      </div>
      <section className="mt-10 border-t-2 border-primary pt-4">
        <div className="font-mono text-[10px] tracking-[.18em] uppercase mb-5">RunPod usage · current session</div>
        <div className="border border-border divide-y divide-border">
          {metrics.provider_usage.map((usage) => (
            <div key={usage.model} className="grid grid-cols-[1fr_auto] gap-4 p-4 md:grid-cols-[1.5fr_1fr_auto] md:items-center">
              <div><div className="text-sm">{usage.model}</div><div className="mt-1 text-[11px] text-muted-foreground">{usage.provider} · {usage.price}</div></div>
              <div className="hidden md:block text-xs text-muted-foreground">{usage.requests} provider {usage.requests === 1 ? "request" : "requests"}</div>
              <div className="text-right font-display text-2xl">{usd.format(usage.cost_usd)}</div>
            </div>
          ))}
        </div>
        <p className="mt-3 text-xs text-muted-foreground">Totals reset if the API service restarts. Costs use RunPod’s response when present and the published model rate only as a documented fallback.</p>
      </section>
    </div>
  );
}

function Metric({ label, value, icon: Icon, accent = false }: { label: string; value: number | string; icon: typeof Activity; accent?: boolean }) {
  return <div className={`border border-border p-4 md:p-5 ${accent ? "bg-primary text-primary-foreground border-primary" : "bg-card"}`}><Icon className="h-4 w-4 mb-8 opacity-70" /><div className="font-display text-4xl">{value}</div><div className="font-mono text-[9px] tracking-[.15em] uppercase mt-2 opacity-65">{label}</div></div>;
}

function Row({ label, value, danger = false }: { label: string; value: number; danger?: boolean }) {
  return <div className="flex items-center justify-between py-4 border-b border-border text-sm"><span className="text-muted-foreground">{label}</span><span className={danger ? "text-destructive font-medium" : ""}>{value}</span></div>;
}