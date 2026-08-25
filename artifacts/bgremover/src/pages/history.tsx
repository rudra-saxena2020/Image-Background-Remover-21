import { useEffect, useState } from "react";
import { getGetHistoryQueryKey, useGetHistory } from "@workspace/api-client-react";
import { Button } from "@/components/ui/button";
import { Download, Clock, Image as ImageIcon, LayoutGrid, Calendar, Activity } from "lucide-react";
import { format } from "date-fns";
import { apiFetch, apiUrl } from "@/lib/api";

export function History() {
  const { data: history, isLoading } = useGetHistory({
    query: {
      queryKey: getGetHistoryQueryKey(),
      refetchInterval: 5000,
    }
  });

  const handleDownload = async (url: string, filename: string) => {
    try {
      const response = await apiFetch(url);
      const blob = await response.blob();
      const objectUrl = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = objectUrl;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(objectUrl);
    } catch (err) {
      console.error("Failed to download", err);
    }
  };

  return (
    <div className="container mx-auto p-4 md:p-8 max-w-6xl animate-in fade-in duration-500">
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight mb-2">Session History</h1>
          <p className="text-muted-foreground">Recent backgrounds removed in this session.</p>
        </div>
        <div className="hidden sm:flex items-center justify-center h-12 w-12 rounded-xl bg-primary/10 text-primary">
          <LayoutGrid className="h-6 w-6" />
        </div>
      </div>

      {isLoading && !history ? (
        <div className="flex flex-col items-center justify-center py-20 text-muted-foreground">
          <Activity className="h-8 w-8 animate-pulse mb-4" />
          <p>Loading history...</p>
        </div>
      ) : !history || history.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-20 text-center bg-card border border-dashed rounded-xl max-w-2xl mx-auto">
          <div className="bg-secondary p-4 rounded-full mb-4">
            <ImageIcon className="h-8 w-8 text-muted-foreground" />
          </div>
          <h3 className="text-xl font-semibold mb-2">No history yet</h3>
          <p className="text-muted-foreground mb-6 max-w-sm">
            Process a product to see it appear here. History is kept for the duration of your session.
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {history.map((item) => {
            const hasResult = !!item.result_url;
            return (
              <div key={item.id} className="bg-card rounded-xl border overflow-hidden flex flex-col group transition-all hover:shadow-md hover:border-primary/30">
                <div className="relative h-56 bg-[repeating-conic-gradient(#e0e0e0_0%_25%,_#ffffff_0%_50%)] bg-[length:10px_10px] flex items-center justify-center border-b">
                  {hasResult ? (
                    <img 
                      src={item.result_url!} 
                      alt={item.filename} 
                      className="w-full h-full object-contain p-4 drop-shadow-lg" 
                      loading="lazy"
                    />
                  ) : (
                    <div className="flex flex-col items-center text-muted-foreground">
                      <ImageIcon className="h-8 w-8 mb-2 opacity-50" />
                      <span className="text-sm">Image unavailable</span>
                    </div>
                  )}
                  
                  <div className="absolute top-2 right-2 flex items-center gap-1.5">
                    <span className="bg-background/80 backdrop-blur-md px-2 py-0.5 rounded text-[10px] uppercase tracking-wider font-bold shadow-sm">
                      {item.mode}
                    </span>
                  </div>
                </div>
                
                <div className="p-4 flex flex-col gap-3 flex-1">
                  <h4 className="font-medium text-sm truncate" title={item.filename}>
                    {item.filename}
                  </h4>
                  
                  <div className="grid grid-cols-2 gap-2 text-xs text-muted-foreground mt-auto pt-2 border-t">
                    <div className="flex items-center gap-1.5" title="Resolution">
                      <ImageIcon className="h-3 w-3" />
                      {item.width}x{item.height}
                    </div>
                    <div className="flex items-center gap-1.5" title="Processing Time">
                      <Clock className="h-3 w-3" />
                      {(item.processing_time_ms / 1000).toFixed(1)}s
                    </div>
                    <div className="flex items-center gap-1.5 col-span-2 mt-1" title="Date">
                      <Calendar className="h-3 w-3" />
                      {format(new Date(item.created_at), 'MMM d, HH:mm')}
                    </div>
                  </div>
                  
                  {hasResult && (
                    <Button 
                      variant="secondary" 
                      className="w-full mt-2" 
                      size="sm"
                      onClick={() => handleDownload(item.result_url!, item.filename.replace(/\.[^.]+$/, '') + '_nobg.png')}
                    >
                      <Download className="mr-2 h-4 w-4" />
                      Download
                    </Button>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
      <PaidShootHistory />
    </div>
  );
}

type PaidShot = {
  id: string;
  number: number;
  title: string;
  status: string;
  image_url: string | null;
  cost_usd: number | null;
};

type PaidShoot = {
  id: string;
  product_name: string;
  provider: string | null;
  remote_model: string | null;
  status: string;
  created_at: string;
  provider_cost_usd: number;
  provider_request_count: number;
  shots: PaidShot[];
};

function PaidShootHistory() {
  const [shoots, setShoots] = useState<PaidShoot[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;
    apiFetch("/api/shoots/history")
      .then((response) => {
        if (!response.ok) throw new Error("Paid shoot history unavailable");
        return response.json();
      })
      .then((data: unknown) => {
        if (active) setShoots(Array.isArray(data) ? (data as PaidShoot[]) : []);
      })
      .catch(() => {
        if (active) setShoots([]);
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  return (
    <section className="mt-14 border-t-2 border-primary pt-5">
      <div className="mb-6">
        <h2 className="text-2xl font-bold tracking-tight">Paid generation history</h2>
        <p className="text-sm text-muted-foreground mt-1">
          RunPod shoots, costs, and validated outputs are retained after an API restart.
        </p>
      </div>
      {loading ? (
        <div className="border border-dashed p-8 text-sm text-muted-foreground">Loading paid shoots…</div>
      ) : shoots.length === 0 ? (
        <div className="border border-dashed p-8 text-sm text-muted-foreground">No paid shoots yet.</div>
      ) : (
        <div className="space-y-6">
          {shoots.map((shoot) => (
            <article key={shoot.id} className="border border-border bg-card">
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 p-4 border-b">
                <div>
                  <h3 className="font-medium">{shoot.product_name}</h3>
                  <p className="text-xs text-muted-foreground mt-1">
                    {shoot.provider || "Hosted provider"} · {shoot.remote_model || "generation"} ·{" "}
                    {format(new Date(shoot.created_at), "MMM d, yyyy HH:mm")}
                  </p>
                </div>
                <div className="flex items-center gap-4 text-xs">
                  <span className="uppercase tracking-wider">{shoot.status}</span>
                  <span>{shoot.provider_request_count} requests</span>
                  <strong className="text-sm">${shoot.provider_cost_usd.toFixed(4)}</strong>
                  {shoot.status.toLowerCase() === "completed" && <a href={apiUrl(`/api/shoots/${shoot.id}/export`)} download className="text-primary hover:underline">Download ZIP</a>}
                </div>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-8 gap-3 p-4">
                {shoot.shots.map((shot) => (
                  <div key={shot.id} className="min-w-0">
                    <div className="aspect-square bg-secondary border flex items-center justify-center overflow-hidden">
                      {shot.image_url ? (
                        <img src={shot.image_url} alt={shot.title} className="w-full h-full object-contain" loading="lazy" />
                      ) : (
                        <span className="text-[10px] text-muted-foreground text-center px-2">{shot.status}</span>
                      )}
                    </div>
                    <div className="mt-2 text-[11px] truncate" title={shot.title}>{shot.number}. {shot.title}</div>
                    <div className="flex items-center justify-between gap-1 mt-1 text-[10px] text-muted-foreground">
                      <span>{shot.cost_usd == null ? "—" : `$${shot.cost_usd.toFixed(4)}`}</span>
                      {shot.image_url && (
                        <a href={shot.image_url} download={`${shoot.product_name}-${shot.number}.png`} className="text-primary hover:underline">
                          Download
                        </a>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </article>
          ))}
        </div>
      )}
    </section>
  );
}
