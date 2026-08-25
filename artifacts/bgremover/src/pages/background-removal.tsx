import { useState } from "react";
import { apiFetch } from "@/lib/api";
import { Download, Image as ImageIcon, LoaderCircle, UploadCloud, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";

export function BackgroundRemoval() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const { toast } = useToast();
  const choose = (picked?: File) => {
    if (!picked || !picked.type.startsWith("image/")) return;
    if (preview) URL.revokeObjectURL(preview);
    setFile(picked); setPreview(URL.createObjectURL(picked)); setResult(null);
  };
  const process = async () => {
    if (!file) return;
    setBusy(true);
    try {
      const form = new FormData(); form.append("image", file); form.append("quality", "fast");
      const response = await apiFetch("/api/remove-background", { method: "POST", body: form });
      if (!response.ok) throw new Error(`${response.status} — Could not process this image.`);
      const blob = await response.blob(); setResult(URL.createObjectURL(blob));
    } catch (error) {
      toast({ title: "Background removal failed", description: error instanceof Error ? error.message : "Please try again.", variant: "destructive" });
    } finally { setBusy(false); }
  };
  return <div className="max-w-[1100px] mx-auto px-5 py-12 md:px-10">
    <div className="border-b-2 border-primary pb-5 mb-10"><div className="font-mono text-[10px] tracking-[.2em] uppercase text-muted-foreground mb-4">Utility / background removal</div><h1 className="font-display text-6xl">Clean extraction.</h1><p className="text-sm text-muted-foreground mt-3">A precise transparent cutout for product detail pages.</p></div>
    {!file ? <label className="min-h-[330px] border border-dashed border-foreground/20 hover:border-primary transition-colors flex flex-col items-center justify-center text-center cursor-pointer"><UploadCloud className="h-7 w-7 mb-4 text-primary" /><span className="font-medium">Choose a product image</span><span className="text-xs text-muted-foreground mt-2">PNG, JPEG or WEBP</span><input className="sr-only" type="file" accept="image/*" onChange={(e) => choose(e.target.files?.[0])} data-testid="input-background-image" /></label> :
      <div className="grid lg:grid-cols-[1fr_300px] gap-8"><div className="min-h-[520px] bg-muted flex items-center justify-center p-8 relative">{result ? <img src={result} alt="Processed transparent product" className="max-w-full max-h-[520px] object-contain drop-shadow-xl" /> : <img src={preview!} alt="Product source" className="max-w-full max-h-[520px] object-contain" />}</div><div className="border-t-2 border-primary pt-4"><div className="flex justify-between text-xs mb-8"><span className="text-muted-foreground">Source</span><span className="truncate max-w-[170px]">{file.name}</span></div><Button className="w-full h-11" onClick={process} disabled={busy || Boolean(result)} data-testid="button-remove-background">{busy ? <><LoaderCircle className="animate-spin" /> Processing</> : result ? "Extraction complete" : <><ImageIcon /> Remove background</>}</Button>{result && <a href={result} download={`${file.name.replace(/\.[^.]+$/, "")}_nobg.png`} className="mt-3 flex items-center justify-center gap-2 border py-2.5 text-sm hover:border-primary" data-testid="link-download-background"><Download className="h-4 w-4" /> Download PNG</a>}<button onClick={() => { setFile(null); setPreview(null); setResult(null); }} className="mt-5 w-full text-xs text-muted-foreground hover:text-foreground flex items-center justify-center gap-2" data-testid="button-clear-background"><X className="h-3 w-3" /> Start over</button></div></div>}
  </div>;
}