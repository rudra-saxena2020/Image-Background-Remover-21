import { useState, useEffect, useRef } from "react";
import { UploadZone } from "@/components/UploadZone";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Loader2, Download, Play, CheckCircle2, XCircle, Clock, ArchiveRestore, Image as ImageIcon } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { cn } from "@/lib/utils";
import JSZip from "jszip";

type BatchItem = {
  id: string;
  file: File;
  previewUrl: string;
  status: "waiting" | "processing" | "completed" | "failed";
  resultUrl?: string;
  timeMs?: number;
  error?: string;
};

export function Batch() {
  const [items, setItems] = useState<BatchItem[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const itemsRef = useRef<BatchItem[]>([]);
  const { toast } = useToast();

  useEffect(() => {
    itemsRef.current = items;
  }, [items]);

  const handleFilesSelect = (files: File[]) => {
    const newItems = files.map(f => ({
      id: Math.random().toString(36).substring(7),
      file: f,
      previewUrl: URL.createObjectURL(f),
      status: "waiting" as const
    }));
    
    setItems(prev => {
      const next = [...prev, ...newItems].slice(0, 20);
      const discarded = [...prev, ...newItems].slice(20);
      discarded.forEach(item => URL.revokeObjectURL(item.previewUrl));
      return next;
    }); // Max 20
  };

  useEffect(() => {
    return () => {
      itemsRef.current.forEach(item => {
        URL.revokeObjectURL(item.previewUrl);
        if (item.resultUrl) URL.revokeObjectURL(item.resultUrl);
      });
    };
  }, []);

  const clearItems = () => {
    if (isProcessing) return;
    items.forEach(item => {
      URL.revokeObjectURL(item.previewUrl);
      if (item.resultUrl) URL.revokeObjectURL(item.resultUrl);
    });
    setItems([]);
  };

  const handleProcessAll = async () => {
    setIsProcessing(true);
    
    for (let i = 0; i < items.length; i++) {
      if (items[i].status === "completed") continue;
      
      setItems(prev => prev.map((item, idx) => 
        idx === i ? { ...item, status: "processing" } : item
      ));

      try {
        const formData = new FormData();
        formData.append('image', items[i].file);
        formData.append('quality', 'fast'); // Defaulting to fast for batch
        
        const res = await fetch('/api/remove-background', { 
          method: 'POST', 
          body: formData 
        });
        
        if (!res.ok) {
          let errMsg = 'Failed to process image';
          try {
            const errBody = await res.json();
            errMsg = errBody.detail || errBody.error || errMsg;
          } catch {
            // ignore
          }
          throw new Error(errMsg);
        }
        
        const totalMs = parseFloat(res.headers.get('X-Total-Time') || '0');
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        
        setItems(prev => prev.map((item, idx) => 
          idx === i ? { 
            ...item, 
            status: "completed", 
            resultUrl: url,
            timeMs: totalMs
          } : item
        ));
      } catch (err: any) {
        setItems(prev => prev.map((item, idx) => 
          idx === i ? { 
            ...item, 
            status: "failed", 
            error: err.message || "Failed" 
          } : item
        ));
      }
    }
    
    setIsProcessing(false);
  };

  const handleDownloadAll = async () => {
    const completedItems = items.filter(i => i.status === "completed" && i.resultUrl);
    if (completedItems.length === 0) return;
    
    toast({
      title: "Generating ZIP",
      description: "Compressing your images...",
    });

    try {
      const zip = new JSZip();
      for (const item of completedItems) {
        if (!item.resultUrl) continue;
        const response = await fetch(item.resultUrl);
        const blob = await response.blob();
        zip.file(item.file.name.replace(/\.[^.]+$/, '') + '_nobg.png', blob);
      }
      const zipBlob = await zip.generateAsync({ type: 'blob' });
      const url = URL.createObjectURL(zipBlob);
      
      const a = document.createElement('a');
      a.href = url;
      a.download = `birefnet_batch_${new Date().getTime()}.zip`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      toast({
        title: "Download failed",
        description: "Could not create ZIP file.",
        variant: "destructive"
      });
    }
  };

  const completedCount = items.filter(i => i.status === "completed").length;
  const failedCount = items.filter(i => i.status === "failed").length;
  const processedCount = completedCount + failedCount;
  const progressPercent = items.length ? (processedCount / items.length) * 100 : 0;

  return (
    <div className="container mx-auto p-4 md:p-8 max-w-6xl animate-in fade-in duration-500">
      <div className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight mb-2">Batch Processing</h1>
        <p className="text-muted-foreground">Process up to 20 images sequentially.</p>
      </div>

      {!items.length ? (
        <div className="max-w-2xl mx-auto mt-12">
          <UploadZone 
            onFilesSelect={handleFilesSelect} 
            multiple={true} 
            maxFiles={20} 
          />
        </div>
      ) : (
        <div className="space-y-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-card p-5 rounded-xl border shadow-sm">
            <div className="flex-1 w-full max-w-md">
              <div className="flex justify-between text-sm mb-2 font-medium">
                <span>{processedCount} / {items.length} processed</span>
                <span className="text-muted-foreground">{Math.round(progressPercent)}%</span>
              </div>
              <Progress value={progressPercent} className="h-2" />
            </div>
            
            <div className="flex items-center gap-3">
            <Button 
                variant="outline" 
                onClick={clearItems}
                disabled={isProcessing}
              >
                Clear All
              </Button>
              {completedCount > 0 && !isProcessing && (
                <Button variant="secondary" onClick={handleDownloadAll}>
                  <ArchiveRestore className="mr-2 h-4 w-4" />
                  Download ZIP
                </Button>
              )}
              {processedCount < items.length && (
                <Button 
                  onClick={handleProcessAll} 
                  disabled={isProcessing}
                  className="shadow-md shadow-primary/20"
                >
                  {isProcessing ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Processing...
                    </>
                  ) : (
                    <>
                      <Play className="mr-2 h-4 w-4" />
                      Process All
                    </>
                  )}
                </Button>
              )}
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {items.map((item) => (
              <div key={item.id} className="bg-card rounded-xl border overflow-hidden flex flex-col group transition-all hover:shadow-md hover:border-primary/30">
                <div className="relative h-48 bg-secondary/50 flex items-center justify-center p-4">
                  {item.status === 'completed' && item.resultUrl ? (
                    <div className="absolute inset-0 bg-[repeating-conic-gradient(#e0e0e0_0%_25%,_#ffffff_0%_50%)] bg-[length:10px_10px]">
                      <img src={item.resultUrl} alt="Result" className="w-full h-full object-contain p-4 drop-shadow-lg" />
                    </div>
                  ) : (
                    <img 
                      src={item.previewUrl} 
                      alt="Preview" 
                      className={cn(
                        "max-w-full max-h-full object-contain transition-all",
                        item.status === "processing" && "opacity-50 blur-sm scale-95"
                      )} 
                    />
                  )}
                  
                  {item.status === "processing" && (
                    <div className="absolute inset-0 flex items-center justify-center bg-background/20 backdrop-blur-[2px]">
                      <div className="bg-background shadow-lg rounded-full p-3">
                        <Loader2 className="h-6 w-6 text-primary animate-spin" />
                      </div>
                    </div>
                  )}
                </div>
                
                <div className="p-4 flex flex-col gap-3 flex-1 border-t">
                  <div className="flex items-start justify-between gap-2">
                    <span className="text-sm font-medium truncate" title={item.file.name}>{item.file.name}</span>
                  </div>
                  
                  <div className="flex items-center justify-between mt-auto pt-2 border-t border-border/50">
                    <div className="flex items-center">
                      {item.status === "waiting" && <span className="text-xs font-medium text-muted-foreground flex items-center gap-1.5"><Clock className="h-3.5 w-3.5" /> Waiting</span>}
                      {item.status === "processing" && <span className="text-xs font-medium text-primary flex items-center gap-1.5 animate-pulse"><Loader2 className="h-3.5 w-3.5 animate-spin" /> Processing</span>}
                      {item.status === "completed" && <span className="text-xs font-medium text-emerald-600 flex items-center gap-1.5"><CheckCircle2 className="h-3.5 w-3.5" /> {item.timeMs ? `${(item.timeMs/1000).toFixed(1)}s` : 'Done'}</span>}
                      {item.status === "failed" && <span className="text-xs font-medium text-destructive flex items-center gap-1.5" title={item.error}><XCircle className="h-3.5 w-3.5" /> Failed</span>}
                    </div>
                    
                    {item.status === "completed" && item.resultUrl && (
                      <a 
                        href={item.resultUrl} 
                        download={item.file.name.replace(/\.[^.]+$/, '') + '_nobg.png'}
                        className="text-primary hover:bg-primary/10 p-1.5 rounded-md transition-colors"
                        title="Download single image"
                      >
                        <Download className="h-4 w-4" />
                      </a>
                    )}
                  </div>
                </div>
              </div>
            ))}
            
            {items.length < 20 && !isProcessing && (
              <div className="bg-secondary/20 rounded-xl border-2 border-dashed flex flex-col items-center justify-center p-6 text-center min-h-[250px]">
                <UploadZone 
                  onFilesSelect={handleFilesSelect} 
                  multiple={true} 
                  maxFiles={20 - items.length} 
                />
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
