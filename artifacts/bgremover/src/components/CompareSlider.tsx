import { useState, useRef, useEffect, MouseEvent as ReactMouseEvent, TouchEvent as ReactTouchEvent } from "react";
import { cn } from "@/lib/utils";
import { ArrowLeftRight } from "lucide-react";

interface CompareSliderProps {
  originalUrl: string;
  resultUrl: string;
  bgColor: string;
}

export function CompareSlider({ originalUrl, resultUrl, bgColor }: CompareSliderProps) {
  const [position, setPosition] = useState(50);
  const [isDragging, setIsDragging] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const handleMove = (clientX: number) => {
    if (!containerRef.current || !isDragging) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(clientX - rect.left, rect.width));
    const percent = Math.max(0, Math.min((x / rect.width) * 100, 100));
    setPosition(percent);
  };

  const handleMouseMove = (e: MouseEvent) => {
    handleMove(e.clientX);
  };

  const handleTouchMove = (e: TouchEvent) => {
    handleMove(e.touches[0].clientX);
  };

  const handleUp = () => {
    setIsDragging(false);
  };

  useEffect(() => {
    if (isDragging) {
      window.addEventListener("mousemove", handleMouseMove);
      window.addEventListener("mouseup", handleUp);
      window.addEventListener("touchmove", handleTouchMove, { passive: false });
      window.addEventListener("touchend", handleUp);
    }

    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleUp);
      window.removeEventListener("touchmove", handleTouchMove);
      window.removeEventListener("touchend", handleUp);
    };
  }, [isDragging]);

  const handleMouseDown = (e: ReactMouseEvent) => {
    setIsDragging(true);
    handleMove(e.clientX);
  };

  const handleTouchStart = (e: ReactTouchEvent) => {
    setIsDragging(true);
    handleMove(e.touches[0].clientX);
  };

  const bgStyle = {
    backgroundColor: bgColor !== 'transparent' ? bgColor : undefined,
    backgroundImage: bgColor === 'transparent' ? 'repeating-conic-gradient(#e0e0e0 0% 25%, #ffffff 0% 50%)' : 'none',
    backgroundSize: '20px 20px'
  };

  return (
    <div 
      ref={containerRef}
      className="relative w-full h-full min-h-[400px] overflow-hidden rounded-xl border select-none touch-none"
      onMouseDown={handleMouseDown}
      onTouchStart={handleTouchStart}
    >
      {/* Background (Result) */}
      <div className="absolute inset-0 w-full h-full" style={bgStyle}>
        <img 
          src={resultUrl} 
          alt="Result" 
          className="absolute top-0 left-0 w-full h-full object-contain pointer-events-none drop-shadow-xl" 
          draggable={false} 
        />
      </div>

      {/* Foreground (Original) */}
      <div 
        className="absolute top-0 left-0 h-full overflow-hidden bg-secondary/30 border-r border-white/20"
        style={{ width: `${position}%` }}
      >
        <img 
          src={originalUrl} 
          alt="Original" 
          className="absolute top-0 left-0 h-full object-contain pointer-events-none"
          style={{ width: containerRef.current?.offsetWidth || '100vw', maxWidth: 'none' }}
          draggable={false}
        />
      </div>

      {/* Slider Handle */}
      <div 
        className="absolute inset-y-0 w-[2px] bg-white shadow-[0_0_10px_rgba(0,0,0,0.5)] cursor-col-resize z-10"
        style={{ left: `calc(${position}% - 1px)` }}
      >
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 flex items-center justify-center w-8 h-8 bg-white border border-border rounded-full shadow-lg text-foreground hover:scale-110 transition-transform active:scale-95">
          <ArrowLeftRight className="h-4 w-4" />
        </div>
      </div>
    </div>
  );
}
