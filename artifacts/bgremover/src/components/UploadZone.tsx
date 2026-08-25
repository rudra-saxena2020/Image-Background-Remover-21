import { useState, useRef, useCallback } from "react";
import { UploadCloud, Image as ImageIcon, FileWarning, X } from "lucide-react";
import { cn } from "@/lib/utils";

interface UploadZoneProps {
  onFileSelect?: (file: File) => void;
  onFilesSelect?: (files: File[]) => void;
  accept?: string;
  maxSizeMB?: number;
  disabled?: boolean;
  multiple?: boolean;
  maxFiles?: number;
}

export function UploadZone({ 
  onFileSelect,
  onFilesSelect,
  accept = ".jpg,.jpeg,.png,.webp", 
  maxSizeMB = 20,
  disabled = false,
  multiple = false,
  maxFiles = 20
}: UploadZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (!disabled) setIsDragging(true);
  }, [disabled]);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const validateAndProcessFiles = (files: FileList | File[]) => {
    setError(null);
    const validTypes = ['image/jpeg', 'image/png', 'image/webp'];
    const validFiles: File[] = [];
    
    const filesArray = Array.from(files);
    
    if (multiple && filesArray.length > maxFiles) {
      setError(`You can only upload up to ${maxFiles} files at once.`);
      return;
    }
    
    for (const file of filesArray) {
      if (!validTypes.includes(file.type)) {
        setError(`Invalid file format for ${file.name}. Please upload JPG, PNG, or WEBP.`);
        return;
      }
      if (file.size > maxSizeMB * 1024 * 1024) {
        setError(`File ${file.name} is too large. Maximum size is ${maxSizeMB}MB.`);
        return;
      }
      validFiles.push(file);
    }

    if (validFiles.length > 0) {
      if (multiple && onFilesSelect) {
        onFilesSelect(validFiles);
      } else if (!multiple && onFileSelect) {
        onFileSelect(validFiles[0]);
      }
    }
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    
    if (disabled) return;

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndProcessFiles(e.dataTransfer.files);
      e.dataTransfer.clearData();
    }
  }, [disabled, multiple]);

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      validateAndProcessFiles(e.target.files);
      if (fileInputRef.current) {
        fileInputRef.current.value = ''; // Reset so same file can be selected again
      }
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (disabled) return;
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInputRef.current?.click();
    }
  };

  return (
    <div className="w-full">
      <div
        className={cn(
          "relative group overflow-hidden rounded-xl border-2 border-dashed transition-all duration-200 ease-in-out",
          isDragging 
            ? "border-primary bg-primary/5 scale-[1.01]" 
            : "border-border/60 hover:border-primary/50 hover:bg-secondary/30 bg-card",
          disabled && "opacity-50 cursor-not-allowed pointer-events-none"
        )}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !disabled && fileInputRef.current?.click()}
        onKeyDown={handleKeyDown}
        role="button"
        tabIndex={disabled ? -1 : 0}
        aria-disabled={disabled}
        aria-label={multiple ? "Upload product images" : "Upload product image"}
      >
        <input
          type="file"
          ref={fileInputRef}
          className="hidden"
          accept={accept}
          onChange={handleFileInput}
          disabled={disabled}
          multiple={multiple}
        />
        
        <div className="flex flex-col items-center justify-center py-16 px-6 text-center">
          <div className={cn(
            "mb-6 flex h-20 w-20 items-center justify-center rounded-full transition-all duration-300",
            isDragging ? "bg-primary text-primary-foreground shadow-lg shadow-primary/20 scale-110" : "bg-secondary text-muted-foreground group-hover:bg-primary/10 group-hover:text-primary"
          )}>
            <UploadCloud className="h-10 w-10" />
          </div>
          
          <h3 className="mb-2 text-xl font-semibold tracking-tight text-foreground">
            {multiple ? "Upload Product Images" : "Upload Product Image"}
          </h3>
          
          <p className="mb-6 text-sm text-muted-foreground max-w-sm">
            Drag & drop {multiple ? `up to ${maxFiles} product images` : "your product image"} here, or click to browse your files.
          </p>
          
          <div className="flex items-center gap-4 text-xs font-medium text-muted-foreground">
            <span className="flex items-center gap-1.5 px-2 py-1 rounded bg-secondary">
              <ImageIcon className="h-3.5 w-3.5" /> JPG, PNG, WEBP
            </span>
            <span className="flex items-center gap-1.5 px-2 py-1 rounded bg-secondary">
              Up to {maxSizeMB}MB {multiple && `per file`}
            </span>
          </div>
        </div>
      </div>

      {error && (
        <div className="mt-4 flex items-center gap-3 rounded-lg bg-destructive/10 px-4 py-3 text-sm text-destructive animate-in slide-in-from-top-2">
          <FileWarning className="h-4 w-4 shrink-0" />
          <p className="flex-1 font-medium">{error}</p>
          <button 
            onClick={() => setError(null)}
            className="p-1 hover:bg-destructive/20 rounded-md transition-colors"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
      )}
    </div>
  );
}
