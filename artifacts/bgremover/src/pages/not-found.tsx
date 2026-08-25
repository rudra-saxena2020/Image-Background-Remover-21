import { Link } from "wouter";
import { Layers } from "lucide-react";

export default function NotFound() {
  return (
    <div className="flex-1 flex flex-col items-center justify-center p-8 text-center animate-in fade-in zoom-in duration-500">
      <div className="flex h-16 w-16 items-center justify-center rounded-xl bg-secondary text-muted-foreground mb-6">
        <Layers className="h-8 w-8" />
      </div>
      <h1 className="text-3xl font-bold tracking-tight text-foreground sm:text-4xl mb-2">
        Page not found
      </h1>
      <p className="text-lg text-muted-foreground mb-8 max-w-[500px]">
        The page you're looking for doesn't exist or has been moved.
      </p>
      <Link 
        href="/" 
        className="inline-flex h-10 items-center justify-center rounded-md bg-primary px-8 text-sm font-medium text-primary-foreground shadow transition-colors hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
      >
        Back to Studio
      </Link>
    </div>
  );
}
