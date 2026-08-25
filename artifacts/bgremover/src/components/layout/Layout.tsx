import { ReactNode } from "react";
import { Navbar } from "./Navbar";

export function Layout({ children }: { children: ReactNode }) {
  return <div className="min-h-[100dvh] bg-background text-foreground">
    <Navbar />
    <main className="md:pl-[250px] min-h-[100dvh]">{children}</main>
  </div>;
}