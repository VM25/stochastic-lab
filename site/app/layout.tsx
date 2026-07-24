import type { Metadata } from "next";
import "../styles/globals.css";
import { display, body, mono, label } from "./fonts";
import { SiteNav } from "@/components/SiteNav";

export const metadata: Metadata = {
  title: {
    default: "DiffusionWorks — Stochastic Derivatives Numerics",
    template: "%s · DiffusionWorks",
  },
  description:
    "A read-only presentation of the DiffusionWorks stochastic-derivatives engine: fifteen numerical experiments, ten pass and five warning, every number traceable to a committed artifact.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  const fontVars = `${display.variable} ${body.variable} ${mono.variable} ${label.variable}`;
  return (
    <html lang="en" className={fontVars}>
      <body>
        <a className="skip-link" href="#main">
          Skip to content
        </a>
        <div className="shell">
          <SiteNav />
          <main id="main" className="main" tabIndex={-1}>
            {children}
            <footer className="site-footer">
              <p>
                Presentation layer only. Every figure and number derives from the committed
                records at generator commit <code>3846fb3</code>; nothing here recomputes the
                engine.
              </p>
            </footer>
          </main>
        </div>
      </body>
    </html>
  );
}
