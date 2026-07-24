import type { Metadata } from "next";
import "../styles/globals.css";
import { display, body, mono, label } from "./fonts";
import { SiteNav } from "@/components/SiteNav";

const SITE_URL = "https://diffusionworks.netlify.app";
const DESCRIPTION =
  "A study of stochastic models, numerical pricing methods, calibration and validation — fifteen investigations of how derivative-pricing methods behave, and where they fall short.";

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: {
    default: "DiffusionWorks — Stochastic Derivatives Numerics",
    template: "%s · DiffusionWorks",
  },
  description: DESCRIPTION,
  alternates: { canonical: "/" },
  openGraph: {
    type: "website",
    url: SITE_URL,
    siteName: "DiffusionWorks",
    title: "DiffusionWorks — Stochastic Derivatives Numerics",
    description: DESCRIPTION,
  },
  twitter: {
    card: "summary",
    title: "DiffusionWorks — Stochastic Derivatives Numerics",
    description: DESCRIPTION,
  },
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
                DiffusionWorks is a personal quantitative research project studying stochastic
                models and the numerical methods used to price derivatives.
              </p>
              <a href="https://github.com/VM25/stochastic-lab">GitHub ↗</a>
            </footer>
          </main>
        </div>
      </body>
    </html>
  );
}
