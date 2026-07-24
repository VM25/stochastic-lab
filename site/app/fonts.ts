import { Zilla_Slab, Fira_Sans, Fira_Mono, Source_Code_Pro } from "next/font/google";

// Four roles, each distinct, none from the DESIGN-SPEC forbidden list. next/font
// self-hosts them into the static export, so the site makes no runtime request to
// Google Fonts.

// Display: a geometric slab serif. Technical and instrument-like, deliberately not
// a book serif (which the spec forbids as "academic typesetting").
export const display = Zilla_Slab({
  subsets: ["latin"],
  weight: ["500", "600", "700"],
  variable: "--font-display",
  display: "swap",
});

// Body: a humanist technical sans.
export const body = Fira_Sans({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-body",
  display: "swap",
});

// Data: a monospace with tabular figures, for aligned numerical columns,
// reproduction commands, and code.
export const mono = Fira_Mono({
  subsets: ["latin"],
  weight: ["400", "500", "700"],
  variable: "--font-mono",
  display: "swap",
});

// Labels: used only for fully capitalized eyebrows and metadata, the one place the
// spec permits Source Code Pro.
export const label = Source_Code_Pro({
  subsets: ["latin"],
  weight: ["500", "600"],
  variable: "--font-label",
  display: "swap",
});
