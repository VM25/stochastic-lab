// The site reads as a research project: what it studies, the models and instruments,
// the methods, the studies and their results, calibration, validation, and what the
// work concludes. Each entry is a section of that narrative.

export interface NavItem {
  href: string;
  label: string;
  index: string;
  blurb: string;
}

export const NAV: NavItem[] = [
  { href: "/", label: "Overview", index: "01", blurb: "What DiffusionWorks studies" },
  { href: "/models/", label: "Stochastic models", index: "02", blurb: "Black–Scholes and Heston dynamics" },
  { href: "/instruments/", label: "Derivative structures", index: "03", blurb: "The contracts and their payoffs" },
  { href: "/methods/", label: "Numerical methods", index: "04", blurb: "Simulation, PDE, and calibration" },
  { href: "/studies/", label: "Research studies & results", index: "05", blurb: "Fifteen investigations, five caveats" },
  { href: "/calibration/", label: "Heston calibration", index: "06", blurb: "Fitting a model to a real market" },
  { href: "/validation/", label: "Validation", index: "07", blurb: "How the results are checked" },
  { href: "/limitations/", label: "Limitations & findings", index: "08", blurb: "Where the methods fall short" },
];
