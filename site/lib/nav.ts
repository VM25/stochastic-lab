// The primary navigation is the project's own pipeline, made into a table of
// contents: Model -> Numerical Method -> Computation -> Error -> Validation ->
// Evidence. Each stage names a real page, and the meta pages follow.

export interface NavItem {
  href: string;
  label: string;
  stage: string;
  blurb: string;
}

export const NAV: NavItem[] = [
  { href: "/", label: "Overview", stage: "00", blurb: "What was built, and what it found" },
  { href: "/models/", label: "Models & instruments", stage: "01", blurb: "Black–Scholes, Heston, and the contracts" },
  { href: "/methods/", label: "Numerical methods", stage: "02", blurb: "Simulation, PDE, Greeks, calibration" },
  { href: "/experiments/", label: "Experiment program", stage: "03", blurb: "Fifteen experiments, ten pass, five warn" },
  { href: "/validation/", label: "Validation", stage: "04", blurb: "References, invariants, coverage" },
  { href: "/calibration/", label: "Calibration & market surface", stage: "05", blurb: "A good fit that identifies little" },
  { href: "/architecture/", label: "Architecture & reproducibility", stage: "06", blurb: "How a number traces to a commit" },
  { href: "/limitations/", label: "Limitations & failure regions", stage: "07", blurb: "Where the numerics fall short" },
];

export const SPINE = ["Model", "Numerical method", "Computation", "Error", "Validation", "Evidence"];
