import type { MetadataRoute } from "next";
import { ORDERED_RECORDS } from "@/lib/records.generated";

const SITE_URL = "https://diffusionworks.netlify.app";

export default function sitemap(): MetadataRoute.Sitemap {
  const sections = [
    "",
    "models",
    "instruments",
    "methods",
    "studies",
    "calibration",
    "validation",
    "limitations",
  ].map((path) => ({
    url: `${SITE_URL}/${path}`.replace(/\/$/, "") + "/",
    changeFrequency: "yearly" as const,
    priority: path === "" ? 1 : 0.7,
  }));

  const studies = ORDERED_RECORDS.map((r) => ({
    url: `${SITE_URL}/studies/${r.slug}/`,
    changeFrequency: "yearly" as const,
    priority: 0.5,
  }));

  return [...sections, ...studies];
}
