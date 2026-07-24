import type { Metadata } from "next";
import { Section } from "@/components/Section";
import { ExperimentLattice } from "@/components/ExperimentLattice";
import { ExperimentIndex } from "@/components/ExperimentIndex";
import { ORDERED_RECORDS, STATUS_COUNTS } from "@/lib/records.generated";

export const metadata: Metadata = {
  title: "Experiment program",
  description:
    "Fifteen numerical experiments over the DiffusionWorks engine — ten pass, five warning — each traceable to a committed record.",
};

export default function ExperimentsPage() {
  return (
    <div className="page">
      <p className="eyebrow">Stage 03 — computation & error</p>
      <h1>Experiment program</h1>
      <p className="page-lede">
        Fifteen mandatory experiments measure the engine against theory, against independent
        methods, and against its own limits. Ten pass and five warn; none fails. A warning is
        not a malfunction — it marks a caveat that changes what may be quoted, and each is
        stated in full on its own page.
      </p>

      <Section idx="03.0" title="The lattice" id="lattice">
        <ExperimentLattice />
      </Section>

      <Section idx="03.1" title="All experiments" id="index">
        <p>
          {STATUS_COUNTS.pass ?? 0} pass, {STATUS_COUNTS.warning ?? 0} warning. Filter by status,
          then open any experiment for its question, method, result, uncertainty,
          interpretation, limitations, and reproduction command.
        </p>
        <ExperimentIndex records={ORDERED_RECORDS} />
      </Section>
    </div>
  );
}
