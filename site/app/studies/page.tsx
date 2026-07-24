import type { Metadata } from "next";
import { Section } from "@/components/Section";
import { ExperimentLattice } from "@/components/ExperimentLattice";
import { ExperimentIndex } from "@/components/ExperimentIndex";
import { PageHeader } from "@/components/PageHeader";
import { ORDERED_RECORDS, STATUS_COUNTS } from "@/lib/records.generated";

export const metadata: Metadata = {
  title: "Research studies & results",
  description:
    "Fifteen numerical studies of derivative pricing — ten with clean results, five with flagged caveats worth reading.",
};

export default function StudiesPage() {
  const pass = STATUS_COUNTS.pass ?? 0;
  const warning = STATUS_COUNTS.warning ?? 0;

  return (
    <div className="page">
      <PageHeader
        stage="Research studies & results"
        title="Fifteen studies of what the methods actually do"
        lede="Each study asks one question about a pricing method, answers it with measured evidence, and states plainly what the answer does and does not establish. Ten reach a clean result; five carry a caveat that a careful reader should know before trusting the number."
      />

      <Section idx="A" title="The studies at a glance" id="lattice">
        <p>
          Green marks a clean result, amber a flagged caveat. Select any study to read the
          question it asks, what it found, the evidence, and where the finding stops.
        </p>
        <ExperimentLattice />
      </Section>

      <Section idx="B" title="Browse every study" id="index">
        <p>
          {pass} clean, {warning} flagged. Filter by outcome, then open a study for its full
          account.
        </p>
        <ExperimentIndex records={ORDERED_RECORDS} />
      </Section>
    </div>
  );
}
