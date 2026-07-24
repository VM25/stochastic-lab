import type { ExperimentRecord } from "@/lib/records.generated";
import styles from "./RecordProvenance.module.css";

// The build provenance and downloadable artifacts for one record. This is what makes
// the page auditable: the reader can fetch the exact JSON and CSV the site rendered,
// with their SHA-256, and see the compiler, flags, and commit that produced them.

export function RecordProvenance({ record }: { record: ExperimentRecord }) {
  const b = record.build_metadata;
  const runtime = record.runtime_seconds;
  const runtimeText = runtime >= 60 ? `${(runtime / 60).toFixed(1)} min` : `${runtime.toFixed(1)} s`;

  return (
    <div className={styles.wrap}>
      <div className={styles.downloads}>
        <a href={`/artifacts/${record.id}.json`} download className={styles.download}>
          <span className={styles.dlName}>{record.id}.json</span>
          <span className={styles.dlKind}>record</span>
        </a>
        <a href={`/artifacts/${record.id}.csv`} download className={styles.download}>
          <span className={styles.dlName}>{record.id}.csv</span>
          <span className={styles.dlKind}>summary table</span>
        </a>
      </div>

      <dl className={styles.meta}>
        <div>
          <dt>Generator commit</dt>
          <dd className="numeric">
            {b.git_commit_short}
            {b.git_dirty ? " (dirty)" : ""}
          </dd>
        </div>
        <div>
          <dt>Compiler</dt>
          <dd>
            {b.compiler_id} {b.compiler_version}, C++{b.cxx_standard}
          </dd>
        </div>
        <div>
          <dt>Flags</dt>
          <dd className="numeric">{b.build_flags}</dd>
        </div>
        <div>
          <dt>Hardware</dt>
          <dd>
            {b.cpu_brand}, {b.logical_cores} cores
          </dd>
        </div>
        <div>
          <dt>Runtime (diagnostic)</dt>
          <dd className="numeric">{runtimeText}</dd>
        </div>
        <div>
          <dt>Record SHA-256</dt>
          <dd className="numeric" title={record.json_sha256}>
            {record.json_sha256.slice(0, 16)}…
          </dd>
        </div>
      </dl>
      <p className={styles.note}>
        Runtime is a non-gating diagnostic: it never determines a status, a ranking, or a
        headline.
      </p>
    </div>
  );
}
