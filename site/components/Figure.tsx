import styles from "./Figure.module.css";

// A committed figure, presented with the context DESIGN-SPEC requires of every
// chart: what it shows, and a text alternative for readers who cannot see it. The
// PNG is a pre-rendered engine artifact, lazy-loaded, never recomputed in the page.

export function Figure({
  file,
  alt,
  caption,
  source,
}: {
  file: string;
  alt: string;
  caption: React.ReactNode;
  source?: string;
}) {
  return (
    <figure className={styles.figure}>
      <div className={styles.frame}>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={`/figures/${file}`} alt={alt} loading="lazy" decoding="async" />
      </div>
      <figcaption className={styles.caption}>
        <span className={styles.figLabel}>Figure</span>
        <span className={styles.figText}>{caption}</span>
        <a className={styles.figSource} href={source ?? `/figures/${file}`} download>
          {file}
        </a>
      </figcaption>
    </figure>
  );
}
