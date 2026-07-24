import styles from "./Figure.module.css";

// A study figure, presented with the context a chart needs: what it shows, and a text
// alternative for readers who cannot see it. The image is a rendered result, shown for
// reading — not offered as a file to download.

export function Figure({
  file,
  alt,
  caption,
}: {
  file: string;
  alt: string;
  caption: React.ReactNode;
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
      </figcaption>
    </figure>
  );
}
