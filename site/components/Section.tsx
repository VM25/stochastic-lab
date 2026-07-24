export function Section({
  idx,
  title,
  id,
  children,
}: {
  idx: string;
  title: string;
  id?: string;
  children: React.ReactNode;
}) {
  return (
    <section className="section" id={id} aria-labelledby={id ? `${id}-h` : undefined}>
      <div className="section-head">
        <span className="idx" aria-hidden="true">
          {idx}
        </span>
        <h2 id={id ? `${id}-h` : undefined}>{title}</h2>
      </div>
      {children}
    </section>
  );
}
