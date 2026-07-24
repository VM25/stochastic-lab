export function PageHeader({
  stage,
  title,
  lede,
}: {
  stage: string;
  title: string;
  lede: React.ReactNode;
}) {
  return (
    <header>
      <p className="eyebrow">{stage}</p>
      <h1>{title}</h1>
      <p className="page-lede">{lede}</p>
    </header>
  );
}
