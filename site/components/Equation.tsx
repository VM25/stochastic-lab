import katex from "katex";

// Equations are rendered to HTML at build time and shipped as static markup, so the
// browser runs no math library. An accessible text alternative is required: KaTeX
// emits MathML alongside the visual output, and every block equation also carries a
// plain-language `describe` for screen readers and as a chart-style caption.

export function EquationBlock({ tex, describe }: { tex: string; describe: string }) {
  const html = katex.renderToString(tex, {
    displayMode: true,
    throwOnError: true,
    output: "htmlAndMathml",
  });
  return (
    <figure className="equation-block" role="group" aria-label={describe}>
      <div dangerouslySetInnerHTML={{ __html: html }} />
      <figcaption className="sr-only">{describe}</figcaption>
    </figure>
  );
}

export function InlineMath({ tex }: { tex: string }) {
  const html = katex.renderToString(tex, {
    displayMode: false,
    throwOnError: true,
    output: "htmlAndMathml",
  });
  return <span dangerouslySetInnerHTML={{ __html: html }} />;
}
