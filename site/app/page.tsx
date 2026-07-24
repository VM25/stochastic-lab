import Link from "next/link";
import { Section } from "@/components/Section";
import { StochasticPaths } from "@/components/StochasticPaths";
import { ExperimentLattice } from "@/components/ExperimentLattice";
import { EquationBlock } from "@/components/Equation";
import { STATUS_COUNTS, ORDERED_RECORDS } from "@/lib/records.generated";
import styles from "./page.module.css";

const total = ORDERED_RECORDS.length;

export default function OverviewPage() {
  const pass = STATUS_COUNTS.pass ?? 0;
  const warning = STATUS_COUNTS.warning ?? 0;

  return (
    <div className="page">
      <header className={styles.hero}>
        <p className="eyebrow">A study of stochastic models, numerical pricing methods, calibration and validation</p>
        <h1 className={styles.title}>
          DiffusionWorks
          <span className={styles.titleSub}>Stochastic Derivatives Numerics</span>
        </h1>
        <p className={styles.thesis}>
          Options are priced by modelling the random path a price takes through time. DiffusionWorks
          studies how well the standard mathematical models and the numerical methods used to price
          them actually perform — where they are accurate, where they agree, and where they quietly
          break down.
        </p>

        <StochasticPaths />
        <figure className={styles.heroCaption}>
          <EquationBlock
            tex="dS_t = \mu\,S_t\,dt + \sigma\,S_t\,dW_t"
            describe="The change in the price equals a drift term mu times the price times the time step, plus a volatility term sigma times the price times the increment of a Brownian motion."
          />
          <figcaption>
            Thirty sample paths of the diffusion that underlies every price on this site, with the
            distribution of outcomes it produces sketched at the right.
          </figcaption>
        </figure>
      </header>

      <Section idx="01" title="What DiffusionWorks studies" id="what">
        <p>
          A model turns assumptions about randomness into a price. A numerical method turns that
          model into an actual number a computer can produce. Both introduce error, and the
          interesting question is not whether the error exists but <em>how large it is, how it
          behaves, and when it can be trusted.</em>
        </p>
        <p>
          This project puts those questions to the test. It implements two classical models —
          Black–Scholes and Heston — and the main families of pricing method, then runs fifteen
          controlled studies that measure the error directly, compare independent methods against
          one another, and fit a model to a real options market. The results are reported in full,
          including the cases where a method falls short.
        </p>
        <dl className={styles.ledger} aria-label="Summary of the fifteen studies">
          <div className={styles.ledgerItem}>
            <dt>Studies</dt>
            <dd className="numeric">{total}</dd>
          </div>
          <div className={styles.ledgerItem} data-status="pass">
            <dt>Clean result</dt>
            <dd className="numeric">{pass}</dd>
          </div>
          <div className={styles.ledgerItem} data-status="warning">
            <dt>Flagged caveat</dt>
            <dd className="numeric">{warning}</dd>
          </div>
        </dl>
        <p className={styles.ledgerNote}>
          The five flagged studies are the most useful results here. Each is a place where the
          obvious reading of a number would be wrong — a convergence rate quoted too early, a
          calibration that fits without pinning the model down, a confidence interval trusted where
          it should not be.
        </p>
      </Section>

      <Section idx="02" title="Why it matters" id="why">
        <p>
          Every traded option is priced by a model, and every model is run through a numerical
          method that only approximates the true answer. When the approximation is good, nobody
          notices. When it is not — a barrier watched too coarsely, a variance process pushed to its
          limit, a surface that fits many different models equally well — the price can be confidently
          wrong. Knowing where those boundaries lie is the difference between a number and a
          trustworthy number.
        </p>
      </Section>

      <Section idx="03" title="The studies" id="studies">
        <p>
          Fifteen investigations, each answering one question with measured evidence. Amber marks a
          finding that carries a caveat.
        </p>
        <ExperimentLattice />
        <p className={styles.latticeFoot}>
          <Link href="/studies/">Read the research studies →</Link>
        </p>
      </Section>

      <Section idx="04" title="How the work is done" id="approach">
        <p>
          The project follows a simple discipline. Every result is measured, not asserted. Where a
          rate or a bias cannot be resolved above the noise of the simulation, it is reported as
          unresolved rather than fitted to a confident-looking number. Independent methods are made
          to price the same contract so they can be checked against each other. And a disappointing
          result is kept and explained, never quietly dropped — which is why five of the fifteen
          studies carry a caveat rather than a clean pass.
        </p>
      </Section>

      <Section idx="05" title="Where to go next" id="links">
        <ul className={styles.linkGrid}>
          <li>
            <Link href="/models/">Stochastic models</Link>
            <span>Black–Scholes and Heston, and the regions where each stops being reliable.</span>
          </li>
          <li>
            <Link href="/instruments/">Derivative structures</Link>
            <span>European, Asian, and barrier options, and the geometry of their payoffs.</span>
          </li>
          <li>
            <Link href="/methods/">Numerical methods</Link>
            <span>Simulation, finite differences, sensitivities, and calibration.</span>
          </li>
          <li>
            <Link href="/calibration/">Heston calibration</Link>
            <span>Fitting the model to a real market, and what it does and does not determine.</span>
          </li>
          <li>
            <Link href="/validation/">Validation</Link>
            <span>How every result is checked against theory and independent methods.</span>
          </li>
          <li>
            <Link href="/limitations/">Limitations &amp; findings</Link>
            <span>The five flagged studies, and where each method falls short.</span>
          </li>
        </ul>
      </Section>
    </div>
  );
}
