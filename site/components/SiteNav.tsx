"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { NAV } from "@/lib/nav";
import styles from "./SiteNav.module.css";

function isActive(pathname: string, href: string): boolean {
  if (href === "/") return pathname === "/";
  return pathname.startsWith(href);
}

export function SiteNav() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);

  // Close the mobile menu on navigation.
  useEffect(() => {
    setOpen(false);
  }, [pathname]);

  return (
    <header className={styles.rail}>
      <div className={styles.brand}>
        <Link href="/" className={styles.wordmark} aria-label="DiffusionWorks, home">
          <span className={styles.mark} aria-hidden="true" />
          <span className={styles.wordmarkText}>
            Diffusion<span className={styles.works}>Works</span>
          </span>
        </Link>
        <button
          className={styles.toggle}
          aria-expanded={open}
          aria-controls="site-nav-list"
          onClick={() => setOpen((v) => !v)}
        >
          {open ? "Close" : "Menu"}
        </button>
      </div>

      <nav className={styles.nav} aria-label="Sections">
        <p className={styles.spineLabel}>The pipeline</p>
        <ul id="site-nav-list" className={styles.list} data-open={open}>
          {NAV.map((item) => {
            const active = isActive(pathname, item.href);
            return (
              <li key={item.href}>
                <Link
                  href={item.href}
                  className={styles.link}
                  data-active={active}
                  aria-current={active ? "page" : undefined}
                >
                  <span className={styles.stage} aria-hidden="true">
                    {item.stage}
                  </span>
                  <span className={styles.linkBody}>
                    <span className={styles.linkLabel}>{item.label}</span>
                    <span className={styles.linkBlurb}>{item.blurb}</span>
                  </span>
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>

      <div className={styles.foot}>
        <a href="https://github.com/VM25/stochastic-lab" className={styles.footLink}>
          Source ↗
        </a>
        <span className={styles.footNote}>Read-only over committed artifacts</span>
      </div>
    </header>
  );
}
