import React from 'react';
import { BodyShell, Section, Row } from './primitives';

const ROLES = [
  {
    id: 'nyuad',
    role: 'Graduate Assistant',
    org: 'NYU Abu Dhabi — New York Office',
    dates: '2026 – present',
    notes: "Supporting academic and cultural initiatives across NYU's global network, coordinating with leadership and visiting government partners.",
  },
  {
    id: 'tcs',
    role: 'System Engineer',
    org: 'Tata Consultancy Services · Northern Trust',
    dates: '2024 – 2025',
    notes: 'Built a Java/JDBC/PostgreSQL user-provisioning REST API serving 2,600+ users across 26M+ weekly transactions with 99.2% uptime, and shipped a GenAI data-validation pipeline that cut QA effort 30%.',
  },
  {
    id: 'synopsys',
    role: 'Deep Learning Intern',
    org: 'Synopsys Inc. — Bioinformatics',
    dates: '2023',
    notes: 'Trained a PyTorch drug-target binding model over 3.2M protein-drug samples, reaching 86% accuracy and accelerating research validation from weeks to 48 hours.',
  },
  {
    id: 'esec',
    role: 'Android Security Intern',
    org: 'eSec Forte Technologies',
    dates: '2022',
    notes: 'Built a custom Android Auto security scanner, uncovered 15+ zero-day style vulnerabilities, and shipped a Flutter enterprise proof of concept across multiple release cycles.',
  },
];

const ExperienceContent = ({ mat }) => (
  <BodyShell>
    {ROLES.map((r, i) => (
      <Section key={r.id} id={r.id} style={i === ROLES.length - 1 ? { marginBottom: 0 } : undefined}>
        <Row mat={mat} role={r.role} org={r.org} dates={r.dates} notes={r.notes} />
      </Section>
    ))}
  </BodyShell>
);

export default ExperienceContent;
