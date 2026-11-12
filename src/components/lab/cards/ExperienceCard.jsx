import React from 'react';
import { Briefcase } from 'lucide-react';
import { Eyebrow, Section, WorkCard, accents } from './primitives';
import Highlightable from '../Highlightable';

const ExperienceCard = () => (
  <div className="space-y-6">
    <Eyebrow icon={Briefcase} color={accents.blue} kicker="Work" title="Experience">
      <Highlightable>
        Enterprise backend delivery, research-heavy AI work, and developer education — each role narrowed what I chose to optimize for in the next one.
      </Highlightable>
    </Eyebrow>

    <div className="space-y-4">
      <Section id="nyuad">
        <WorkCard
          role="Graduate Assistant"
          company="NYU Abu Dhabi — New York Office"
          date="2026 – Present"
          color={accents.rose}
          desc="Supporting academic and cultural initiatives across NYU&rsquo;s global network, coordinating with leadership and visiting government partners."
        />
      </Section>
      <Section id="tcs">
        <WorkCard
          role="System Engineer"
          company="Tata Consultancy Services · Northern Trust"
          date="2024 – 2025"
          color={accents.blue}
          desc="Built a Java, JDBC, and PostgreSQL user-provisioning REST API serving 2,600+ users across 26M+ weekly transactions with 99.2% uptime, and shipped a GenAI data-validation pipeline that cut QA effort 30%."
        />
      </Section>
      <Section id="synopsys">
        <WorkCard
          role="Deep Learning Intern"
          company="Synopsys Inc. — Bioinformatics / Healthcare"
          date="2023"
          color={accents.purple}
          desc="Trained a PyTorch drug-target binding model over 3.2M protein-drug samples, reaching 86% accuracy and accelerating research validation from weeks to 48 hours."
        />
      </Section>
      <Section id="esec">
        <WorkCard
          role="Android Security Intern"
          company="eSec Forte Technologies"
          date="2022"
          color={accents.emerald}
          desc="Built a custom Android Auto security scanner, uncovered 15+ zero-day style vulnerabilities, and shipped a Flutter enterprise proof of concept across multiple release cycles."
        />
      </Section>
    </div>
  </div>
);

export default ExperienceCard;
