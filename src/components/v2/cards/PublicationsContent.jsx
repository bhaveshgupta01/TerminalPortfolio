import React from 'react';
import Highlightable from '../Highlightable';
import { BodyShell, Section } from './primitives';

const PublicationsContent = ({ mat }) => (
  <BodyShell>
    <Section id="icaia-mammography">
      <a
        href="https://doi.org/10.24874/PES.SI.25.03A.007"
        target="_blank"
        rel="noopener noreferrer"
        style={{
          display: 'block',
          padding: 16,
          borderRadius: 12,
          background: 'rgba(0,0,0,0.28)',
          border: `1px solid ${mat.mute}33`,
          color: 'inherit',
          textDecoration: 'none',
        }}
      >
        <div
          style={{
            fontFamily: "'Inter Tight', sans-serif",
            fontSize: 15,
            fontWeight: 600,
            fontStyle: 'italic',
            color: mat.ink,
            lineHeight: 1.4,
            marginBottom: 8,
          }}
        >
          <Highlightable>
            &quot;Comparative Analysis of Deep Transfer Learning Techniques for Mammographic Image Classification&quot;
          </Highlightable>
        </div>
        <div style={{ fontSize: 12, color: mat.mute, marginBottom: 6 }}>
          <Highlightable>Proceedings on Engineering Sciences · ICAIA 2024</Highlightable>
          {' · '}
          <span style={{ color: mat.accent }}>
            <Highlightable>Best Presenter Award</Highlightable>
          </span>
        </div>
        <div style={{ fontSize: 12, color: mat.ink, opacity: 0.78, lineHeight: 1.55, marginBottom: 12 }}>
          <Highlightable>
            Compared ResNet50, VGG16, and InceptionV3 on CBIS-DDSM; ResNet50 reached 98% classification accuracy after fine-tuning the final blocks.
          </Highlightable>
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
          {['Transfer Learning', 'ResNet50', 'Medical Imaging', 'DOI-indexed'].map((tag) => (
            <span
              key={tag}
              style={{
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: 9,
                letterSpacing: 0.5,
                padding: '4px 9px',
                borderRadius: 999,
                border: `1px solid ${mat.mute}55`,
                color: mat.accent,
              }}
            >
              <Highlightable>{tag}</Highlightable>
            </span>
          ))}
        </div>
      </a>
    </Section>
  </BodyShell>
);

export default PublicationsContent;
