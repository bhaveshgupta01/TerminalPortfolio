import React from 'react';
import { BookOpen, ExternalLink } from 'lucide-react';
import Highlightable from '../Highlightable';
import { Eyebrow, Section, accents } from './primitives';

const PublicationsCard = () => (
  <div className="space-y-6">
    <Eyebrow icon={BookOpen} color={accents.purple} kicker="Research" title="Publications">
      <Highlightable>
        Peer-reviewed research where I learned what it means to defend a result, not just ship one.
      </Highlightable>
    </Eyebrow>

    <Section id="icaia-mammography">
      <a
        href="https://doi.org/10.24874/PES.SI.25.03A.007"
        target="_blank"
        rel="noopener noreferrer"
        className="group block rounded-[28px] border border-subtle/15 bg-surface-2/70 p-6 transition-colors hover:border-violet-500/30"
      >
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="text-lg font-bold italic tracking-tight text-ink transition-colors group-hover:text-violet-400">
              <Highlightable>
                &quot;Comparative Analysis of Deep Transfer Learning Techniques for Mammographic Image Classification&quot;
              </Highlightable>
            </p>
            <p className="mt-3 text-sm text-muted">
              <Highlightable>Proceedings on Engineering Sciences · ICAIA 2024</Highlightable>
              {' · '}
              <span className="text-violet-500 dark:text-violet-400">
                <Highlightable>Best Presenter Award</Highlightable>
              </span>
            </p>
            <p className="mt-3 text-[13px] leading-relaxed text-faint">
              <Highlightable>
                Compared ResNet50, VGG16, and InceptionV3 on CBIS-DDSM; ResNet50 reached 98% classification accuracy after fine-tuning the final blocks.
              </Highlightable>
            </p>
          </div>
          <ExternalLink size={18} className="shrink-0 text-faint transition-colors group-hover:text-violet-400" />
        </div>
        <div className="mt-5 flex flex-wrap gap-2">
          {['Transfer Learning', 'ResNet50', 'Medical Imaging', 'DOI-indexed'].map((item) => (
            <span key={item} className={`rounded-full border px-2.5 py-1 text-[11px] font-medium ${accents.purple}`}>
              <Highlightable>{item}</Highlightable>
            </span>
          ))}
        </div>
      </a>
    </Section>
  </div>
);

export default PublicationsCard;
