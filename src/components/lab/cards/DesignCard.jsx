import React from 'react';
import { Link } from 'react-router-dom';
import { Palette } from 'lucide-react';
import { designs } from '../../../data/designs';
import Highlightable from '../Highlightable';
import { Eyebrow, Section, accents } from './primitives';

const featured = designs.slice(0, 4);
const categoryCounts = Object.entries(
  designs.reduce((acc, d) => {
    acc[d.category] = (acc[d.category] || 0) + 1;
    return acc;
  }, {}),
).map(([category, count]) => ({ category, count }));

const DesignCard = () => (
  <div className="space-y-6">
    <Eyebrow icon={Palette} color={accents.amber} kicker="Visual Work" title="Design Gallery">
      <Highlightable>
        The visual side of the work — posters, banners, and social assets I designed for events, clubs, and campaigns, compressed into a preview strip with a jump to the full archive.
      </Highlightable>
    </Eyebrow>

    <Section id="categories">
      <div className="flex flex-wrap gap-2">
        {categoryCounts.map(({ category, count }) => (
          <span key={category} className="rounded-full border border-subtle/15 bg-surface-2/70 px-3 py-1.5 text-xs font-medium text-ink">
            <Highlightable>{category}</Highlightable>
            {' '}
            <span className="text-faint">({count})</span>
          </span>
        ))}
      </div>
    </Section>

    <Section id="featured">
      <div className="grid gap-4 lg:grid-cols-2">
        {featured.map((design) => (
          <div key={design.id} className="overflow-hidden rounded-[24px] border border-subtle/15 bg-surface-2/70">
            <div className="aspect-[16/10] overflow-hidden bg-black/20">
              <img src={design.image} alt={design.title} className="h-full w-full object-cover" loading="lazy" />
            </div>
            <div className="p-4">
              <div className="text-sm font-bold tracking-tight text-ink">
                <Highlightable>{design.title}</Highlightable>
              </div>
              <div className="mt-1 text-xs uppercase tracking-[0.18em] text-faint">{design.category}</div>
              <p className="mt-3 text-[13px] leading-relaxed text-muted">
                <Highlightable>{design.description}</Highlightable>
              </p>
            </div>
          </div>
        ))}
      </div>
    </Section>

    <div className="flex flex-wrap gap-3">
      <Link
        to="/design"
        className="inline-flex items-center gap-2 rounded-full bg-ink px-4 py-2.5 text-sm font-semibold text-app transition-opacity hover:opacity-90"
      >
        <Palette size={15} />
        Open full design gallery
      </Link>
      <span className="inline-flex items-center gap-2 rounded-full border border-subtle/15 px-4 py-2.5 text-sm text-muted">
        Canva, Photoshop, Illustrator, CorelDraw, Figma
      </span>
    </div>
  </div>
);

export default DesignCard;
