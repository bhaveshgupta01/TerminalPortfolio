import React from 'react';
import { Cpu } from 'lucide-react';
import Highlightable from '../Highlightable';
import { Eyebrow, Section, accents } from './primitives';

const SkillSection = ({ id, title, items, color }) => (
  <Section id={id} className="rounded-2xl border-subtle/15 bg-surface-2/65 p-5">
    <h4 className="text-base font-bold tracking-tight text-ink">{title}</h4>
    <div className="mt-4 flex flex-wrap gap-2">
      {items.map((item) => (
        <span key={item} className={`rounded-full border px-2.5 py-1 text-[11px] font-medium ${color}`}>
          <Highlightable>{item}</Highlightable>
        </span>
      ))}
    </div>
  </Section>
);

const SkillsCard = () => (
  <div className="space-y-6">
    <Eyebrow icon={Cpu} color={accents.amber} kicker="Stack" title="Technical Arsenal">
      <Highlightable>
        The layers I actually ship with — not every library I&apos;ve touched. Grouped so a reader (or the AI) can jump straight to what&apos;s relevant.
      </Highlightable>
    </Eyebrow>

    <div className="grid gap-4 lg:grid-cols-2">
      <SkillSection id="languages" title="Languages" color={accents.rose} items={['Kotlin', 'Java', 'Python', 'TypeScript', 'JavaScript', 'SQL', 'Dart', 'C++']} />
      <SkillSection id="ai-ml" title="AI / ML" color={accents.blue} items={['PyTorch', 'TensorFlow', 'LangChain', 'LangGraph', 'ChromaDB', 'Gemini Live API', 'OpenAI GPT-4', 'Hugging Face']} />
      <SkillSection id="mobile" title="Android &amp; Mobile" color={accents.emerald} items={['Android SDK', 'Jetpack Compose', 'Material 3', 'Coroutines', 'Retrofit', 'Room', 'React Native', 'Flutter']} />
      <SkillSection id="backend" title="Backend &amp; APIs" color={accents.amber} items={['FastAPI', 'Spring (basic)', 'REST', 'PostgreSQL', 'JDBC', 'WebSocket', 'Neo4j', 'OpenAPI']} />
      <SkillSection id="frontend" title="Frontend &amp; Web" color={accents.purple} items={['React', 'Next.js', 'Vite', 'Tailwind', 'Three.js', 'Expo']} />
      <SkillSection id="infra" title="Infra &amp; Tools" color={accents.rose} items={['Docker', 'Git', 'GCP', 'Firebase', 'Cloud Run', 'Railway', 'Linux', 'Figma']} />
    </div>
  </div>
);

export default SkillsCard;
