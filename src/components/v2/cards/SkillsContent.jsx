import React from 'react';
import Highlightable from '../Highlightable';
import { BodyShell, Section, Label } from './primitives';

const GROUPS = [
  { id: 'languages', label: 'Languages', items: ['Kotlin', 'Java', 'Python', 'TypeScript', 'JavaScript', 'SQL', 'Dart', 'C++'] },
  { id: 'ai-ml', label: 'AI / ML', items: ['PyTorch', 'TensorFlow', 'LangChain', 'LangGraph', 'ChromaDB', 'Gemini Live API', 'OpenAI GPT-4', 'Hugging Face'] },
  { id: 'mobile', label: 'Android & Mobile', items: ['Android SDK', 'Jetpack Compose', 'Material 3', 'Coroutines', 'Retrofit', 'Room', 'React Native', 'Flutter'] },
  { id: 'backend', label: 'Backend & APIs', items: ['FastAPI', 'Spring (basic)', 'REST', 'PostgreSQL', 'JDBC', 'WebSocket', 'Neo4j', 'OpenAPI'] },
  { id: 'frontend', label: 'Frontend & Web', items: ['React', 'Next.js', 'Vite', 'Tailwind', 'Three.js', 'Expo'] },
  { id: 'infra', label: 'Infra & Tools', items: ['Docker', 'Git', 'GCP', 'Firebase', 'Cloud Run', 'Railway', 'Linux', 'Figma'] },
];

const SkillsContent = ({ mat }) => (
  <BodyShell>
    {GROUPS.map((g) => (
      <Section key={g.id} id={g.id}>
        <Label mat={mat}>{g.label}</Label>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 4 }}>
          {g.items.map((item) => (
            <span
              key={item}
              style={{
                fontSize: 12,
                padding: '5px 11px',
                borderRadius: 6,
                background: 'rgba(0,0,0,0.25)',
                border: `1px solid ${mat.mute}44`,
                color: mat.ink,
              }}
            >
              <Highlightable>{item}</Highlightable>
            </span>
          ))}
        </div>
      </Section>
    ))}
  </BodyShell>
);

export default SkillsContent;
