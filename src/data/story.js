// Single source of truth for the home page AND the chat assistant's knowledge
// (api/_profile.js builds its prompt from this file). Edit here, both update.

export const ERAS = {
  ipu: { label: 'IPU', name: 'GGSIPU, Delhi', years: '2020 – 2024' },
  tcs: { label: 'TCS', name: 'Tata Consultancy Services', years: '2024 – 2025' },
  nyu: { label: 'NYU', name: 'NYU Courant, New York', years: '2025 – 2027' },
};

export const PERSON = {
  name: 'Bhavesh Gupta',
  line: 'MS Computer Science at NYU Courant. I build AI systems end to end: Android, backend and agents.',
  location: 'Manhattan, New York',
  email: 'bg2896@nyu.edu',
  github: 'https://github.com/bhaveshgupta01',
  linkedin: 'https://linkedin.com/in/bhaveshgupta01',
};

export const CHAPTERS = [
  {
    era: 'ipu',
    title: 'B.Tech, Information Technology',
    place: 'USICT, GGS Indraprastha University, Delhi',
    text: 'Started with Android, ran Google workshops for 50+ students, interned twice and published a first-author paper on mammography classification. CGPA 8.42/10.',
  },
  {
    era: 'tcs',
    title: 'System Engineer',
    place: 'Tata Consultancy Services, Gurugram',
    text: 'Thirteen months on capital-markets software for Northern Trust, a US custodian bank: 26M+ transactions a week, 2,600+ users.',
  },
  {
    era: 'nyu',
    title: 'MS, Computer Science',
    place: 'NYU Courant Institute, New York',
    text: 'Agentic AI, machine-learning research and a lot of hackathons. J. N. Tata Scholar, and Graduate Assistant at the NYU Abu Dhabi New York office.',
  },
];

export const INTERNSHIPS = [
  {
    id: 'esec',
    era: 'ipu',
    company: 'eSec Forte Technologies',
    role: 'Android Security Intern',
    when: 'Jul – Sep 2022',
    text: 'Built an Android Auto security scanner in Kotlin (static and dynamic analysis) that surfaced 15+ data-exfiltration vulnerabilities, and shipped a Flutter proof of concept to 100+ enterprise clients over three release cycles.',
  },
  {
    id: 'synopsys',
    era: 'ipu',
    company: 'Synopsys',
    role: 'Deep Learning Intern, Bioinformatics',
    when: 'Jul – Sep 2023',
    text: 'Trained a PyTorch drug-target binding-affinity model on 3.2M protein-drug samples to 86% accuracy, cutting validation from weeks to about 48 hours.',
  },
  {
    id: 'nyuad',
    era: 'nyu',
    company: 'NYU Abu Dhabi, New York Office',
    role: 'Graduate Assistant, Academic Support',
    when: 'Feb 2026 – present',
    text: 'Support academic and cultural programmes across NYU’s global network at 19 Washington Square North. Over summer 2026, worked on the web and database side of the NYUAD research faculty directory (390 faculty profiles).',
  },
];

export const TCS = {
  role: 'System Engineer',
  company: 'Tata Consultancy Services',
  client: 'Client: Northern Trust, US custodian bank',
  when: 'Jun 2024 – Jul 2025 · Gurugram',
  stats: [
    { value: '26M+', label: 'transactions a week' },
    { value: '2,600+', label: 'users served' },
    { value: '99.2%', label: 'uptime' },
    { value: '30%', label: 'less manual QA' },
  ],
  points: [
    'Built a user-provisioning REST API in Java, JDBC and PostgreSQL, backed by 12+ automated test suites.',
    'Shipped a LangChain + GPT-4 + Selenium data-validation pipeline that caught 15+ critical issues before production.',
    'Worked on securities settlement and SWIFT MT/MX message processing on TCS BaNCS for ISO 20022 readiness, across 8 production releases.',
  ],
};

// era: which line the project belongs to. repo only where a real link exists.
// stats: [value, label] pairs shown large in the project card; award: shown as a badge.
export const PROJECTS = [
  {
    id: 'voicegraph', era: 'nyu', name: 'VoiceGraph',
    context: 'Build With AI NYC · team of 3', stats: [['569', 'nodes'], ['882', 'edges'], ['8', 'query tools']],
    tagline: 'Talk to a 3D knowledge graph',
    text: 'Upload PDFs, videos or URLs and it builds a knowledge graph you can talk to. Gemini Live handles two-way audio while the answer is traced across a Three.js graph of 569 nodes and 882 edges. Three-phase extraction pipeline and a GraphRAG agent with 8 query tools. Built with Pranjal Mishra and Anzhelika Siui at Build With AI NYC.',
    stack: ['Gemini Live', 'LangChain', 'Neo4j', 'Three.js', 'FastAPI'],
    repo: 'https://github.com/pmxlr8/VoiceGraph',
  },
  {
    id: 'sentinel', era: 'nyu', name: 'Sentinel',
    context: 'AWS Summit NYC · Jun 2026 · team of 2', award: '1st place',
    tagline: '1st place, AWS Jam at AWS Summit NYC 2026',
    text: 'A secured AI assistant built against the clock with a teammate I met that morning. The model was the easy half; the win came from wiring Cognito, Verified Permissions and tool-level authorisation so it held up under scoring.',
    stack: ['Amazon Bedrock', 'Bedrock Agents', 'Lambda', 'Cognito', 'Verified Permissions'],
  },
  {
    id: 'war-room', era: 'nyu', name: 'Virtual War Room',
    context: 'Google × Columbia hackathon', award: 'Top Technical Build', stats: [['4', 'AI personas'], ['50+', 'teams competing']],
    tagline: 'Top Technical Build, Google × Columbia',
    text: 'A voice boardroom with four AI personas that debate your documents, raise a hand to interject and then vote. Picked as the strongest technical build out of 50+ teams.',
    stack: ['Gemini Live', 'Google ADK', 'FastAPI', 'React', 'WebSocket'],
    repo: 'https://github.com/bhaveshgupta01/warroom-apple-style',
  },
  {
    id: 'pulse', era: 'nyu', name: 'Pulse',
    context: 'Pulse NYC hackathon', award: '1st place', stats: [['<5s', 'event to live campaign']],
    tagline: '1st place, Pulse NYC hackathon',
    text: 'An AI marketing platform that detects live broadcast events and deploys a targeted campaign in under five seconds.',
    stack: ['Next.js', 'TypeScript', 'Gemini'],
  },
  {
    id: 'market-maker', era: 'nyu', name: 'Market-Maker Agent',
    context: 'Hack the Chip · Oct 2026', stats: [['798', 'on-device generations'], ['6 of 9', 'verified, up from 0 of 18'], ['16', 'simulated markets']],
    tagline: 'Verified AI code generation on AWS Trainium2',
    text: 'Built at Hack the Chip (Amazon Annapurna Labs × NYU Tandon). A model running on a Trainium2 chip writes a market maker’s quoting strategy, a deterministic simulator plays it across 16 markets, and a checker that names the cause of each failure feeds the next attempt. Naming the cause took the pass rate from 0 of 18 to 6 of 9 at equal budget.',
    stack: ['Trainium2', 'vLLM', 'EKS', 'Python'],
    repo: 'https://github.com/yahavb/trainium-agent-labs/pull/21',
    repoLabel: 'Pull request',
  },
  {
    id: 'wage', era: 'nyu', name: 'WAGE.md',
    context: 'Datadog Agentic Engineering Hackathon · May 2026 · solo', award: '2nd place, Nimble track', stats: [['8 hrs', 'start to finish']],
    tagline: '2nd place, Nimble track, Datadog hackathon',
    text: 'A solo eight-hour build at the Datadog Agentic Engineering Hackathon in New York.',
    stack: ['Python', 'Agents'],
    repo: 'https://github.com/bhaveshgupta01/Wage_MD',
  },
  {
    id: 'comparison-pdi', era: 'nyu', name: 'ComparisionPDI',
    context: 'NYU Machine Learning · Spring 2026', stats: [['36', 'controlled runs'], ['88', 'hyperparameter sweeps'], ['27,715', 'binding measurements']],
    tagline: 'How should a transformer fuse drug and protein?',
    text: 'A controlled 2 × 2 comparison of fusion strategies for drug-target binding prediction on BindingDB: 4 variants × 3 splits × 3 seeds, run on NYU’s HPC cluster. No strategy wins everywhere, and late fusion, the field default, is never the best. NYU Machine Learning course project; I wrote and ran all of the code.',
    stack: ['PyTorch', 'Transformers', 'BindingDB', 'NYU HPC'],
    repo: 'https://github.com/bhaveshgupta01/ComparisionPDI',
  },
  {
    id: 'gybuddy', era: 'nyu', name: 'GyBuddy',
    context: 'Google AI hackathon · solo', stats: [['10', 'agent tools'], ['65', 'unit tests'], ['~8,000', 'lines of TypeScript']],
    tagline: 'A running coach that talks back',
    text: 'A React Native app where Gemini coaches your run by voice in real time. It draws heart- and star-shaped routes, notices when you are on a treadmill and has 10 tools for pace, weather and nearby places.',
    stack: ['React Native', 'Expo', 'Gemini Live', 'Google Maps', 'Firebase'],
    repo: 'https://github.com/bhaveshgupta01/GybBuddy',
  },
  {
    id: 'signalflow', era: 'nyu', name: 'SignalFlow',
    context: 'Side project', stats: [['6', 'async triggers'], ['85', 'tools via MCP'], ['7 of 9', 'trades won overnight']],
    tagline: 'An autonomous trading agent',
    text: 'Six async triggers watch prediction markets, whale wallets and funding rates. Gemini evaluates each signal, a hard risk gate decides, and trades run with stop-loss and take-profit. In one overnight paper run it took a $100 virtual wallet to $105.56.',
    stack: ['Python', 'Gemini', 'MCP', 'Hyperliquid', 'Streamlit'],
    repo: 'https://github.com/bhaveshgupta01/signalflow',
  },
  {
    id: 'quantian', era: 'nyu', name: 'QuantIAN',
    context: 'NYU Cloud Computing', stats: [['3', 'clouds']],
    tagline: 'Market analytics across three clouds',
    text: 'A peer-to-peer market analytics system spanning AWS, Azure and GCP, with MQTT IoT feeds, a hash-chained ledger and Isolation Forest anomaly detection. NYU Cloud Computing course project.',
    stack: ['AWS', 'Azure', 'GCP', 'MQTT', 'Python'],
    repo: 'https://github.com/bhaveshgupta01/CrossCloudAnalyser',
  },
  {
    id: 'moma-quest', era: 'nyu', name: 'MoMA Quest',
    context: 'Side project', stats: [['200+', 'artworks']],
    tagline: 'A museum visit as a game',
    text: 'A progressive web app that turns MoMA into quests. Gemini writes challenges from a pool of 200+ artworks, with badges, QR invites and weekly leaderboards.',
    stack: ['Next.js', 'Gemini', 'PWA', 'Firestore'],
    repo: 'https://github.com/bhaveshgupta01/MoMAQuest',
  },
  {
    id: 'split-dnn', era: 'nyu', name: 'Securing the Edge',
    context: 'Edge ML research project',
    tagline: 'Private on-device inference',
    text: 'Splits a neural network between a Raspberry Pi and a server so raw inputs never leave the device, and studies what INT8 quantisation of the edge half does to speed, bandwidth and privacy.',
    stack: ['PyTorch', 'Quantisation', 'Raspberry Pi'],
    repo: 'https://github.com/bhaveshgupta01/efficient-split-dnn-privacy',
  },
  {
    id: 'agentic-rag', era: 'nyu', name: 'Agentic RAG Engine',
    context: 'Side project',
    tagline: 'Retrieval that grades itself',
    text: 'A LangGraph pipeline that scores its own retrieval and falls back to web search when the local documents are not good enough.',
    stack: ['LangGraph', 'ChromaDB', 'Mistral 7B', 'FastAPI'],
    repo: 'https://github.com/bhaveshgupta01/Agentic-RAG-Engine',
  },
  {
    id: 'superbowl', era: 'nyu', name: 'Super Bowl Ad Generator',
    context: 'Side project',
    tagline: 'Event-triggered social posts',
    text: 'Detects big live events and drafts social posts for them in real time, with length, hashtags and call to action tuned by Gemini.',
    stack: ['React', 'TypeScript', 'Gemini'],
    repo: 'https://github.com/bhaveshgupta01/SuperBowlAd',
  },
  {
    id: 'mammography', era: 'ipu', name: 'Mammography Classification',
    context: 'Bachelor’s thesis · ICAIA ’24', award: 'Best Presenter', stats: [['98%', 'classification accuracy']],
    tagline: 'Bachelor’s thesis, published',
    text: 'A comparison of deep transfer-learning backbones for classifying mammograms, reaching 98% accuracy. First-author paper in Proceedings on Engineering Sciences; Best Presenter at ICAIA ’24.',
    stack: ['PyTorch', 'Transfer learning', 'Medical imaging'],
    repo: 'https://doi.org/10.24874/PES.SI.25.03A.007',
    repoLabel: 'Paper',
  },
  {
    id: 'campus-map', era: 'ipu', name: 'IPNav',
    context: 'Google Developer Student Club', stats: [['200+', 'students at orientation']],
    tagline: 'Campus map used by 200+ students',
    text: 'A Jetpack Compose campus map for GGSIPU with step-free route planning that prefers ramps and lifts, and full TalkBack support. Adopted during orientation.',
    stack: ['Kotlin', 'Jetpack Compose', 'Material 3'],
    repo: 'https://github.com/bhaveshgupta01/IPNav',
  },
  {
    id: 'bunkey', era: 'ipu', name: 'BunKey',
    context: 'Android side project',
    tagline: 'A cipher with your own key',
    text: 'An Android encrypter and decrypter built on a custom dual-key cipher, so a message encoded with your key cannot be decoded anywhere else.',
    stack: ['Kotlin', 'Android SDK'],
    repo: 'https://github.com/bhaveshgupta01/BunKey',
  },
  {
    id: 'saste-kapade', era: 'ipu', name: 'Saste Kapade',
    context: 'Flutter side project',
    tagline: 'Clothes rental app',
    text: 'A Flutter app for renting clothes.',
    stack: ['Flutter', 'Dart'],
    repo: 'https://github.com/bhaveshgupta01/cloth_rental-master',
  },
];

export const EXTRAS = {
  ipu: [
    { title: 'Android Lead and Compose Camp Facilitator', org: 'Google Developer Student Club', text: 'Taught Jetpack Compose to 50+ students across 12+ workshops as Google’s official facilitator.' },
    { title: 'Technical Festival Lead', org: 'Infoxpression', text: 'Ran 30+ events across a three-day festival.' },
    { title: 'Student Council and Alumni Committee', org: 'GGSIPU', text: 'Nominated council member; founding committee for a 10,000+ alumni database.' },
    { title: 'Project Lead, “Dor”', org: 'Enactus', text: 'A social-entrepreneurship initiative.' },
  ],
  nyu: [
    { title: 'Volunteer', org: 'Washington Square Halloween Parade', text: 'Helped run the parade, co-organised by NYU.' },
  ],
};

export const HONORS = {
  research: {
    era: 'ipu',
    title: 'Comparative Analysis of Deep Transfer Learning Techniques for Mammographic Image Classification',
    meta: 'First author · Proceedings on Engineering Sciences · Best Presenter, ICAIA ’24',
    link: 'https://doi.org/10.24874/PES.SI.25.03A.007',
  },
  scholarship: {
    era: 'nyu',
    title: 'J. N. Tata Endowment Scholar',
    meta: '2026 – 27 · Merit award for postgraduate study abroad, given to about 100 students a year since 1892.',
  },
  hackathons: [
    { place: '1st', name: 'AWS Jam, AWS Summit NYC', when: 'Jun 2026', project: 'sentinel' },
    { place: '1st', name: 'Pulse NYC', when: '', project: 'pulse' },
    { place: 'Top build', name: 'Google × Columbia', when: '', project: 'war-room' },
    { place: '2nd', name: 'Datadog Agentic Engineering, Nimble track', when: 'May 2026', project: 'wage' },
  ],
};

// Section ids double as scroll targets for the chat assistant.
export const SECTIONS = [
  { id: 'hero', label: 'Hi' },
  { id: 'name', label: 'Bhavesh' },
  { id: 'chapters', label: 'The long road' },
  { id: 'internships', label: 'Internships' },
  { id: 'tcs', label: 'Work' },
  { id: 'projects', label: 'Things I built' },
  { id: 'extras', label: 'Beyond code' },
  { id: 'honors', label: 'Recognition' },
  { id: 'gallery', label: 'Visual work' },
  { id: 'contact', label: 'Say hi' },
];
