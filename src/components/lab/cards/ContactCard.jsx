import React from 'react';
import {
  Mail,
  Phone,
  MapPin,
  QrCode,
  Download,
  Linkedin,
  Github,
} from 'lucide-react';
import LogoIcon from '../../LogoIcon';
import { downloadVCard } from '../../../lib/vcard';
import Highlightable from '../Highlightable';
import { Eyebrow, Section, accents } from './primitives';

const ContactRow = ({ icon: Icon, label, value }) => (
  <div className="flex items-center gap-3 rounded-2xl border border-subtle/15 bg-surface-2/60 p-3.5">
    <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-accent/12 text-accent">
      <Icon size={15} />
    </div>
    <div className="min-w-0">
      <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-faint">{label}</div>
      <div className="truncate text-sm font-medium text-ink">
        <Highlightable>{value}</Highlightable>
      </div>
    </div>
  </div>
);

const ContactCard = () => (
  <div className="space-y-6">
    <Eyebrow icon={Mail} color={accents.rose} kicker="Reach Out" title="Contact">
      <Highlightable>
        The easiest thing to do is tap the card and save it — my vCard installs straight into Contacts on iOS, Android, macOS, Gmail, and Outlook.
      </Highlightable>
    </Eyebrow>

    <Section id="vcard">
      <button
        type="button"
        onClick={downloadVCard}
        className="group relative w-full overflow-hidden rounded-[28px] border border-subtle/15 bg-surface-2/70 p-6 text-left transition-colors hover:border-accent/25"
      >
        <LogoIcon
          aria-hidden
          className="pointer-events-none absolute -bottom-6 -right-6 h-40 w-40 text-ink opacity-10"
          strokeWidth={30}
        />
        <div className="relative z-10 flex items-start gap-4">
          <div className="h-20 w-20 overflow-hidden rounded-2xl bg-surface ring-1 ring-subtle/15">
            <img src="/profile_photo.png" alt="Bhavesh Gupta" className="h-full w-full object-cover" />
          </div>
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-3">
              <h4 className="text-2xl font-bold tracking-tight text-ink">Bhavesh Gupta</h4>
              <span className="rounded-full bg-accent/12 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-accent">
                vCard
              </span>
            </div>
            <p className="mt-2 text-sm text-muted">
              <Highlightable>MSCS @ NYU Courant · Graduate Assistant @ NYUAD NY</Highlightable>
            </p>
            <p className="mt-2 text-sm leading-relaxed text-faint">
              <Highlightable>Building AI for problems that matter.</Highlightable>
            </p>
          </div>
          <div className="hidden h-16 w-16 shrink-0 rounded-xl bg-white p-1 md:block">
            <img src="/qrcode1.jpeg" alt="QR code" className="h-full w-full object-contain" />
          </div>
        </div>

        <div className="relative z-10 mt-6 grid gap-3 sm:grid-cols-2">
          <ContactRow icon={Mail} label="Email" value="bg2896@nyu.edu" />
          <ContactRow icon={Phone} label="Phone" value="+1 (201) 492-8876" />
          <ContactRow icon={MapPin} label="Location" value="Manhattan, NY" />
          <ContactRow icon={QrCode} label="Web" value="libralpanda.vercel.app" />
        </div>

        <div className="relative z-10 mt-6 flex flex-wrap items-center justify-between gap-4">
          <div className="text-xs text-faint">
            Adds directly to Contacts on iOS, Android, macOS, Gmail, and Outlook.
          </div>
          <span className="inline-flex items-center gap-2 rounded-full bg-ink px-4 py-2.5 text-sm font-semibold text-app">
            <Download size={15} />
            Save Contact
          </span>
        </div>
      </button>
    </Section>

    <Section id="channels">
      <div className="flex flex-wrap gap-3">
        <a href="mailto:bg2896@nyu.edu" className="glass-chip inline-flex items-center gap-2 rounded-full px-4 py-2.5 text-sm text-ink hover:text-accent">
          <Mail size={14} />
          Email
        </a>
        <a href="https://linkedin.com/in/bhaveshgupta01" target="_blank" rel="noopener noreferrer" className="glass-chip inline-flex items-center gap-2 rounded-full px-4 py-2.5 text-sm text-ink hover:text-accent">
          <Linkedin size={14} />
          LinkedIn
        </a>
        <a href="https://github.com/bhaveshgupta01" target="_blank" rel="noopener noreferrer" className="glass-chip inline-flex items-center gap-2 rounded-full px-4 py-2.5 text-sm text-ink hover:text-accent">
          <Github size={14} />
          GitHub
        </a>
      </div>
    </Section>
  </div>
);

export default ContactCard;
