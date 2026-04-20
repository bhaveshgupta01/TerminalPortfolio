import React, { useState } from 'react';
import {
  Palette, Calendar, Wrench, X, ChevronRight,
  ArrowLeft, Maximize2, Download,
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { designs } from '../data/designs';
import ThemeToggle from '../components/ThemeToggle';
import BackgroundAmbient from '../components/BackgroundAmbient';

const CATEGORIES = [
  { id: 'all',          label: 'All Designs' },
  { id: 'posters',      label: 'Posters' },
  { id: 'banners',      label: 'Banners' },
  { id: 'certificate',  label: 'Certificates' },
  { id: 'social-media', label: 'Social Media' },
  { id: 'invitations',  label: 'Invitations' },
];

const Design = () => {
  const navigate = useNavigate();
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [lightboxImage, setLightboxImage] = useState(null);

  const getDesignsByCategory = (catId) => designs.filter(d => d.category === catId);

  return (
    <div className="min-h-screen w-full text-ink font-sans selection:bg-accent/25 relative">
      <BackgroundAmbient />

      {/* ===== HEADER (consistent with Home) ===== */}
      <div className="sticky top-0 z-40 glass-strong border-b border-subtle/15">
        <div className="max-w-[1600px] mx-auto px-4 md:px-6 py-3 flex items-center justify-between gap-3">
            <div className="flex items-center gap-3 relative z-10 min-w-0">
                <button
                  onClick={() => navigate('/')}
                  className="glass-chip w-9 h-9 rounded-full flex items-center justify-center text-muted hover:text-ink transition-colors shrink-0"
                  aria-label="Back"
                >
                    <ArrowLeft size={15} />
                </button>
                <div className="min-w-0">
                    <h1 className="text-base md:text-lg font-bold text-ink flex items-center gap-2 tracking-tight">
                      <Palette className="text-accent" size={18} />
                      <span>Design <span className="text-accent">Gallery</span></span>
                    </h1>
                </div>
            </div>

            <div className="hidden md:flex gap-1.5 overflow-x-auto scrollbar-hide relative z-10">
                {CATEGORIES.map((cat) => (
                    <motion.button
                        key={cat.id}
                        onClick={() => setSelectedCategory(cat.id)}
                        whileTap={{ scale: 0.96 }}
                        className={`px-3.5 py-1.5 rounded-full text-xs font-medium transition-all whitespace-nowrap border ${
                            selectedCategory === cat.id
                            ? 'bg-accent/15 text-accent border-accent/30'
                            : 'glass-chip text-muted hover:text-ink'
                        }`}
                    >
                        {cat.label}
                    </motion.button>
                ))}
            </div>

            <ThemeToggle />
        </div>

        {/* Mobile category chips (own row) */}
        <div className="md:hidden px-4 pb-3 overflow-x-auto scrollbar-hide">
          <div className="flex gap-1.5 whitespace-nowrap">
            {CATEGORIES.map((cat) => (
              <button
                key={cat.id}
                onClick={() => setSelectedCategory(cat.id)}
                className={`px-3 py-1.5 rounded-full text-[11px] font-medium transition-all whitespace-nowrap border ${
                  selectedCategory === cat.id
                  ? 'bg-accent/15 text-accent border-accent/30'
                  : 'glass-chip text-muted'
                }`}
              >
                {cat.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* ===== MAIN CONTENT ===== */}
      <div className="max-w-[1600px] mx-auto p-4 md:p-8 space-y-12 pb-24">

        {/* ALL-VIEW: Netflix-style horizontal rails per category */}
        {selectedCategory === 'all' ? (
             CATEGORIES.filter(c => c.id !== 'all').map((cat) => {
                const catDesigns = getDesignsByCategory(cat.id);
                if (catDesigns.length === 0) return null;

                return (
                    <section key={cat.id} className="space-y-4 animate-fade-in group/section">
                        <div className="flex items-center justify-between px-1">
                            <h2 className="text-lg md:text-2xl font-bold text-ink flex items-center gap-3 tracking-tight">
                                <span className="w-1 h-5 md:h-6 rounded-full bg-gradient-to-b from-rose-400 to-amber-300"></span>
                                {cat.label}
                            </h2>
                            <button
                                onClick={() => setSelectedCategory(cat.id)}
                                className="text-[11px] font-semibold text-faint hover:text-ink flex items-center gap-1 transition-colors opacity-0 group-hover/section:opacity-100"
                            >
                                Explore All <ChevronRight size={13} />
                            </button>
                        </div>

                        <div className="flex overflow-x-auto gap-4 pb-4 scrollbar-hide snap-x snap-mandatory flex-nowrap px-1">
                            {catDesigns.map((design) => (
                                <div key={design.id} className="snap-start flex-shrink-0">
                                    <DesignCard
                                      design={design}
                                      isCarousel={true}
                                      onClick={() => setLightboxImage(design)}
                                    />
                                </div>
                            ))}
                        </div>
                    </section>
                )
             })
        ) : (
            <div className="animate-fade-in">
                <div className="mb-8 px-1">
                    <h2 className="text-2xl md:text-3xl font-bold text-ink mb-1 tracking-tight">
                      {CATEGORIES.find(c => c.id === selectedCategory)?.label}
                    </h2>
                    <p className="text-muted text-sm">Browsing collection.</p>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 xl:grid-cols-4 gap-5">
                    {getDesignsByCategory(selectedCategory).map((design) => (
                        <DesignCard
                          key={design.id}
                          design={design}
                          isCarousel={false}
                          onClick={() => setLightboxImage(design)}
                        />
                    ))}
                </div>
            </div>
        )}

      </div>

      {/* ===== LIGHTBOX MODAL (glass) ===== */}
      <AnimatePresence>
        {lightboxImage && (
          <motion.div
            className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 p-4 backdrop-blur-xl"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            onClick={() => setLightboxImage(null)}
          >
            <button
              className="absolute top-5 right-5 glass-chip w-10 h-10 rounded-full flex items-center justify-center text-ink hover:text-accent transition-colors z-50"
              onClick={() => setLightboxImage(null)}
              aria-label="Close"
            >
              <X size={18} />
            </button>

            <motion.div
              initial={{ opacity: 0, scale: 0.96 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.96 }}
              transition={{ type: 'spring', stiffness: 280, damping: 26 }}
              className="max-w-6xl w-full max-h-[90vh] flex flex-col md:flex-row glass-strong rounded-3xl overflow-hidden"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex-1 bg-black/30 flex items-center justify-center p-4 relative">
                 <img
                   src={lightboxImage.image}
                   alt={lightboxImage.title}
                   className="max-w-full max-h-[50vh] md:max-h-[80vh] object-contain shadow-2xl rounded-lg"
                 />
              </div>

              <div className="w-full md:w-96 border-l border-subtle/15 p-6 md:p-8 flex flex-col h-full overflow-y-auto relative z-10">
                  <div className="flex items-center gap-2 mb-4 flex-wrap">
                       <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-1 rounded bg-accent/15 text-accent border border-accent/25">
                          {lightboxImage.category}
                       </span>
                       <span className="text-xs text-muted flex items-center gap-1">
                          <Calendar size={12}/> {lightboxImage.date}
                       </span>
                  </div>
                  <h2 className="text-xl md:text-2xl font-bold text-ink mb-3 leading-tight tracking-tight">{lightboxImage.title}</h2>
                  <p className="text-muted leading-relaxed text-[13.5px] mb-6">{lightboxImage.description}</p>

                  <div className="mt-auto pt-6 border-t border-subtle/15 space-y-4">
                       <div>
                          <p className="text-[10px] text-faint font-bold uppercase mb-2 tracking-wider">Tools</p>
                          <div className="flex flex-wrap gap-1.5">
                              {lightboxImage.tools.map((tool, i) => (
                                  <span key={i} className="flex items-center gap-1 text-[11px] text-ink glass-chip px-2 py-1 rounded">
                                      <Wrench size={10} /> {tool}
                                  </span>
                              ))}
                          </div>
                       </div>
                       <a
                         href={lightboxImage.image}
                         download
                         className="flex items-center justify-center gap-2 w-full py-3 bg-ink text-app hover:opacity-90 rounded-xl font-bold text-sm transition-opacity"
                       >
                          <Download size={16} /> Download Asset
                       </a>
                  </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

/* ---------------- CARD ---------------- */
const DesignCard = ({ design, onClick, isCarousel }) => (
    <motion.div
        onClick={onClick}
        whileHover={{ y: isCarousel ? 0 : -4, scale: isCarousel ? 1.03 : 1 }}
        transition={{ type: 'spring', stiffness: 280, damping: 24 }}
        className={`
            group relative bg-surface-2 border border-subtle/15 rounded-2xl overflow-hidden cursor-pointer
            transition-colors duration-300 hover:border-accent/30 hover:shadow-2xl hover:shadow-accent/10
            ${isCarousel ? 'w-72 md:w-80 flex-shrink-0 hover:z-10' : 'w-full'}
        `}
    >
        <div className="aspect-[16/10] w-full overflow-hidden bg-black relative">
            <div className="absolute inset-0 bg-black/20 group-hover:bg-black/40 transition-colors z-10" />

            <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity z-20">
                <span className="glass-chip flex items-center gap-1.5 px-3 py-1.5 rounded-full text-ink text-xs font-medium transform scale-90 group-hover:scale-100 transition-transform">
                    <Maximize2 size={12} /> View
                </span>
            </div>

            <img
                src={design.image}
                alt={design.title}
                className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-110 opacity-90 group-hover:opacity-100"
                loading="lazy"
            />
        </div>

        <div className="p-4">
            <h3 className="text-ink font-bold text-sm md:text-base truncate mb-1 group-hover:text-accent transition-colors tracking-tight">
                {design.title}
            </h3>
            <div className="flex items-center justify-between text-xs text-faint">
                <span className="capitalize">{design.category}</span>
                <span>{design.date}</span>
            </div>
        </div>
    </motion.div>
);

export default Design;
