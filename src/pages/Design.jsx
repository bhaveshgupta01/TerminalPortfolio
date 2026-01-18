import React, { useState } from 'react';
import { 
  Palette, Calendar, Wrench, X, ChevronRight, 
  ArrowLeft, Maximize2, Download 
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { designs } from '../data/designs';

const Design = () => {
  const navigate = useNavigate();
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [lightboxImage, setLightboxImage] = useState(null);

  const categories = [
    { id: 'all', label: 'All Designs', color: 'cyan' },
    { id: 'posters', label: 'Posters', color: 'purple' },
    { id: 'banners', label: 'Banners', color: 'orange' },
    { id: 'certificate', label: 'Certificates', color: 'yellow' },
    { id: 'social-media', label: 'Social Media', color: 'pink' },
    { id: 'invitations', label: 'Invitations', color: 'green' },
  ];

  const getDesignsByCategory = (catId) => designs.filter(d => d.category === catId);

  return (
    <div className="min-h-screen w-full bg-[#141414] text-slate-200 font-sans selection:bg-purple-500/30">
      
      {/* GLOBAL STYLES FOR SCROLLBAR HIDING */}
      <style>{`
        .scrollbar-hide::-webkit-scrollbar { display: none; }
        .scrollbar-hide { -ms-overflow-style: none; scrollbar-width: none; }
      `}</style>

      {/* --- HEADER --- */}
      <div className="sticky top-0 z-40 bg-[#141414]/90 backdrop-blur-xl border-b border-white/5">
        <div className="max-w-[1600px] mx-auto px-6 py-4 flex items-center justify-between">
            <div className="flex items-center gap-4">
                <button 
                  onClick={() => navigate('/')}
                  className="p-2 rounded-full hover:bg-white/10 text-slate-400 hover:text-white transition-all"
                >
                    <ArrowLeft size={20} />
                </button>
                <div>
                    <h1 className="text-xl font-bold text-white flex items-center gap-2">
                      <Palette className="text-purple-400" size={20} /> Design Gallery
                    </h1>
                </div>
            </div>
            
            {/* Category Chips */}
            <div className="flex gap-2 overflow-x-auto scrollbar-hide max-w-[60%] md:max-w-none justify-end">
                {categories.map((cat) => (
                    <button
                        key={cat.id}
                        onClick={() => setSelectedCategory(cat.id)}
                        className={`px-4 py-1.5 rounded-full text-xs font-medium transition-all whitespace-nowrap border ${
                            selectedCategory === cat.id
                            ? `bg-purple-500/20 text-purple-300 border-purple-500/40`
                            : 'bg-[#1E1E1E] text-slate-400 border-white/5 hover:bg-white/10'
                        }`}
                    >
                        {cat.label}
                    </button>
                ))}
            </div>
        </div>
      </div>

      {/* --- MAIN CONTENT --- */}
      <div className="max-w-[1600px] mx-auto p-6 md:p-8 space-y-12 pb-24">
        
        {/* === VIEW 1: NETFLIX CAROUSEL (ALL) === */}
        {selectedCategory === 'all' ? (
             categories.filter(c => c.id !== 'all').map((cat) => {
                const catDesigns = getDesignsByCategory(cat.id);
                if (catDesigns.length === 0) return null;

                return (
                    <section key={cat.id} className="space-y-4 animate-fade-in group/section">
                        <div className="flex items-center justify-between px-2">
                            <h2 className="text-xl md:text-2xl font-bold text-white flex items-center gap-3">
                                <span className={`w-1 md:w-1.5 h-5 md:h-6 rounded-full bg-${cat.color === 'purple' ? 'purple-500' : 'blue-500'}`}></span>
                                {cat.label}
                            </h2>
                            <button 
                                onClick={() => setSelectedCategory(cat.id)}
                                className="text-xs font-bold text-slate-500 hover:text-white flex items-center gap-1 transition-colors opacity-0 group-hover/section:opacity-100"
                            >
                                Explore All <ChevronRight size={14} />
                            </button>
                        </div>

                        {/* HORIZONTAL SCROLL CONTAINER */}
                        <div className="flex overflow-x-auto gap-4 pb-4 scrollbar-hide snap-x snap-mandatory flex-nowrap px-2">
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
        /* === VIEW 2: STANDARD GRID (SPECIFIC CATEGORY) === */
            <div className="animate-fade-in">
                <div className="mb-8 px-2">
                    <h2 className="text-3xl font-bold text-white mb-2">{categories.find(c => c.id === selectedCategory)?.label}</h2>
                    <p className="text-slate-400 text-sm">Browsing collection.</p>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 xl:grid-cols-4 gap-6">
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

      {/* --- LIGHTBOX MODAL --- */}
      {lightboxImage && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/95 p-4 backdrop-blur-md animate-in fade-in duration-200" onClick={() => setLightboxImage(null)}>
          
          <button 
            className="absolute top-6 right-6 p-3 bg-[#1E1E1E] border border-white/10 rounded-full text-white hover:bg-white/10 transition-colors z-50 shadow-xl"
            onClick={() => setLightboxImage(null)}
          >
            <X size={24} />
          </button>
          
          <div 
            className="max-w-6xl w-full max-h-[90vh] flex flex-col md:flex-row bg-[#0a0a0a] border border-white/10 rounded-3xl overflow-hidden shadow-2xl ring-1 ring-white/10" 
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex-1 bg-[#050505] flex items-center justify-center p-4 relative">
               <img 
                 src={lightboxImage.image} 
                 alt={lightboxImage.title} 
                 className="max-w-full max-h-[50vh] md:max-h-[80vh] object-contain shadow-2xl rounded-lg" 
               />
            </div>
            
            <div className="w-full md:w-96 bg-[#141414] border-l border-white/5 p-8 flex flex-col h-full overflow-y-auto">
                <div className="flex items-center gap-2 mb-4">
                     <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-1 rounded bg-purple-500/10 text-purple-400 border border-purple-500/20">
                        {lightboxImage.category}
                     </span>
                     <span className="text-xs text-slate-500 flex items-center gap-1">
                        <Calendar size={12}/> {lightboxImage.date}
                     </span>
                </div>
                <h2 className="text-2xl font-bold text-white mb-4 leading-tight">{lightboxImage.title}</h2>
                <p className="text-slate-400 leading-relaxed text-sm mb-6">{lightboxImage.description}</p>
              
                <div className="mt-auto pt-6 border-t border-white/5 space-y-4">
                     <div>
                        <p className="text-xs text-slate-500 font-bold uppercase mb-2">Tools</p>
                        <div className="flex flex-wrap gap-2">
                            {lightboxImage.tools.map((tool, i) => (
                                <span key={i} className="flex items-center gap-1 text-xs text-slate-300 bg-[#1E1E1E] px-2 py-1 rounded border border-white/10">
                                    <Wrench size={10} /> {tool}
                                </span>
                            ))}
                        </div>
                     </div>
                     <a 
                       href={lightboxImage.image} 
                       download 
                       className="flex items-center justify-center gap-2 w-full py-3 bg-white text-black hover:bg-slate-200 rounded-xl font-bold text-sm transition-colors"
                     >
                        <Download size={16} /> Download Asset
                     </a>
                </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

/* --- FIXED CARD COMPONENT --- */
const DesignCard = ({ design, onClick, isCarousel }) => (
    <div 
        onClick={onClick}
        // KEY FIX: Uses fixed width (w-72/80) if isCarousel, otherwise w-full for grid
        className={`
            group relative bg-[#1E1E1E] border border-white/5 rounded-xl overflow-hidden cursor-pointer 
            transition-all duration-300 hover:shadow-2xl hover:shadow-purple-500/5 hover:border-purple-500/30
            ${isCarousel ? 'w-72 md:w-80 flex-shrink-0 hover:scale-105 hover:z-10' : 'w-full hover:-translate-y-1'}
        `}
    >
        {/* Image Area */}
        <div className="aspect-[16/10] w-full overflow-hidden bg-black relative">
            <div className="absolute inset-0 bg-black/20 group-hover:bg-black/40 transition-colors z-10" />
            
            {/* Hover Badge */}
            <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity z-20">
                <span className="flex items-center gap-2 px-3 py-1.5 bg-white/10 backdrop-blur-md border border-white/20 rounded-full text-white text-xs font-medium transform scale-90 group-hover:scale-100 transition-transform">
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

        {/* Text Area */}
        <div className="p-4">
            <h3 className="text-white font-bold text-sm md:text-base truncate mb-1 group-hover:text-purple-400 transition-colors">
                {design.title}
            </h3>
            <div className="flex items-center justify-between text-xs text-slate-500">
                <span>{design.category}</span>
                <span>{design.date}</span>
            </div>
        </div>
    </div>
);

export default Design;