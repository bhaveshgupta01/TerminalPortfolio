import React from 'react';

const Logo = ({ className = "" }) => {
    return (
        <svg
            viewBox="0 0 200 200"
            className={className}
            xmlns="http://www.w3.org/2000/svg"
        >
            <defs>
                <linearGradient id="neonGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style={{ stopColor: '#00FF41', stopOpacity: 1 }} />
                    <stop offset="50%" style={{ stopColor: '#FF6B35', stopOpacity: 1 }} />
                    <stop offset="100%" style={{ stopColor: '#F7B801', stopOpacity: 1 }} />
                </linearGradient>
                <filter id="glow">
                    <feGaussianBlur stdDeviation="4" result="coloredBlur" />
                    <feMerge>
                        <feMergeNode in="coloredBlur" />
                        <feMergeNode in="SourceGraphic" />
                    </feMerge>
                </filter>
            </defs>

            {/* B */}
            <path
                d="M 40 50 L 40 150 L 90 150 C 110 150 120 140 120 120 C 120 110 115 105 105 102 C 115 99 120 94 120 84 C 120 64 110 50 90 50 Z M 60 70 L 85 70 C 95 70 100 75 100 84 C 100 93 95 98 85 98 L 60 98 Z M 60 115 L 85 115 C 95 115 100 120 100 128 C 100 136 95 140 85 140 L 60 140 Z"
                fill="url(#neonGradient)"
                filter="url(#glow)"
                opacity="0.9"
            />

            {/* G */}
            <path
                d="M 180 100 C 180 130 165 150 140 150 C 115 150 100 130 100 100 C 100 70 115 50 140 50 C 155 50 165 57 172 70 L 155 80 C 150 72 145 68 140 68 C 127 68 120 80 120 100 C 120 120 127 132 140 132 C 148 132 155 127 158 120 L 140 120 L 140 105 L 178 105 Z"
                fill="url(#neonGradient)"
                filter="url(#glow)"
                opacity="0.9"
            />
        </svg>
    );
};

export default Logo;
