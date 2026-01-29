import React from 'react';

const Logo = ({ className = "w-8 h-8", textClassName = "font-bold text-xl tracking-tight", showText = true }) => {
    return (
        <div className="flex items-center gap-2 group cursor-pointer">
            <div className={`relative flex items-center justify-center transition-transform duration-500 group-hover:scale-110 ${className}`}>
                {/* Resonance Circles */}
                <div className="absolute inset-0 bg-[var(--color-primary)]/10 animate-ping rounded-full opacity-20"></div>
                <div className="absolute inset-[-4px] border border-[var(--color-primary)]/20 rounded-full opacity-0 group-hover:opacity-100 transition-opacity duration-700"></div>

                <div className="absolute inset-0 bg-[var(--color-primary)]/20 blur-lg rounded-full group-hover:bg-[var(--color-primary)]/30 transition-all duration-500"></div>

                {/* Logo Icon */}
                <svg
                    viewBox="0 0 100 100"
                    fill="none"
                    xmlns="http://www.w3.org/2000/svg"
                    className="relative w-full h-full drop-shadow-[0_0_8px_rgba(0,255,136,0.5)]"
                >
                    {/* Ring */}
                    <rect
                        x="10"
                        y="25"
                        width="80"
                        height="50"
                        rx="25"
                        stroke="currentColor"
                        strokeWidth="8"
                        className="text-[var(--color-primary)] opacity-80"
                    />

                    {/* Waveform Line (Animated Resonance) */}
                    <path
                        d="M20 50 Q 35 30, 50 50 T 80 50"
                        stroke="currentColor"
                        strokeWidth="8"
                        strokeLinecap="round"
                        className="text-[var(--color-neon)] glow-primary"
                    >
                        <animate
                            attributeName="d"
                            dur="2s"
                            repeatCount="indefinity"
                            values="M20 50 Q 35 30, 50 50 T 80 50; M20 50 Q 35 70, 50 50 T 80 50; M20 50 Q 35 30, 50 50 T 80 50"
                        />
                    </path>
                </svg>
            </div>
            {showText && (
                <span className={`${textClassName} text-[var(--color-text-main)] group-hover:text-white transition-colors text-glow-neon`}>
                    Tone
                </span>
            )}
        </div>
    );
};

export default Logo;
