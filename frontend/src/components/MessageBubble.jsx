import { motion } from 'framer-motion';
import { useState, useRef } from 'react';

export default function MessageBubble({ message, isOwn, onReply }) {
    if (!message) return null;

    const { text, timestamp, analysis, sender, replyTo } = message;
    const longPressTimerRef = useRef(null);
    
    // Status and Toxic detection - expanded to catch all negative labels
    const status = analysis?.label?.toLowerCase() || 'safe';
    const isToxic = ['toxic', 'warning', 'negative', 'extreme'].includes(status);

    const statusColors = {
        safe: 'bg-[var(--color-safe)]',
        warning: 'bg-yellow-500',
        toxic: 'bg-red-500',
        positive: 'bg-[var(--color-safe)]',
        neutral: 'bg-gray-500'
    };

    // Standardize timestamp handling
    const msgDate = timestamp?.toDate ? timestamp.toDate() : new Date(timestamp);
    const formattedTime = msgDate.toLocaleTimeString([], {
        hour: '2-digit',
        minute: '2-digit',
        hour12: true
    });

    const handleContextMenu = (e) => {
        e.preventDefault();
        if (onReply) onReply(message);
    };

    return (
        <motion.div
            initial={{ opacity: 0, y: 10, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            transition={{ duration: 0.2 }}
            className={`flex ${isOwn ? 'justify-end' : 'justify-start'} mb-4 px-4`}
        >
            <div
                onContextMenu={handleContextMenu}
                className={`
                    relative max-w-[85%] md:max-w-[70%] rounded-2xl px-4 py-2.5 shadow-sm cursor-pointer
                    transition-all duration-300
                    ${isToxic 
                        ? 'border-2 border-red-500 bg-red-900/20 shadow-[0_0_15px_rgba(239,68,68,0.3)]' 
                        : isOwn 
                            ? 'bg-[#2a2a2a] text-[#e0ddd9] rounded-tr-sm' 
                            : 'bg-[#1a1a1a] text-[#e0ddd9] border border-[#2f3335] rounded-tl-sm'
                    }
                `}
            >
                {/* 1. Toxic Label (Resonance Shield) */}
                {isToxic && (
                    <div className="text-[9px] font-black text-red-500 uppercase tracking-tighter mb-1 flex items-center gap-1">
                        <div className="w-1 h-1 bg-red-500 rounded-full animate-pulse" />
                        Resonance Shield Active
                    </div>
                )}

                {/* 2. WhatsApp-Style Quoted Reply Block */}
                {replyTo && (
                    <div className="mb-2 flex items-stretch bg-black/25 rounded-lg overflow-hidden border-l-4 border-[var(--color-primary)]">
                        <div className="p-2 py-1 flex flex-col min-w-0">
                            <span className="text-[10px] font-bold text-[var(--color-primary)] truncate">
                                {replyTo.senderName || 'User'}
                            </span>
                            <span className="text-xs italic text-gray-400 line-clamp-1 break-all">
                                {replyTo.text}
                            </span>
                        </div>
                    </div>
                )}

                {/* 3. Sender Name (For Groups) */}
                {!isOwn && sender?.name && !isToxic && (
                    <p className="text-[#34a853] text-[10px] mb-1 font-black uppercase tracking-wider">
                        {sender.name}
                    </p>
                )}

                {/* 4. Message Content */}
                <p className={`text-sm leading-relaxed break-words ${isToxic ? 'text-red-100 font-medium' : ''}`}>
                    {text}
                </p>

                {/* 5. Footer: Time & Status Dot */}
                <div className={`flex items-center gap-2 mt-1.5 ${isOwn ? 'justify-end' : 'justify-start'}`}>
                    <span className="text-[#5a5a5a] text-[10px] font-medium">{formattedTime}</span>
                    <span className={`w-1.5 h-1.5 rounded-full ${statusColors[status] || 'bg-[#5a5a5a]'}`} />
                </div>
            </div>
        </motion.div>
    );
}