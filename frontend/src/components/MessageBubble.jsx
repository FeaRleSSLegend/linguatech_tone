import { motion } from 'framer-motion';
import { useState, useRef } from 'react';

/**
 * MessageBubble Component
 * Displays a single chat message with appropriate styling based on sender type
 *
 * @param {Object} props
 * @param {Object} props.message - The message object from backend/context
 * @param {boolean} props.isOwn - Whether the message was sent by the current user
 * @param {Function} props.onReply - Callback when user wants to reply to this message
 */
export default function MessageBubble({ message, isOwn, onReply }) {
    if (!message) return null;

    const { text, timestamp, analysis, sender } = message;
    const [showReplyPrompt, setShowReplyPrompt] = useState(false);
    const longPressTimerRef = useRef(null);
    const status = analysis?.label || 'safe';
    const emoji = analysis?.emoji || '';

    // Status colors for the indicator dot
    const statusColors = {
        safe: 'bg-[var(--color-safe)]',
        warning: 'bg-yellow-500',
        toxic: 'bg-red-500',
        positive: 'bg-[var(--color-safe)]',
        neutral: 'bg-gray-500'
    };

    const formattedTime = new Date(timestamp).toLocaleTimeString([], {
        hour: '2-digit',
        minute: '2-digit'
    });

    // Right-click handler (PC)
    const handleContextMenu = (e) => {
        e.preventDefault();
        if (onReply) {
            onReply(message);
        }
    };

    // Long-press handlers (Mobile)
    const handleTouchStart = () => {
        longPressTimerRef.current = setTimeout(() => {
            setShowReplyPrompt(true);
            if (onReply) {
                onReply(message);
            }
        }, 500); // 500ms long press
    };

    const handleTouchEnd = () => {
        if (longPressTimerRef.current) {
            clearTimeout(longPressTimerRef.current);
        }
    };

    return (
        <motion.div
            initial={{ opacity: 0, y: 10, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            transition={{ duration: 0.2 }}
            className={`flex ${isOwn ? 'justify-end' : 'justify-start'} mb-4`}
        >
            <div
                onContextMenu={handleContextMenu}
                onTouchStart={handleTouchStart}
                onTouchEnd={handleTouchEnd}
                className={`
                    relative max-w-[80%] md:max-w-[70%] rounded-2xl px-4 py-2.5 shadow-sm cursor-pointer
                    hover:ring-1 hover:ring-[var(--color-primary)]/30 transition-all
                    ${isOwn
                        ? 'bg-[#2a2a2a] text-[#e0ddd9] rounded-tr-sm'
                        : 'bg-[#1a1a1a] text-[#e0ddd9] border border-[#2f3335] rounded-tl-sm'
                    }
                `}
            >
                {/* Sender Name (for group chats) */}
                {!isOwn && sender?.name && (
                    <p className="text-[#34a853] text-[10px] mb-1 font-black uppercase tracking-wider">{sender.name}</p>
                )}

                {/* Message Content */}
                <p className="text-sm leading-relaxed break-words">
                    {text}
                </p>

                {/* Footer: Timestamp & Status */}
                <div className={`flex items-center gap-2 mt-1.5 ${isOwn ? 'justify-end' : 'justify-start'}`}>
                    {emoji && <span className="text-xs">{emoji}</span>}
                    <span className="text-[#5a5a5a] text-[10px] font-medium">{formattedTime}</span>
                    <span className={`w-1.5 h-1.5 rounded-full ${statusColors[status] || 'bg-[#5a5a5a]'}`} />
                </div>
            </div>
        </motion.div>
    );
}
