import { motion, AnimatePresence } from 'framer-motion';
import { Check, X, UserPlus } from 'lucide-react';
import { useRef, useEffect, useState } from 'react';

const springTransition = {
    type: "spring",
    stiffness: 400,
    damping: 30
};

export default function NotificationDropdown({
    isOpen,
    onClose,
    requests = [],
    onAccept,
    onReject,
    anchorRef
}) {
    const dropdownRef = useRef(null);
    const [coords, setCoords] = useState({ top: 0, left: 0 });

    // 🔥 FIX: Calculate position based on the Bell's location
    useEffect(() => {
        if (isOpen && anchorRef?.current) {
            const rect = anchorRef.current.getBoundingClientRect();
            setCoords({
                // Position it below the bell, aligned to the left of the bell
                top: rect.bottom + 8, 
                left: rect.left - 200 // Offset so it doesn't go off-screen left
            });
        }
    }, [isOpen, anchorRef]);

    useEffect(() => {
        if (!isOpen) return;
        const handleClickOutside = (event) => {
            if (
                dropdownRef.current &&
                !dropdownRef.current.contains(event.target) &&
                anchorRef?.current &&
                !anchorRef.current.contains(event.target)
            ) {
                onClose();
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, [isOpen, onClose, anchorRef]);

    return (
        <AnimatePresence>
            {isOpen && (
                <motion.div
                    ref={dropdownRef}
                    initial={{ opacity: 0, y: -10, scale: 0.95 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: -10, scale: 0.95 }}
                    transition={springTransition}
                    // 🔥 CHANGED: Use 'fixed' instead of 'absolute'
                    className="fixed w-80 bg-[#0d0d0d] border border-[#2f3335] rounded-xl shadow-2xl overflow-hidden z-[9999]"
                    style={{ 
                        top: coords.top,
                        left: coords.left,
                        maxHeight: 'calc(100vh - 100px)',
                        filter: 'drop-shadow(0 20px 25px rgba(0,0,0,0.7))' 
                    }}
                >
                    {/* Header */}
                    <div className="px-4 py-3 border-b border-[#2f3335] bg-[#111111]">
                        <h3 className="text-sm font-bold text-white flex items-center gap-2">
                            <UserPlus className="w-4 h-4 text-[var(--color-primary)]" />
                            Friend Requests
                        </h3>
                    </div>

                    {/* Content */}
                    <div className="max-h-96 overflow-y-auto scrollbar-thin scrollbar-thumb-[#2f3335]">
                        {requests.length === 0 ? (
                            <div className="px-4 py-8 text-center">
                                <UserPlus className="w-10 h-10 text-[#2f3335] mx-auto mb-2 opacity-50" />
                                <p className="text-sm text-[#958d73]">No pending requests</p>
                            </div>
                        ) : (
                            <div className="py-2">
                                {requests.map((request) => (
                                    <motion.div
                                        key={request.id}
                                        className="px-4 py-3 hover:bg-[#1a1a1a] transition-colors flex items-center gap-3"
                                    >
                                        <div className="w-10 h-10 rounded-full bg-[#2f3335] flex items-center justify-center text-white font-medium text-sm shrink-0">
                                            {request.fromUser?.name?.substring(0, 2).toUpperCase() || 'U'}
                                        </div>
                                        <div className="flex-1 min-w-0">
                                            <p className="text-sm font-medium text-white truncate">
                                                {request.fromUser?.name || 'Unknown User'}
                                            </p>
                                            <p className="text-[10px] text-[#958d73] truncate">
                                                {request.fromUser?.email}
                                            </p>
                                        </div>
                                        <div className="flex items-center gap-1 shrink-0">
                                            <button
                                                onClick={() => onAccept(request.id, request.fromUserId)}
                                                className="p-1.5 bg-[var(--color-primary)] hover:bg-[#2d9248] text-black rounded-lg transition-all"
                                            >
                                                <Check className="w-3.5 h-3.5" />
                                            </button>
                                            <button
                                                onClick={() => onReject(request.id)}
                                                className="p-1.5 bg-[#1a1a1a] border border-[#2f3335] hover:border-red-500/50 text-red-400 rounded-lg transition-all"
                                            >
                                                <X className="w-3.5 h-3.5" />
                                            </button>
                                        </div>
                                    </motion.div>
                                ))}
                            </div>
                        )}
                    </div>
                </motion.div>
            )}
        </AnimatePresence>
    );
}