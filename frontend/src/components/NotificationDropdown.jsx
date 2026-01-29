import { motion, AnimatePresence } from 'framer-motion';
import { Check, X, UserPlus } from 'lucide-react';
import { useRef, useEffect } from 'react';

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

    // Close dropdown when clicking outside
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
                    className="absolute top-full right-0 mt-2 w-80 max-w-[calc(100vw-2rem)] bg-[#0d0d0d] border border-[#2f3335] rounded-xl shadow-2xl overflow-hidden z-50 origin-top-right"
                    style={{ maxHeight: 'calc(100vh - 100px)' }}
                >
                    {/* Header */}
                    <div className="px-4 py-3 border-b border-[#2f3335] bg-[#111111]">
                        <h3 className="text-sm font-bold text-white flex items-center gap-2">
                            <UserPlus className="w-4 h-4 text-[var(--color-primary)]" />
                            Friend Requests
                        </h3>
                    </div>

                    {/* Content */}
                    <div className="max-h-96 overflow-y-auto">
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
                                        initial={{ opacity: 0, x: -20 }}
                                        animate={{ opacity: 1, x: 0 }}
                                        exit={{ opacity: 0, x: 20 }}
                                        transition={springTransition}
                                        className="px-4 py-3 hover:bg-[#1a1a1a] transition-colors flex items-center gap-3"
                                    >
                                        {/* Avatar */}
                                        <div className="w-10 h-10 rounded-full bg-[#2f3335] flex items-center justify-center text-white font-medium text-sm shrink-0">
                                            {request.fromUser?.name?.substring(0, 2).toUpperCase() || 'U'}
                                        </div>

                                        {/* User Info */}
                                        <div className="flex-1 min-w-0">
                                            <p className="text-sm font-medium text-white truncate">
                                                {request.fromUser?.name || 'Unknown User'}
                                            </p>
                                            <p className="text-xs text-[#958d73] truncate">
                                                {request.fromUser?.email}
                                            </p>
                                        </div>

                                        {/* Action Buttons */}
                                        <div className="flex items-center gap-1 shrink-0">
                                            <button
                                                onClick={() => onAccept(request.id, request.fromUserId)}
                                                className="p-2 bg-[var(--color-primary)] hover:bg-[#2d9248] text-black rounded-lg transition-all"
                                                title="Accept"
                                            >
                                                <Check className="w-4 h-4" />
                                            </button>
                                            <button
                                                onClick={() => onReject(request.id)}
                                                className="p-2 bg-transparent border border-red-500/50 hover:bg-red-500/10 text-red-400 rounded-lg transition-all"
                                                title="Reject"
                                            >
                                                <X className="w-4 h-4" />
                                            </button>
                                        </div>
                                    </motion.div>
                                ))}
                            </div>
                        )}
                    </div>

                    {/* Footer */}
                    {requests.length > 0 && (
                        <div className="px-4 py-2 border-t border-[#2f3335] bg-[#111111] text-center">
                            <p className="text-xs text-[#958d73]">
                                {requests.length} pending request{requests.length !== 1 ? 's' : ''}
                            </p>
                        </div>
                    )}
                </motion.div>
            )}
        </AnimatePresence>
    );
}
