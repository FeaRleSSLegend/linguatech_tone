import { motion, AnimatePresence } from 'framer-motion';

/**
 * SuggestionCard Component
 * Modal popup that shows when a message is flagged as toxic
 */
export default function SuggestionCard({
    isVisible,
    originalMessage,
    suggestion,
    reason,
    onUseSuggestion,
    onEdit,
    onSendAnyway,
    onClose
}) {
    return (
        <AnimatePresence>
            {isVisible && (
                <>
                    {/* Backdrop */}
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        onClick={onClose}
                        className="fixed inset-0 bg-black/60 backdrop-blur-sm z-40"
                    />

                    {/* Modal */}
                    <motion.div
                        initial={{ opacity: 0, y: 50, scale: 0.95 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: 50, scale: 0.95 }}
                        transition={{ type: 'spring', damping: 25, stiffness: 300 }}
                        className="fixed bottom-4 left-4 right-4 md:left-1/2 md:right-auto md:-translate-x-1/2 md:bottom-auto md:top-1/2 md:-translate-y-1/2 md:w-full md:max-w-lg z-50"
                    >
                        <div className="bg-[#0d0d0d] border border-[#2f3335] rounded-2xl p-6 shadow-2xl">
                            {/* Header */}
                            <div className="flex items-center gap-3 mb-4">
                                <h3 className="text-[#e0ddd9] text-lg font-semibold">Safety Suggestion</h3>
                            </div>

                            {/* Original Message */}
                            <div className="mb-4">
                                <p className="text-[#958d73] text-xs uppercase tracking-wider mb-2">Original Message</p>
                                <p className="text-[#e0ddd9] bg-[#1a1a1a] border border-red-500/30 rounded-lg p-3 text-sm">
                                    {originalMessage}
                                </p>
                            </div>

                            {/* Reason */}
                            <div className="mb-4">
                                <p className="text-[#958d73] text-xs uppercase tracking-wider mb-2">Why it was flagged</p>
                                <p className="text-red-400 text-sm">{reason}</p>
                            </div>

                            {/* Suggested Rephrase (ONE suggestion only) */}
                            <div className="mb-6">
                                <p className="text-[#958d73] text-xs uppercase tracking-wider mb-2">Suggested alternative</p>
                                {suggestion ? (
                                    <button
                                        onClick={() => onUseSuggestion(typeof suggestion === 'string' ? suggestion : suggestion.suggestion || suggestion)}
                                        className="w-full text-left bg-[#1a1a1a] border border-[#2f3335] hover:border-[var(--color-primary)] hover:bg-[var(--color-primary)]/10 rounded-lg p-3 text-sm text-[#e0ddd9] transition-all group"
                                    >
                                        <span className="flex items-start gap-2">
                                            <span className="mt-0.5 w-4 h-4 rounded-full border border-[var(--color-primary)] flex items-center justify-center shrink-0">
                                                <span className="w-2 h-2 rounded-full bg-[var(--color-primary)] opacity-0 group-hover:opacity-100 transition-opacity" />
                                            </span>
                                            <span className="flex-1">
                                                {typeof suggestion === 'string' ? suggestion : suggestion.suggestion || 'Click to use this suggestion'}
                                            </span>
                                        </span>
                                    </button>
                                ) : (
                                    <p className="text-[#958d73] text-sm italic">Analyzing best alternative...</p>
                                )}
                            </div>

                            {/* Actions */}
                            <div className="text-xs text-[#5a5a5a] text-center mb-3">
                                Click the suggestion above to use it, or choose an option below
                            </div>
                            <div className="flex flex-col sm:flex-row gap-3">
                                <button
                                    onClick={onEdit}
                                    className="flex-1 bg-[#2f3335] hover:bg-[#3f4345] text-[#e0ddd9] font-medium py-2.5 px-4 rounded-lg transition-colors"
                                >
                                    Edit Original
                                </button>
                                <button
                                    onClick={onSendAnyway}
                                    className="flex-1 bg-transparent border border-red-500/50 hover:bg-red-500/10 text-red-400 font-medium py-2.5 px-4 rounded-lg transition-colors"
                                >
                                    Send Anyway
                                </button>
                            </div>
                        </div> {/* <--- I ADDED THIS MISSING TAG */}
                    </motion.div>
                </>
            )}
        </AnimatePresence>
    );
}