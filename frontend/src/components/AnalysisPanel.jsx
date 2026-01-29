import { motion } from 'framer-motion';

/**
 * AnalysisPanel Component
 * Displays real-time sentiment and toxicity analysis
 * 
 * @param {Object} props
 * @param {Object} props.analysis - The analysis data from API
 * @param {boolean} props.isAnalyzing - Loading state
 */
export default function AnalysisPanel({ analysis, isAnalyzing = false }) {
    const {
        sentiment = { label: 'neutral', confidence: 0 },
        toxicity = { label: 'safe', confidence: 0 },
        feedback = ''
    } = analysis || {};



    // Toxicity level colors
    const toxicityColors = {
        safe: { bg: 'bg-[var(--color-safe)]', text: 'text-[var(--color-safe)]' },
        warning: { bg: 'bg-yellow-500', text: 'text-yellow-500' },
        toxic: { bg: 'bg-red-500', text: 'text-red-500' }
    };

    const toxicityLevel = toxicity.confidence > 0.7 ? 'toxic' : toxicity.confidence > 0.4 ? 'warning' : 'safe';
    const toxicityColor = toxicityColors[toxicityLevel];

    return (
        <div className="bg-[#0d0d0d] border border-[#2f3335] rounded-2xl p-5 h-full">
            {/* Header */}
            <h2 className="text-[#e0ddd9] text-xl font-semibold mb-6 flex items-center gap-2 text-glow-neon">
                Live Tone Analysis
            </h2>

            {/* Analyzing Indicator */}
            {isAnalyzing && (
                <div className="flex items-center gap-2 mb-4 text-[#958d73]">
                    <motion.div
                        animate={{ rotate: 360 }}
                        transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                        className="w-4 h-4 border-2 border-[#958d73] border-t-transparent rounded-full"
                    />
                    <span className="text-sm">Analyzing...</span>
                </div>
            )}

            {/* Sentiment Section */}
            <div className="mb-6">
                <h3 className="text-[#958d73] text-sm uppercase tracking-wider mb-3">Sentiment</h3>
                <div className="flex items-center gap-3">
                    <div>
                        <p className="text-[#e0ddd9] capitalize font-medium">{sentiment.label}</p>
                        <p className="text-[#958d73] text-sm">
                            {Math.round(sentiment.confidence * 100)}% confidence
                        </p>
                    </div>
                </div>
            </div>

            {/* Toxicity Meter */}
            <div className="mb-6">
                <h3 className="text-[#958d73] text-sm uppercase tracking-wider mb-3">Tone Resonance</h3>

                {/* Meter Bar */}
                <div className="relative h-3 bg-[#1a1a1a] rounded-full overflow-hidden mb-2">
                    <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${toxicity.confidence * 100}%` }}
                        transition={{ duration: 0.5, ease: 'easeOut' }}
                        className={`absolute top-0 left-0 h-full ${toxicityColor.bg} rounded-full`}
                    />
                </div>

                {/* Labels */}
                <div className="flex justify-between text-xs">
                    <span className="text-[var(--color-safe)]">Safe</span>
                    <span className="text-yellow-500">Warning</span>
                    <span className="text-red-500">Toxic</span>
                </div>

                {/* Confidence */}
                <p className={`mt-2 ${toxicityColor.text} font-medium`}>
                    {Math.round(toxicity.confidence * 100)}% emotional resonance
                </p>
            </div>

            {/* Feedback */}
            {feedback && (
                <div className="bg-[#1a1a1a] border border-[#2f3335] rounded-xl p-4">
                    <h3 className="text-[#958d73] text-sm uppercase tracking-wider mb-2">Feedback</h3>
                    <p className="text-[#e0ddd9] text-sm leading-relaxed">{feedback}</p>
                </div>
            )}

            {/* No Input State */}
            {!analysis && !isAnalyzing && (
                <div className="text-center py-8">
                    <p className="text-[#958d73]">Start typing to see real-time analysis</p>
                </div>
            )}
        </div>
    );
}
