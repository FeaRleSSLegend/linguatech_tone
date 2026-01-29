import { Link } from 'react-router-dom';
import { ArrowLeft, Scale } from 'lucide-react';
import { motion } from 'framer-motion';
import Logo from '../components/Logo';

export default function TermsOfService() {
    return (
        <div className="min-h-screen bg-[#080808] text-[#e0ddd9] font-sans selection:bg-[#00ff88]/30">
            {/* Header */}
            <header className="fixed top-0 left-0 right-0 z-50 bg-[#080808]/80 backdrop-blur-lg border-b border-white/5">
                <div className="max-w-4xl mx-auto px-6 h-20 flex items-center justify-between">
                    <Link to="/" className="flex items-center gap-2 group">
                        <Logo className="w-8 h-8" />
                    </Link>
                    <Link to="/" className="text-sm font-medium text-[#958d73] hover:text-white transition-colors flex items-center gap-2">
                        <ArrowLeft className="w-4 h-4" /> Back to Home
                    </Link>
                </div>
            </header>

            <main className="pt-32 pb-20 px-6 max-w-4xl mx-auto">
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6 }}
                >
                    <div className="mb-6">
                        <h1 className="text-4xl font-bold tracking-tight">Terms of Service</h1>
                    </div>

                    <p className="text-[#958d73] mb-12">Last Updated: January 25, 2026</p>

                    <div className="space-y-12 text-[#e0ddd9] leading-relaxed">
                        <section>
                            <h2 className="text-2xl font-bold mb-4 text-white">1. Acceptance of Terms</h2>
                            <p className="text-[#958d73]">
                                By accessing or using Tone, you agree to be bound by these Terms of Service. If you do not agree to
                                these terms, please do not use our service.
                            </p>
                        </section>

                        <section>
                            <h2 className="text-2xl font-bold mb-4 text-white">2. Description of Service</h2>
                            <p className="text-[#958d73]">
                                Tone provides AI-powered analysis of message tone and resonance. The service is intended to help users
                                understand and refine their communication styles.
                            </p>
                        </section>

                        <section>
                            <h2 className="text-2xl font-bold mb-4 text-white">3. User Responsibilities</h2>
                            <p className="text-[#958d73] mb-4">
                                You are responsible for:
                            </p>
                            <ul className="list-disc pl-6 space-y-2 text-[#958d73]">
                                <li>Maintaining the confidentiality of your account.</li>
                                <li>All activities that occur under your account.</li>
                                <li>Ensuring your use of the service complies with applicable laws.</li>
                                <li>Treating others with respect within the platform.</li>
                            </ul>
                        </section>

                        <section>
                            <h2 className="text-2xl font-bold mb-4 text-white">4. Intellectual Property</h2>
                            <p className="text-[#958d73]">
                                All content, features, and functionality of Tone are the exclusive property of Tone and its licensors.
                                You are granted a limited, non-exclusive license to use the service for personal or professional communication refining.
                            </p>
                        </section>

                        <section>
                            <h2 className="text-2xl font-bold mb-4 text-white">5. Limitation of Liability</h2>
                            <p className="text-[#958d73]">
                                Tone provides analysis based on AI models. We do not guarantee 100% accuracy and are not liable for
                                any interpretations or actions taken based on the resonance insights provided.
                            </p>
                        </section>
                    </div>
                </motion.div>
            </main>

            <footer className="py-12 px-6 border-t border-white/5 bg-[#050505] text-center">
                <Logo className="w-8 h-8 mx-auto mb-4" />
                <p className="text-[#5a5a5a] text-sm">© 2026 Tone. Perfecting the frequency of communication.</p>
            </footer>
        </div>
    );
}
