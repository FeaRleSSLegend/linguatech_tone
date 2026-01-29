import { Link } from 'react-router-dom';
import { ArrowLeft, Shield } from 'lucide-react';
import { motion } from 'framer-motion';
import Logo from '../components/Logo';

export default function PrivacyPolicy() {
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
                        <h1 className="text-4xl font-bold tracking-tight">Privacy Policy</h1>
                    </div>

                    <p className="text-[#958d73] mb-12">Last Updated: January 25, 2026</p>

                    <div className="space-y-12 text-[#e0ddd9] leading-relaxed">
                        <section>
                            <h2 className="text-2xl font-bold mb-4 text-white">1. Introduction</h2>
                            <p className="text-[#958d73]">
                                Welcome to Tone. We are committed to protecting your personal information and your right to privacy.
                                This Privacy Policy explains how we collect, use, and safeguard your data when you use our message tone analysis service.
                            </p>
                        </section>

                        <section>
                            <h2 className="text-2xl font-bold mb-4 text-white">2. Data We Collect</h2>
                            <div className="space-y-4 text-[#958d73]">
                                <p>We collect information that you provide directly to us:</p>
                                <ul className="list-disc pl-6 space-y-2">
                                    <li>Account Information: Name, email address, and password.</li>
                                    <li>Content Data: The messages you type into our interface for tone analysis.</li>
                                    <li>Usage Data: Information about how you interact with our application.</li>
                                </ul>
                            </div>
                        </section>

                        <section>
                            <h2 className="text-2xl font-bold mb-4 text-white">3. How We Use Your Data</h2>
                            <p className="text-[#958d73] mb-4">
                                Our primary goal is to provide real-time tone resonance insights. We use your data to:
                            </p>
                            <ul className="list-disc pl-6 space-y-2 text-[#958d73]">
                                <li>Analyze the emotional tone of your messages.</li>
                                <li>Provide suggestions for refining your communication.</li>
                                <li>Maintain and improve our AI models (anonymized data only).</li>
                                <li>Secure your account and prevent unauthorized access.</li>
                            </ul>
                        </section>

                        <section>
                            <h2 className="text-2xl font-bold mb-4 text-white">4. Data Security</h2>
                            <p className="text-[#958d73]">
                                We implement industry-standard security measures to protect your data. Your messages are transmitted securely
                                and are processed in real-time. We do not sell your personal information to third parties.
                            </p> section.
                        </section>

                        <section>
                            <h2 className="text-2xl font-bold mb-4 text-white">5. Your Rights</h2>
                            <p className="text-[#958d73]">
                                You have the right to access, correct, or delete your personal data at any time. You can manage your account
                                settings within the Tone application or contact our support team.
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
