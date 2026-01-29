import { Link, useNavigate } from 'react-router-dom';
import { ArrowRight, Check, Shield, MessageSquare, Zap, FileText, Menu, X } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import Logo from '../components/Logo';

export default function LandingPage() {
    const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
    const { user, loading } = useAuth();
    const navigate = useNavigate();

    useEffect(() => {
        if (!loading && user) {
            if (user.role === 'admin' || user.role === 'super_admin') {
                navigate('/admin/reports');
            } else {
                navigate('/app');
            }
        }
    }, [user, loading, navigate]);

    return (
        <div className="min-h-screen bg-[#050505] text-[#e0ddd9] font-sans selection:bg-[#34a853]/30">
            {/* Header */}
            <header className="fixed top-0 left-0 right-0 z-50 bg-[#050505]/80 backdrop-blur-md border-b border-white/5">
                <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <Logo className="w-8 h-8" />
                    </div>

                    {/* Desktop Nav */}
                    <nav className="hidden md:flex items-center gap-10">
                        <Link to="/login" className="text-sm font-medium text-[#e0ddd9] hover:text-white transition-colors">Sign In</Link>
                        <Link
                            to="/signup"
                            className="bg-[#34a853] hover:bg-[#2d9248] text-white text-sm font-bold px-6 py-2.5 rounded-lg transition-all shadow-lg shadow-emerald-900/20"
                        >
                            Get Started
                        </Link>
                    </nav>

                    {/* Mobile Menu Toggle */}
                    <button className="md:hidden text-[#e0ddd9]" onClick={() => setMobileMenuOpen(!mobileMenuOpen)}>
                        {mobileMenuOpen ? <X /> : <Menu />}
                    </button>
                </div>

                {/* Mobile Nav */}
                <AnimatePresence>
                    {mobileMenuOpen && (
                        <motion.div
                            initial={{ opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: 'auto' }}
                            exit={{ opacity: 0, height: 0 }}
                            className="md:hidden bg-[#0d0d0d] border-b border-white/5 overflow-hidden"
                        >
                            <div className="p-6 space-y-4">
                                <Link to="/login" className="block text-center py-3 text-[#e0ddd9] font-medium border border-white/5 rounded-xl">Sign In</Link>
                                <Link to="/signup" className="block text-center bg-[#34a853] text-white py-3 rounded-xl font-bold">Get Started</Link>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>
            </header>

            {/* Hero Section */}
            <section className="relative pt-32 pb-20 md:pt-48 md:pb-32 px-6 overflow-hidden">
                <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full max-w-4xl h-full bg-emerald-500/5 blur-[120px] rounded-full pointer-events-none" />

                <div className="max-w-7xl mx-auto text-center relative z-10">
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
                    >
                        <h1 className="text-4xl md:text-6xl font-black tracking-tight mb-6 leading-[1.15]">
                            Set the Right Tone <br />
                            <span className="text-[#34a853]">Every Time</span>
                        </h1>
                        <p className="text-[#958d73] text-base md:text-lg max-w-xl mx-auto mb-10 leading-relaxed">
                            AI-powered chat that detects toxic content in real-time, suggests kinder alternatives,
                            and creates a safer space for students to communicate.
                        </p>
                        <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
                            <Link
                                to="/signup"
                                className="w-full sm:w-auto bg-[#34a853] hover:bg-[#2d9248] text-white font-bold px-8 py-3 rounded-lg transition-all flex items-center justify-center gap-2 group shadow-xl shadow-emerald-900/20"
                            >
                                Create Account
                                <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                            </Link>
                            <Link
                                to="/login"
                                className="w-full sm:w-auto bg-transparent border border-[#2f3335] hover:bg-white/5 text-white font-bold px-8 py-3 rounded-lg transition-all"
                            >
                                Sign In
                            </Link>
                        </div>
                    </motion.div>
                </div>
            </section>

            {/* Key Features */}
            <section className="py-24 bg-[#050505]">
                <div className="max-w-7xl mx-auto px-6">
                    <div className="text-center mb-16">
                        <h2 className="text-3xl md:text-4xl font-black mb-4">Key Features</h2>
                        <p className="text-[#958d73] text-base max-w-xl mx-auto">
                            Everything you need to create a safer communication environment for your school.
                        </p>
                    </div>

                    <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6">
                        <FeatureCard
                            icon={<Zap className="w-6 h-6" />}
                            title="Real-time Toxicity Detection"
                            desc="AI-powered analysis flags harmful content before it's sent, protecting students and creating safer conversations."
                        />
                        <FeatureCard
                            icon={<MessageSquare className="w-6 h-6" />}
                            title="Smart Rephrasing Suggestions"
                            desc="Get instant suggestions to rephrase messages more kindly while keeping your original meaning intact."
                        />
                        <FeatureCard
                            icon={<Zap className="w-6 h-6" />} // Using Zap as placeholder for 1-on-1 icon
                            title="1-on-1 & Group Chats"
                            desc="Connect with classmates through private messages or create group chats for study groups and projects."
                        />
                        <FeatureCard
                            icon={<Shield className="w-6 h-6" />}
                            title="Moderation Logging"
                            desc="All flagged messages are logged for review, helping moderators maintain a positive environment."
                        />
                    </div>
                </div>
            </section>

            {/* Why Schools Choose Tone */}
            <section className="py-24 bg-[#050505]">
                <div className="max-w-7xl mx-auto px-6">
                    <div className="grid lg:grid-cols-2 gap-20 items-center">
                        <div>
                            <h2 className="text-3xl md:text-5xl font-black mb-6 leading-tight">Why Schools Choose Tone</h2>
                            <p className="text-[#958d73] text-base mb-8 leading-relaxed">
                                Tone is designed specifically for educational environments, helping students learn to communicate respectfully while keeping everyone safe.
                            </p>

                            <div className="space-y-6">
                                {[
                                    "Reduce cyberbullying incidents by up to 70%",
                                    "Create a positive learning environment",
                                    "Teach students to communicate kindly",
                                    "Real-time intervention before harm occurs",
                                    "Easy integration with school systems",
                                    "FERPA and COPPA compliant"
                                ].map((item, i) => (
                                    <div key={i} className="flex items-center gap-4 group">
                                        <div className="w-5 h-5 rounded-full border border-[#34a853] flex items-center justify-center text-[#34a853] shrink-0 group-hover:bg-[#34a853] group-hover:text-white transition-all">
                                            <Check className="w-3 h-3" />
                                        </div>
                                        <span className="text-[#e0ddd9] text-base font-medium">{item}</span>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Interactive Demo Visual */}
                        <div className="bg-[#0d0d0d] border border-white/5 rounded-[2.5rem] p-10 md:p-12 shadow-3xl relative">
                            <div className="space-y-8">
                                {/* Bad Message */}
                                <div className="bg-[#1a1a1a] rounded-[1.5rem] p-6 relative border border-white/5">
                                    <p className="text-[#e0ddd9] mb-4 font-medium italic">"You're so stupid, can't believe you failed again!"</p>
                                    <div className="flex items-center gap-2">
                                        <span className="w-2 h-2 rounded-full bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.6)]" />
                                        <span className="text-red-500 text-xs font-bold uppercase tracking-widest">Toxic detected</span>
                                    </div>
                                </div>

                                <div className="flex justify-center">
                                    <div className="bg-[#34a853]/10 p-3 rounded-full border border-[#34a853]/20">
                                        <ArrowRight className="w-6 h-6 text-[#34a853]" />
                                    </div>
                                </div>

                                {/* Good Message */}
                                <div className="bg-[#1a1a1a] border border-[#34a853]/30 rounded-[1.5rem] p-6 relative">
                                    <p className="text-[#e0ddd9] mb-4 font-medium italic">"I know that was tough, but don't give up. Want to study together next time?"</p>
                                    <div className="flex items-center gap-2">
                                        <span className="w-2 h-2 rounded-full bg-[#34a853] shadow-[0_0_8px_rgba(52,168,83,0.6)]" />
                                        <span className="text-[#34a853] text-xs font-bold uppercase tracking-widest">Safe & supportive</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            {/* CTA */}
            <section className="py-32 bg-[#050505] relative overflow-hidden">
                <div className="absolute inset-0 bg-emerald-500/5 pointer-events-none" />
                <div className="max-w-4xl mx-auto text-center px-6 relative z-10">
                    <h2 className="text-4xl md:text-6xl font-black mb-8">Ready to Create a <br /> Safer Space?</h2>
                    <p className="text-[#958d73] text-lg mb-10">
                        Join Tone today and help build a more positive communication culture in your school.
                    </p>
                    <Link
                        to="/signup"
                        className="w-full sm:w-auto inline-flex items-center justify-center gap-2 bg-[#34a853] hover:bg-[#2d9248] text-white font-bold px-10 py-4 rounded-xl transition-all shadow-xl shadow-emerald-900/30"
                    >
                        Get Started Free
                        <ArrowRight className="w-5 h-5" />
                    </Link>
                </div>
            </section>

            {/* Footer */}
            <footer className="py-20 px-6 border-t border-white/5 bg-[#050505]">
                <div className="max-w-7xl mx-auto flex flex-col items-center">
                    <div className="flex items-center gap-2 mb-10">
                        <Logo className="w-10 h-10" />
                    </div>

                    <div className="flex justify-center gap-12 mb-12 text-[#958d73] text-sm font-bold uppercase tracking-[0.2em]">
                        <Link to="/privacy" className="hover:text-white transition-colors">Privacy Policy</Link>
                        <Link to="/terms" className="hover:text-white transition-colors">Terms of Service</Link>
                    </div>

                    <p className="text-[#5a5a5a] text-sm font-medium">© 2026 Tone. Setting the right tone in every conversation.</p>
                </div>
            </footer>
        </div>
    );
}

function FeatureCard({ icon, title, desc }) {
    return (
        <div className="bg-[#0d0d0d] border border-white/5 p-8 rounded-[1.5rem] hover:border-[#34a853]/30 transition-all duration-500 group">
            <div className="w-12 h-12 rounded-xl bg-[#34a853]/10 flex items-center justify-center mb-6 text-[#34a853] group-hover:scale-110 transition-transform duration-500 border border-[#34a853]/10">
                {icon}
            </div>
            <h3 className="text-lg font-black mb-3 group-hover:text-white transition-colors">{title}</h3>
            <p className="text-[#958d73] text-sm leading-relaxed font-medium">{desc}</p>
        </div>
    );
}
