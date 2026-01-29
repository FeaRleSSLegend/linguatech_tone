import { motion } from 'framer-motion';
import Logo from '../components/Logo';

export default function AuthLayout({ children }) {
    return (
        <div className="min-h-screen w-full bg-[#080808] relative overflow-hidden flex items-center justify-center p-4 sm:p-6 md:p-8">
            {/* Abstract Background Gradient Mesh */}
            <div className="absolute top-[-20%] left-[-10%] w-[100%] sm:w-[50%] h-[100%] sm:h-[50%] bg-[#00ff88]/10 blur-[120px] rounded-full pointer-events-none" />
            <div className="absolute bottom-[-20%] right-[-10%] w-[100%] sm:w-[50%] h-[100%] sm:h-[50%] bg-emerald-900/10 blur-[100px] rounded-full pointer-events-none" />

            {/* Glass Card */}
            <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5, ease: 'easeOut' }}
                className="relative z-10 w-full max-w-md bg-[#ffffff]/5 backdrop-blur-xl border border-white/10 shadow-2xl rounded-2xl p-6 sm:p-8"
            >
                <div className="flex justify-center mb-6 sm:mb-8">
                    <Logo className="w-10 h-10 sm:w-12 sm:h-12" textClassName="text-2xl sm:text-3xl font-bold tracking-tight" />
                </div>

                {children}
            </motion.div>
        </div>
    );
}
