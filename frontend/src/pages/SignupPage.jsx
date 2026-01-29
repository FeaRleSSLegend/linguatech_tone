import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { Mail, Lock, User, Loader2, ArrowRight } from 'lucide-react';
import AuthLayout from '../layouts/AuthLayout';

export default function SignupPage() {
    const [name, setName] = useState('');
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [adminSecret, setAdminSecret] = useState('');
    const [error, setError] = useState('');
    const [isSubmitting, setIsSubmitting] = useState(false);

    const { signup } = useAuth();
    const navigate = useNavigate();

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        setIsSubmitting(true);

        try {
            const user = await signup(name, email, password, adminSecret);
            if (user.role === 'admin' || user.role === 'super_admin') {
                navigate('/admin/reports');
            } else {
                navigate('/app');
            }
        } catch (err) {
            setError(err.message || 'Failed to create account');
        } finally {
            setIsSubmitting(false);
        }
    };

    return (
        <AuthLayout>
            <div className="absolute top-4 left-4">
                <Link to="/" className="text-[#958d73] hover:text-[#e0ddd9] transition-colors flex items-center gap-2 text-sm">
                    <ArrowRight className="w-4 h-4 rotate-180" /> Back
                </Link>
            </div>
            <div className="text-center mb-6 sm:mb-8 mt-4">
                <h2 className="text-xl sm:text-2xl text-[#e0ddd9] font-light mb-2">Create Account</h2>
                <p className="text-[#958d73] text-xs sm:text-sm">Join Tone to start safe conversations</p>
            </div>

            <form onSubmit={handleSubmit} className="space-y-5">
                {/* Name Input */}
                <div className="space-y-1">
                    <label className="text-xs uppercase tracking-wider text-[#958d73] ml-1">Full Name</label>
                    <div className="relative group">
                        <User className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-[#958d73] group-focus-within:text-[#e0ddd9] transition-colors" />
                        <input
                            type="text"
                            value={name}
                            onChange={(e) => setName(e.target.value)}
                            required
                            className="w-full bg-[#1a1a1a]/50 border border-white/10 rounded-lg py-3 pl-10 pr-4 text-[#e0ddd9] placeholder-white/20 focus:outline-none focus:border-[var(--color-primary)]/50 focus:bg-[#1a1a1a] transition-all"
                            placeholder="John Doe"
                        />
                    </div>
                </div>

                {/* Email Input */}
                <div className="space-y-1">
                    <label className="text-xs uppercase tracking-wider text-[#958d73] ml-1">Email</label>
                    <div className="relative group">
                        <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-[#958d73] group-focus-within:text-[#e0ddd9] transition-colors" />
                        <input
                            type="email"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            required
                            className="w-full bg-[#1a1a1a]/50 border border-white/10 rounded-lg py-3 pl-10 pr-4 text-[#e0ddd9] placeholder-white/20 focus:outline-none focus:border-[var(--color-primary)]/50 focus:bg-[#1a1a1a] transition-all"
                            placeholder="name@example.com"
                        />
                    </div>
                </div>

                {/* Password Input */}
                <div className="space-y-1">
                    <label className="text-xs uppercase tracking-wider text-[#958d73] ml-1">Password</label>
                    <div className="relative group">
                        <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-[#958d73] group-focus-within:text-[#e0ddd9] transition-colors" />
                        <input
                            type="password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            required
                            minLength={6}
                            className="w-full bg-[#1a1a1a]/50 border border-white/10 rounded-lg py-3 pl-10 pr-4 text-[#e0ddd9] placeholder-white/20 focus:outline-none focus:border-[var(--color-primary)] focus:bg-[#1a1a1a] transition-all"
                            placeholder="••••••••"
                        />
                    </div>
                </div>

                {/* Admin Secret Input (Optional) */}
                <div className="space-y-1">
                    <label className="text-xs uppercase tracking-wider text-[#958d73] ml-1">Admin Secret (Optional)</label>
                    <div className="relative group">
                        <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-[#958d73] group-focus-within:text-[#e0ddd9] transition-colors" />
                        <input
                            type="password"
                            value={adminSecret}
                            onChange={(e) => setAdminSecret(e.target.value)}
                            className="w-full bg-[#1a1a1a]/50 border border-white/10 rounded-lg py-3 pl-10 pr-4 text-[#e0ddd9] placeholder-white/20 focus:outline-none focus:border-[var(--color-primary)] focus:bg-[#1a1a1a] transition-all"
                            placeholder="Enter secret to register as admin"
                        />
                    </div>
                </div>

                {/* Error Message */}
                {error && (
                    <div className="text-red-400 text-sm text-center bg-red-500/10 border border-red-500/20 rounded-lg p-2">
                        {error}
                    </div>
                )}

                {/* Submit Button */}
                <button
                    type="submit"
                    disabled={isSubmitting}
                    className="w-full bg-[var(--color-primary)] hover:bg-[var(--color-primary-hover)] text-white font-medium py-2.5 rounded-lg transition-all flex items-center justify-center gap-2 group disabled:opacity-70 disabled:cursor-not-allowed"
                >
                    {isSubmitting ? (
                        <Loader2 className="w-5 h-5 animate-spin" />
                    ) : (
                        <>
                            Sign Up
                            <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                        </>
                    )}
                </button>

                {/* Footer */}
                <p className="text-center text-[#958d73] text-sm mt-6">
                    Already have an account?{' '}
                    <Link to="/login" className="text-[var(--color-primary)] hover:text-[var(--color-primary-hover)] transition-colors">
                        Sign in
                    </Link>
                </p>
            </form>
        </AuthLayout>
    );
}
