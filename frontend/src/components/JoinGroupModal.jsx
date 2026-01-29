import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Users, Loader, ArrowRight } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { joinGroupByInviteCode, getGroupByInviteCode } from '../lib/firebase';

const springTransition = {
    type: "spring",
    stiffness: 100,
    damping: 20
};

export default function JoinGroupModal({ isOpen, onClose, inviteCode: prefilledCode = '' }) {
    const { user } = useAuth();
    const [inviteCode, setInviteCode] = useState(prefilledCode);
    const [isLoading, setIsLoading] = useState(false);
    const [groupPreview, setGroupPreview] = useState(null);
    const [error, setError] = useState('');
    const [success, setSuccess] = useState(false);

    const handlePreview = async () => {
        if (!inviteCode.trim() || inviteCode.trim().length !== 6) {
            setError('Please enter a valid 6-character invite code');
            return;
        }

        setIsLoading(true);
        setError('');
        setGroupPreview(null);

        try {
            const group = await getGroupByInviteCode(inviteCode.trim());
            if (group) {
                setGroupPreview(group);
            } else {
                setError('Invalid invite code');
            }
        } catch (error) {
            console.error('Preview error:', error);
            setError('Failed to load group information');
        } finally {
            setIsLoading(false);
        }
    };

    const handleJoin = async () => {
        if (!user) {
            setError('You must be logged in to join a group');
            return;
        }

        setIsLoading(true);
        setError('');

        try {
            const result = await joinGroupByInviteCode(inviteCode.trim(), user.id);

            if (result.alreadyMember) {
                setError('You are already a member of this group');
            } else {
                setSuccess(true);
                setTimeout(() => {
                    handleClose();
                }, 2000);
            }
        } catch (error) {
            console.error('Join error:', error);
            setError(error.message || 'Failed to join group');
        } finally {
            setIsLoading(false);
        }
    };

    const handleClose = () => {
        setInviteCode('');
        setGroupPreview(null);
        setError('');
        setSuccess(false);
        onClose();
    };

    return (
        <AnimatePresence>
            {isOpen && (
                <>
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        transition={springTransition}
                        onClick={handleClose}
                        className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50"
                    />

                    <motion.div
                        initial={{ opacity: 0, scale: 0.95, y: 20 }}
                        animate={{ opacity: 1, scale: 1, y: 0 }}
                        exit={{ opacity: 0, scale: 0.95, y: 20 }}
                        transition={springTransition}
                        className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-md z-50"
                    >
                        <div className="bg-[#0d0d0d] border border-[#2f3335] rounded-2xl shadow-2xl overflow-hidden">
                            {/* Header */}
                            <div className="p-6 border-b border-[#2f3335] flex items-center justify-between">
                                <h2 className="text-lg font-bold text-white flex items-center gap-2">
                                    <Users className="w-5 h-5 text-[var(--color-primary)]" />
                                    Join Group
                                </h2>
                                <button
                                    onClick={handleClose}
                                    className="text-[#958d73] hover:text-white transition-colors p-2 hover:bg-[#1a1a1a] rounded-lg"
                                >
                                    <X className="w-5 h-5" />
                                </button>
                            </div>

                            {/* Content */}
                            <div className="p-6 space-y-4">
                                {success ? (
                                    <motion.div
                                        initial={{ opacity: 0, scale: 0.9 }}
                                        animate={{ opacity: 1, scale: 1 }}
                                        className="text-center py-8"
                                    >
                                        <div className="w-16 h-16 rounded-full bg-[var(--color-primary)] mx-auto flex items-center justify-center mb-4">
                                            <Users className="w-8 h-8 text-black" />
                                        </div>
                                        <h3 className="text-xl font-bold text-white mb-2">Welcome!</h3>
                                        <p className="text-sm text-[#958d73]">
                                            You've successfully joined the group
                                        </p>
                                    </motion.div>
                                ) : (
                                    <>
                                        {/* Invite Code Input */}
                                        <div>
                                            <label className="block text-sm font-medium text-[#e0ddd9] mb-2">
                                                Enter Invite Code
                                            </label>
                                            <div className="flex gap-2">
                                                <input
                                                    type="text"
                                                    placeholder="ABC123"
                                                    value={inviteCode}
                                                    onChange={(e) => setInviteCode(e.target.value.toUpperCase())}
                                                    maxLength={6}
                                                    className="flex-1 bg-[#1a1a1a] border border-[#2f3335] focus:border-[var(--color-primary)] rounded-lg px-4 py-3 text-center text-lg font-mono font-bold text-white placeholder-[#5a5a5a] focus:outline-none transition-colors tracking-wider"
                                                />
                                                {!groupPreview && (
                                                    <button
                                                        onClick={handlePreview}
                                                        disabled={isLoading || inviteCode.trim().length !== 6}
                                                        className={`px-4 py-3 rounded-lg font-medium transition-all ${
                                                            inviteCode.trim().length !== 6
                                                                ? 'bg-[#2f3335] text-[#5a5a5a] cursor-not-allowed'
                                                                : 'bg-[var(--color-primary)] text-black hover:bg-[#2d9248]'
                                                        }`}
                                                    >
                                                        {isLoading ? (
                                                            <Loader className="w-5 h-5 animate-spin" />
                                                        ) : (
                                                            <ArrowRight className="w-5 h-5" />
                                                        )}
                                                    </button>
                                                )}
                                            </div>
                                            <p className="text-xs text-[#958d73] mt-2">
                                                Enter the 6-character code shared by the group admin
                                            </p>
                                        </div>

                                        {/* Error Message */}
                                        {error && (
                                            <motion.div
                                                initial={{ opacity: 0, y: -10 }}
                                                animate={{ opacity: 1, y: 0 }}
                                                className="bg-red-500/10 border border-red-500/30 rounded-lg p-3 text-sm text-red-400"
                                            >
                                                {error}
                                            </motion.div>
                                        )}

                                        {/* Group Preview */}
                                        {groupPreview && (
                                            <motion.div
                                                initial={{ opacity: 0, y: 10 }}
                                                animate={{ opacity: 1, y: 0 }}
                                                className="bg-[#1a1a1a] border border-[#2f3335] rounded-xl p-4 space-y-3"
                                            >
                                                <div className="flex items-center gap-3">
                                                    <div className="w-12 h-12 rounded-full bg-[#2f3335] flex items-center justify-center overflow-hidden">
                                                        {groupPreview.profilePictureUrl ? (
                                                            <img
                                                                src={groupPreview.profilePictureUrl}
                                                                alt={groupPreview.name}
                                                                className="w-full h-full object-cover"
                                                            />
                                                        ) : (
                                                            <Users className="w-6 h-6 text-[#958d73]" />
                                                        )}
                                                    </div>
                                                    <div className="flex-1">
                                                        <h3 className="text-white font-medium">
                                                            {groupPreview.name}
                                                        </h3>
                                                        <p className="text-xs text-[#958d73]">
                                                            {groupPreview.memberCount} / {groupPreview.maxMembers} members
                                                        </p>
                                                    </div>
                                                </div>

                                                {groupPreview.description && (
                                                    <p className="text-sm text-[#958d73]">
                                                        {groupPreview.description}
                                                    </p>
                                                )}

                                                <button
                                                    onClick={handleJoin}
                                                    disabled={isLoading}
                                                    className="w-full py-3 bg-[var(--color-primary)] hover:bg-[#2d9248] text-black font-medium rounded-lg transition-all flex items-center justify-center gap-2"
                                                >
                                                    {isLoading ? (
                                                        <>
                                                            <Loader className="w-4 h-4 animate-spin" />
                                                            Joining...
                                                        </>
                                                    ) : (
                                                        <>
                                                            <Users className="w-4 h-4" />
                                                            Join Group
                                                        </>
                                                    )}
                                                </button>
                                            </motion.div>
                                        )}
                                    </>
                                )}
                            </div>
                        </div>
                    </motion.div>
                </>
            )}
        </AnimatePresence>
    );
}
