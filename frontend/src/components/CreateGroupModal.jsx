import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Users, Image as ImageIcon, Loader, Copy, Check } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { createGroupChat, uploadGroupPicture } from '../lib/firebase';

const springTransition = {
    type: "spring",
    stiffness: 100,
    damping: 20
};

export default function CreateGroupModal({ isOpen, onClose }) {
    const { user } = useAuth();
    const [name, setName] = useState('');
    const [description, setDescription] = useState('');
    const [adminOnlyMessaging, setAdminOnlyMessaging] = useState(false);
    const [profilePicture, setProfilePicture] = useState(null);
    const [profilePicturePreview, setProfilePicturePreview] = useState('');
    const [isCreating, setIsCreating] = useState(false);
    const [createdGroup, setCreatedGroup] = useState(null);
    const [copiedCode, setCopiedCode] = useState(false);

    const handleProfilePictureChange = (e) => {
        const file = e.target.files?.[0];
        if (file) {
            setProfilePicture(file);
            const reader = new FileReader();
            reader.onloadend = () => {
                setProfilePicturePreview(reader.result);
            };
            reader.readAsDataURL(file);
        }
    };

    const handleCreateGroup = async () => {
        if (!name.trim()) {
            alert('Please enter a group name');
            return;
        }

        if (!user) {
            alert('You must be logged in to create a group');
            return;
        }

        setIsCreating(true);

        try {
            let profilePictureUrl = '';

            // Try to upload profile picture first if provided
            if (profilePicture) {
                try {
                    // We'll upload after group creation if needed
                    // For now, we'll create group with placeholder
                    console.log('📸 [CreateGroupModal] Profile picture selected, will use placeholder initially');
                } catch (uploadError) {
                    console.warn('⚠️ [CreateGroupModal] Profile picture prep failed:', uploadError);
                }
            }

            // Create group with placeholder avatar (will be generated in createGroupChat)
            const groupData = {
                name: name.trim(),
                description: description.trim(),
                adminOnlyMessaging,
                profilePictureUrl: '' // Empty = will use ui-avatars.com placeholder
            };

            const group = await createGroupChat(user.id, groupData);

            setCreatedGroup({
                ...group
            });

            console.log('✅ [CreateGroupModal] Group created successfully:', group);
        } catch (error) {
            console.error('❌ [CreateGroupModal] Creation error:', error);
            alert(`Failed to create group: ${error.message}`);
        } finally {
            setIsCreating(false);
        }
    };

    const handleCopyInviteCode = () => {
        if (createdGroup?.inviteCode) {
            navigator.clipboard.writeText(createdGroup.inviteCode);
            setCopiedCode(true);
            setTimeout(() => setCopiedCode(false), 2000);
        }
    };

    const handleClose = () => {
        setName('');
        setDescription('');
        setAdminOnlyMessaging(false);
        setProfilePicture(null);
        setProfilePicturePreview('');
        setCreatedGroup(null);
        setCopiedCode(false);
        onClose();
    };

    if (!user) return null;

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
                                    {createdGroup ? 'Group Created!' : 'Create Group'}
                                </h2>
                                <button
                                    onClick={handleClose}
                                    className="text-[#958d73] hover:text-white transition-colors p-2 hover:bg-[#1a1a1a] rounded-lg"
                                >
                                    <X className="w-5 h-5" />
                                </button>
                            </div>

                            {/* Content */}
                            <div className="p-6 max-h-[70vh] overflow-y-auto">
                                {createdGroup ? (
                                    /* Success View */
                                    <motion.div
                                        initial={{ opacity: 0, y: 20 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        transition={springTransition}
                                        className="space-y-6 text-center"
                                    >
                                        <div className="w-20 h-20 rounded-full bg-[var(--color-primary)] mx-auto flex items-center justify-center">
                                            <Check className="w-10 h-10 text-black" />
                                        </div>

                                        <div>
                                            <h3 className="text-xl font-bold text-white mb-2">
                                                {createdGroup.name}
                                            </h3>
                                            <p className="text-sm text-[#958d73]">
                                                Your group has been created successfully!
                                            </p>
                                        </div>

                                        <div className="bg-[#1a1a1a] border border-[#2f3335] rounded-xl p-4">
                                            <p className="text-xs text-[#958d73] mb-2">Invite Code</p>
                                            <div className="flex items-center justify-center gap-2">
                                                <span className="text-2xl font-mono font-bold text-[var(--color-primary)] tracking-wider">
                                                    {createdGroup.inviteCode}
                                                </span>
                                                <button
                                                    onClick={handleCopyInviteCode}
                                                    className="p-2 hover:bg-[#2f3335] rounded-lg transition-colors"
                                                    title="Copy invite code"
                                                >
                                                    {copiedCode ? (
                                                        <Check className="w-4 h-4 text-[var(--color-primary)]" />
                                                    ) : (
                                                        <Copy className="w-4 h-4 text-[#958d73]" />
                                                    )}
                                                </button>
                                            </div>
                                            <p className="text-xs text-[#958d73] mt-3">
                                                Share this code with others to let them join your group
                                            </p>
                                        </div>

                                        <div className="bg-[#1a1a1a] border border-[#2f3335] rounded-xl p-4">
                                            <p className="text-xs text-[#958d73] mb-2">QR Code (Coming Soon)</p>
                                            <div className="w-32 h-32 mx-auto bg-[#2f3335] rounded-lg flex items-center justify-center">
                                                <p className="text-xs text-[#958d73]">QR Preview</p>
                                            </div>
                                        </div>

                                        <button
                                            onClick={handleClose}
                                            className="w-full py-3 bg-[var(--color-primary)] hover:bg-[#2d9248] text-black font-medium rounded-lg transition-all"
                                        >
                                            Done
                                        </button>
                                    </motion.div>
                                ) : (
                                    /* Create Form */
                                    <div className="space-y-4">
                                        {/* Profile Picture */}
                                        <div>
                                            <label className="block text-sm font-medium text-[#e0ddd9] mb-2">
                                                Group Picture (Optional)
                                            </label>
                                            <div className="flex items-center gap-4">
                                                <div className="w-20 h-20 rounded-full bg-[#2f3335] flex items-center justify-center overflow-hidden">
                                                    {profilePicturePreview ? (
                                                        <img
                                                            src={profilePicturePreview}
                                                            alt="Preview"
                                                            className="w-full h-full object-cover"
                                                        />
                                                    ) : (
                                                        <Users className="w-8 h-8 text-[#958d73]" />
                                                    )}
                                                </div>
                                                <label className="cursor-pointer px-4 py-2 bg-[#1a1a1a] border border-[#2f3335] hover:bg-[#2f3335] text-[#e0ddd9] rounded-lg transition-all text-sm flex items-center gap-2">
                                                    <ImageIcon className="w-4 h-4" />
                                                    Choose Image
                                                    <input
                                                        type="file"
                                                        accept="image/*"
                                                        onChange={handleProfilePictureChange}
                                                        className="hidden"
                                                    />
                                                </label>
                                            </div>
                                        </div>

                                        {/* Group Name */}
                                        <div>
                                            <label className="block text-sm font-medium text-[#e0ddd9] mb-2">
                                                Group Name *
                                            </label>
                                            <input
                                                type="text"
                                                placeholder="My Awesome Group"
                                                value={name}
                                                onChange={(e) => setName(e.target.value)}
                                                maxLength={50}
                                                className="w-full bg-[#1a1a1a] border border-[#2f3335] focus:border-[var(--color-primary)] rounded-lg px-4 py-3 text-sm text-white placeholder-[#5a5a5a] focus:outline-none transition-colors"
                                            />
                                            <p className="text-xs text-[#958d73] mt-1">
                                                {name.length}/50 characters
                                            </p>
                                        </div>

                                        {/* Description */}
                                        <div>
                                            <label className="block text-sm font-medium text-[#e0ddd9] mb-2">
                                                Description (Optional)
                                            </label>
                                            <textarea
                                                placeholder="What's your group about?"
                                                value={description}
                                                onChange={(e) => setDescription(e.target.value)}
                                                maxLength={200}
                                                rows={3}
                                                className="w-full bg-[#1a1a1a] border border-[#2f3335] focus:border-[var(--color-primary)] rounded-lg px-4 py-3 text-sm text-white placeholder-[#5a5a5a] focus:outline-none transition-colors resize-none"
                                            />
                                            <p className="text-xs text-[#958d73] mt-1">
                                                {description.length}/200 characters
                                            </p>
                                        </div>

                                        {/* Admin Only Messaging */}
                                        <div className="flex items-center justify-between p-4 bg-[#1a1a1a] border border-[#2f3335] rounded-lg">
                                            <div>
                                                <p className="text-sm font-medium text-white">Admin Only Messaging</p>
                                                <p className="text-xs text-[#958d73] mt-1">
                                                    Only admins can send messages
                                                </p>
                                            </div>
                                            <button
                                                onClick={() => setAdminOnlyMessaging(!adminOnlyMessaging)}
                                                className={`relative w-12 h-6 rounded-full transition-colors ${
                                                    adminOnlyMessaging ? 'bg-[var(--color-primary)]' : 'bg-[#2f3335]'
                                                }`}
                                            >
                                                <motion.div
                                                    animate={{ x: adminOnlyMessaging ? 24 : 2 }}
                                                    transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                                                    className="absolute top-1 w-4 h-4 bg-white rounded-full"
                                                />
                                            </button>
                                        </div>

                                        {/* Info */}
                                        <div className="bg-[var(--color-primary)]/10 border border-[var(--color-primary)]/30 rounded-lg p-3">
                                            <p className="text-xs text-[var(--color-primary)]">
                                                Maximum 500 members per group
                                            </p>
                                        </div>

                                        {/* Create Button */}
                                        <button
                                            onClick={handleCreateGroup}
                                            disabled={isCreating || !name.trim()}
                                            className={`w-full py-3 rounded-lg font-medium transition-all flex items-center justify-center gap-2 ${
                                                !name.trim()
                                                    ? 'bg-[#2f3335] text-[#5a5a5a] cursor-not-allowed'
                                                    : 'bg-[var(--color-primary)] text-black hover:bg-[#2d9248]'
                                            }`}
                                        >
                                            {isCreating ? (
                                                <>
                                                    <Loader className="w-4 h-4 animate-spin" />
                                                    Creating...
                                                </>
                                            ) : (
                                                <>
                                                    <Users className="w-4 h-4" />
                                                    Create Group
                                                </>
                                            )}
                                        </button>
                                    </div>
                                )}
                            </div>
                        </div>
                    </motion.div>
                </>
            )}
        </AnimatePresence>
    );
}
