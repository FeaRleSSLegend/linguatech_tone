import { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { User, X, Camera, Save, Mail, Upload, Loader } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { updateUserProfile, uploadProfilePicture } from '../lib/firebase';

const springTransition = {
    type: "spring",
    stiffness: 100,
    damping: 20
};

export default function ProfileView({ isOpen, onClose }) {
    const { user, updateUser } = useAuth();
    const [isEditing, setIsEditing] = useState(false);
    const [displayName, setDisplayName] = useState(user?.name || '');
    const [profilePictureUrl, setProfilePictureUrl] = useState(user?.profilePictureUrl || '');
    const [isSaving, setIsSaving] = useState(false);
    const [isUploading, setIsUploading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState('');
    const fileInputRef = useRef(null);

    const handleSave = async () => {
        if (!user) return;

        setIsSaving(true);
        try {
            await updateUserProfile(user.id, {
                name: displayName,
                profilePictureUrl
            });

            // Update local auth context
            if (updateUser) {
                updateUser({
                    ...user,
                    name: displayName,
                    profilePictureUrl
                });
            }

            // Update localStorage
            const storedUser = JSON.parse(localStorage.getItem('tone_user') || '{}');
            localStorage.setItem('tone_user', JSON.stringify({
                ...storedUser,
                name: displayName,
                profilePictureUrl
            }));

            setIsEditing(false);
        } catch (error) {
            console.error('Failed to update profile:', error);
            alert('Failed to update profile. Please try again.');
        } finally {
            setIsSaving(false);
        }
    };

    const handleImageError = () => {
        setProfilePictureUrl('');
    };

    const handleFileSelect = async (e) => {
        const file = e.target.files?.[0];
        if (!file) return;

        setIsUploading(true);
        setUploadProgress('Uploading image...');

        try {
            const downloadURL = await uploadProfilePicture(user.id, file);
            setProfilePictureUrl(downloadURL);
            setUploadProgress('Upload complete!');

            setTimeout(() => {
                setUploadProgress('');
            }, 2000);
        } catch (error) {
            console.error('Upload error:', error);
            setUploadProgress('');

            if (error.message.includes('less than 5MB')) {
                alert('Image must be less than 5MB');
            } else if (error.message.includes('must be an image')) {
                alert('Please select an image file');
            } else {
                alert('Failed to upload image. Please try again.');
            }
        } finally {
            setIsUploading(false);
        }
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
                        onClick={onClose}
                        className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50"
                    />

                    <motion.div
                        initial={{ opacity: 0, x: 300 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: 300 }}
                        transition={springTransition}
                        className="fixed right-0 top-0 h-full w-full md:w-96 bg-[#0d0d0d] border-l border-[#2f3335] z-50 flex flex-col"
                    >
                        {/* Header */}
                        <div className="p-6 border-b border-[#2f3335] flex items-center justify-between">
                            <h2 className="text-lg font-bold text-white flex items-center gap-2">
                                <User className="w-5 h-5 text-[var(--color-primary)]" />
                                Profile
                            </h2>
                            <button
                                onClick={onClose}
                                className="text-[#958d73] hover:text-white transition-colors p-2 hover:bg-[#1a1a1a] rounded-lg"
                            >
                                <X className="w-5 h-5" />
                            </button>
                        </div>

                        {/* Content */}
                        <div className="flex-1 overflow-y-auto p-6 space-y-6">
                            {/* Profile Picture */}
                            <div className="flex flex-col items-center gap-4">
                                <div className="relative group">
                                    <motion.div
                                        whileHover={{ scale: isEditing ? 1.05 : 1 }}
                                        transition={springTransition}
                                        onClick={() => isEditing && !isUploading && fileInputRef.current?.click()}
                                        className={`w-32 h-32 rounded-full bg-[#1a1a1a] border-4 border-[#2f3335] flex items-center justify-center overflow-hidden relative ${
                                            isEditing && !isUploading ? 'cursor-pointer' : ''
                                        }`}
                                    >
                                        {profilePictureUrl ? (
                                            <img
                                                src={profilePictureUrl}
                                                alt={displayName}
                                                onError={handleImageError}
                                                className="w-full h-full object-cover"
                                            />
                                        ) : (
                                            <span className="text-4xl font-bold text-[#958d73]">
                                                {displayName.substring(0, 2).toUpperCase()}
                                            </span>
                                        )}

                                        {isEditing && !isUploading && (
                                            <div className="absolute inset-0 bg-black/50 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                                                <Camera className="w-8 h-8 text-white" />
                                            </div>
                                        )}

                                        {isUploading && (
                                            <div className="absolute inset-0 bg-black/70 flex items-center justify-center">
                                                <Loader className="w-8 h-8 text-white animate-spin" />
                                            </div>
                                        )}
                                    </motion.div>

                                    {isEditing && (
                                        <input
                                            ref={fileInputRef}
                                            type="file"
                                            accept="image/*"
                                            className="hidden"
                                            onChange={handleFileSelect}
                                            disabled={isUploading}
                                        />
                                    )}
                                </div>

                                {isEditing && (
                                    <motion.div
                                        initial={{ opacity: 0, y: -10 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        transition={springTransition}
                                        className="w-full space-y-3"
                                    >
                                        <button
                                            onClick={() => fileInputRef.current?.click()}
                                            disabled={isUploading}
                                            className={`w-full py-3 rounded-lg font-medium transition-all flex items-center justify-center gap-2 ${
                                                isUploading
                                                    ? 'bg-[#2f3335] text-[#5a5a5a] cursor-not-allowed'
                                                    : 'bg-[var(--color-primary)] text-black hover:bg-[#2d9248]'
                                            }`}
                                        >
                                            {isUploading ? (
                                                <>
                                                    <Loader className="w-4 h-4 animate-spin" />
                                                    {uploadProgress}
                                                </>
                                            ) : (
                                                <>
                                                    <Upload className="w-4 h-4" />
                                                    Upload Image
                                                </>
                                            )}
                                        </button>

                                        <div className="relative">
                                            <div className="absolute inset-0 flex items-center">
                                                <div className="w-full border-t border-[#2f3335]"></div>
                                            </div>
                                            <div className="relative flex justify-center text-xs">
                                                <span className="bg-[#0d0d0d] px-2 text-[#5a5a5a]">OR</span>
                                            </div>
                                        </div>

                                        <div>
                                            <label className="text-xs text-[#958d73] uppercase tracking-wider mb-2 block">
                                                Image URL
                                            </label>
                                            <input
                                                type="text"
                                                placeholder="https://example.com/avatar.jpg"
                                                value={profilePictureUrl}
                                                onChange={(e) => setProfilePictureUrl(e.target.value)}
                                                className="w-full bg-[#1a1a1a] border border-[#2f3335] focus:border-[var(--color-primary)] rounded-lg px-3 py-2 text-sm text-white placeholder-[#5a5a5a] focus:outline-none transition-colors"
                                            />
                                        </div>
                                    </motion.div>
                                )}
                            </div>

                            {/* Display Name */}
                            <div>
                                <label className="text-xs text-[#958d73] uppercase tracking-wider mb-2 block">
                                    Display Name
                                </label>
                                {isEditing ? (
                                    <motion.input
                                        initial={{ opacity: 0 }}
                                        animate={{ opacity: 1 }}
                                        transition={springTransition}
                                        type="text"
                                        value={displayName}
                                        onChange={(e) => setDisplayName(e.target.value)}
                                        className="w-full bg-[#1a1a1a] border border-[#2f3335] focus:border-[var(--color-primary)] rounded-lg px-4 py-3 text-sm text-white focus:outline-none transition-colors"
                                        placeholder="Enter your name"
                                    />
                                ) : (
                                    <div className="bg-[#1a1a1a] border border-[#2f3335] rounded-lg px-4 py-3">
                                        <p className="text-white font-medium">{displayName || 'Not set'}</p>
                                    </div>
                                )}
                            </div>

                            {/* Email (Read-only) */}
                            <div>
                                <label className="text-xs text-[#958d73] uppercase tracking-wider mb-2 block flex items-center gap-2">
                                    <Mail className="w-3 h-3" />
                                    Email Address
                                </label>
                                <div className="bg-[#1a1a1a] border border-[#2f3335] rounded-lg px-4 py-3">
                                    <p className="text-[#958d73] text-sm">{user.email}</p>
                                </div>
                                <p className="text-[10px] text-[#5a5a5a] mt-2">
                                    Email cannot be changed
                                </p>
                            </div>

                            {/* Account Info */}
                            <div className="pt-4 border-t border-[#2f3335] space-y-2">
                                <div className="flex items-center justify-between">
                                    <span className="text-xs text-[#958d73] uppercase tracking-wider">Role</span>
                                    <span className="text-xs text-white font-mono bg-[#1a1a1a] px-2 py-1 rounded">
                                        {user.role || 'user'}
                                    </span>
                                </div>
                                <div className="flex items-center justify-between">
                                    <span className="text-xs text-[#958d73] uppercase tracking-wider">Status</span>
                                    <span className="text-xs text-[#34a853] font-mono bg-[#34a853]/10 px-2 py-1 rounded">
                                        Active
                                    </span>
                                </div>
                            </div>
                        </div>

                        {/* Footer Actions */}
                        <div className="p-6 border-t border-[#2f3335] space-y-3">
                            {isEditing ? (
                                <>
                                    <motion.button
                                        whileHover={{ scale: 1.02 }}
                                        whileTap={{ scale: 0.98 }}
                                        transition={springTransition}
                                        onClick={handleSave}
                                        disabled={isSaving}
                                        className={`w-full bg-[#34a853] hover:bg-[#2d9248] text-white font-medium py-3 rounded-xl transition-all flex items-center justify-center gap-2 ${isSaving ? 'opacity-50 cursor-not-allowed' : ''
                                            }`}
                                    >
                                        <Save className="w-4 h-4" />
                                        {isSaving ? 'Saving...' : 'Save Changes'}
                                    </motion.button>
                                    <button
                                        onClick={() => {
                                            setIsEditing(false);
                                            setDisplayName(user.name || '');
                                            setProfilePictureUrl(user.profilePictureUrl || '');
                                        }}
                                        className="w-full bg-transparent border border-[#2f3335] hover:bg-[#1a1a1a] text-white font-medium py-3 rounded-xl transition-all"
                                    >
                                        Cancel
                                    </button>
                                </>
                            ) : (
                                <motion.button
                                    whileHover={{ scale: 1.02 }}
                                    whileTap={{ scale: 0.98 }}
                                    transition={springTransition}
                                    onClick={() => setIsEditing(true)}
                                    className="w-full bg-[#2f3335] hover:bg-[#3f4345] text-white font-medium py-3 rounded-xl transition-all"
                                >
                                    Edit Profile
                                </motion.button>
                            )}
                        </div>
                    </motion.div>
                </>
            )}
        </AnimatePresence>
    );
}
