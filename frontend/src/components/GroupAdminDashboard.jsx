import { useState, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    X, Users, Shield, Settings, Crown, Ban, UserX, RotateCcw,
    Trash2, AlertTriangle, TrendingUp, Lock, Unlock, Image as ImageIcon
} from 'lucide-react';
import { doc, getDoc, updateDoc, deleteDoc, arrayUnion, arrayRemove, serverTimestamp } from 'firebase/firestore';
import { db } from '../lib/firebase';
import { getUserProfile } from '../lib/firebase';

export default function GroupAdminDashboard({ isOpen, onClose, groupId, currentUser }) {
    const [activeTab, setActiveTab] = useState('members');
    const [groupData, setGroupData] = useState(null);
    const [members, setMembers] = useState([]);
    const [loading, setLoading] = useState(true);
    const [blacklistWord, setBlacklistWord] = useState('');

    // Settings form state
    const [groupName, setGroupName] = useState('');
    const [groupDescription, setGroupDescription] = useState('');
    const [groupAvatar, setGroupAvatar] = useState('');
    const [strictMode, setStrictMode] = useState(false);

    // Fetch group data and members
    useEffect(() => {
        if (!isOpen || !groupId) return;

        const fetchGroupData = async () => {
            try {
                const groupRef = doc(db, 'chats', groupId);
                const groupSnap = await getDoc(groupRef);

                if (groupSnap.exists()) {
                    const data = groupSnap.data();
                    setGroupData(data);

                    // Set settings form
                    setGroupName(data.name || '');
                    setGroupDescription(data.description || '');
                    setGroupAvatar(data.profilePictureUrl || '');
                    setStrictMode(data.strictMode || false);

                    // Fetch member details
                    const memberPromises = data.participants.map(async (userId) => {
                        const profile = await getUserProfile(userId);
                        const toxicCount = data.moderationStats?.[userId] || 0;
                        return {
                            id: userId,
                            name: profile?.name || 'Unknown',
                            email: profile?.email || '',
                            role: profile?.role || 'member',
                            toxicCount,
                            isAdmin: data.admins?.includes(userId) || false,
                            isTempBanned: data.tempBannedUsers?.some(b => b.userId === userId && Date.now() < b.bannedUntil) || false,
                            isPermBanned: data.permBannedUsers?.includes(userId) || false
                        };
                    });

                    const membersData = await Promise.all(memberPromises);
                    setMembers(membersData);
                }
            } catch (error) {
                console.error('Error fetching group data:', error);
            } finally {
                setLoading(false);
            }
        };

        fetchGroupData();
    }, [isOpen, groupId]);

    // Top 5 Offenders
    const topOffenders = useMemo(() => {
        return [...members]
            .sort((a, b) => b.toxicCount - a.toxicCount)
            .slice(0, 5)
            .filter(m => m.toxicCount > 0);
    }, [members]);

    // Handle kick member
    const handleKick = async (userId) => {
        if (!window.confirm('Kick this member from the group?')) return;

        try {
            const groupRef = doc(db, 'chats', groupId);
            await updateDoc(groupRef, {
                participants: arrayRemove(userId),
                members: arrayRemove(userId),
                updatedAt: serverTimestamp()
            });

            setMembers(prev => prev.filter(m => m.id !== userId));
            alert('Member kicked successfully');
        } catch (error) {
            console.error('Error kicking member:', error);
            alert('Failed to kick member');
        }
    };

    // Handle ban member
    const handleBan = async (userId, userName) => {
        if (!window.confirm(`Permanently ban ${userName}? They will not be able to rejoin.`)) return;

        try {
            const groupRef = doc(db, 'chats', groupId);
            await updateDoc(groupRef, {
                permBannedUsers: arrayUnion(userId),
                participants: arrayRemove(userId),
                members: arrayRemove(userId),
                updatedAt: serverTimestamp()
            });

            setMembers(prev => prev.filter(m => m.id !== userId));
            alert(`${userName} has been permanently banned`);
        } catch (error) {
            console.error('Error banning member:', error);
            alert('Failed to ban member');
        }
    };

    // Handle reset toxicity count
    const handleResetToxicity = async (userId, userName) => {
        if (!window.confirm(`Reset toxicity count for ${userName}?`)) return;

        try {
            const groupRef = doc(db, 'chats', groupId);
            await updateDoc(groupRef, {
                [`moderationStats.${userId}`]: 0,
                updatedAt: serverTimestamp()
            });

            setMembers(prev => prev.map(m => m.id === userId ? { ...m, toxicCount: 0 } : m));
            alert(`Toxicity count reset for ${userName}`);
        } catch (error) {
            console.error('Error resetting toxicity:', error);
            alert('Failed to reset toxicity count');
        }
    };

    // Handle add blacklist word
    const handleAddBlacklistWord = async () => {
        if (!blacklistWord.trim()) return;

        try {
            const groupRef = doc(db, 'chats', groupId);
            await updateDoc(groupRef, {
                wordBlacklist: arrayUnion(blacklistWord.toLowerCase().trim()),
                updatedAt: serverTimestamp()
            });

            setGroupData(prev => ({
                ...prev,
                wordBlacklist: [...(prev.wordBlacklist || []), blacklistWord.toLowerCase().trim()]
            }));
            setBlacklistWord('');
        } catch (error) {
            console.error('Error adding blacklist word:', error);
            alert('Failed to add word to blacklist');
        }
    };

    // Handle remove blacklist word
    const handleRemoveBlacklistWord = async (word) => {
        try {
            const groupRef = doc(db, 'chats', groupId);
            await updateDoc(groupRef, {
                wordBlacklist: arrayRemove(word),
                updatedAt: serverTimestamp()
            });

            setGroupData(prev => ({
                ...prev,
                wordBlacklist: (prev.wordBlacklist || []).filter(w => w !== word)
            }));
        } catch (error) {
            console.error('Error removing blacklist word:', error);
            alert('Failed to remove word from blacklist');
        }
    };

    // Handle save group settings
    const handleSaveSettings = async () => {
        try {
            const groupRef = doc(db, 'chats', groupId);
            await updateDoc(groupRef, {
                name: groupName,
                description: groupDescription,
                profilePictureUrl: groupAvatar,
                strictMode: strictMode,
                updatedAt: serverTimestamp()
            });

            alert('Group settings updated successfully');
        } catch (error) {
            console.error('Error updating group settings:', error);
            alert('Failed to update group settings');
        }
    };

    if (!isOpen) return null;

    return (
        <AnimatePresence>
            <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
                <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.95 }}
                    className="w-full max-w-4xl bg-[#0d0d0d] border border-[#2f3335] rounded-2xl shadow-2xl overflow-hidden"
                >
                    {/* Header */}
                    <div className="flex items-center justify-between p-6 border-b border-[#2f3335] bg-[#080808]">
                        <div className="flex items-center gap-3">
                            <div className="p-2 bg-[var(--color-primary)]/10 rounded-xl">
                                <Shield className="w-6 h-6 text-[var(--color-primary)]" />
                            </div>
                            <div>
                                <h2 className="text-xl font-bold text-white">Admin Dashboard</h2>
                                <p className="text-sm text-[#958d73]">{groupData?.name || 'Loading...'}</p>
                            </div>
                        </div>
                        <button
                            onClick={onClose}
                            className="p-2 hover:bg-[#1a1a1a] rounded-lg transition-colors"
                        >
                            <X className="w-5 h-5 text-[#958d73]" />
                        </button>
                    </div>

                    {/* Tabs */}
                    <div className="flex border-b border-[#2f3335] bg-[#0a0a0a]">
                        {[
                            { id: 'members', label: 'Members', icon: Users },
                            { id: 'moderation', label: 'Moderation', icon: Shield },
                            { id: 'settings', label: 'Settings', icon: Settings }
                        ].map(tab => (
                            <button
                                key={tab.id}
                                onClick={() => setActiveTab(tab.id)}
                                className={`flex-1 flex items-center justify-center gap-2 px-6 py-4 text-sm font-semibold transition-colors ${
                                    activeTab === tab.id
                                        ? 'text-[var(--color-primary)] border-b-2 border-[var(--color-primary)] bg-[var(--color-primary)]/5'
                                        : 'text-[#958d73] hover:text-[#e0ddd9] hover:bg-[#1a1a1a]'
                                }`}
                            >
                                <tab.icon className="w-4 h-4" />
                                {tab.label}
                            </button>
                        ))}
                    </div>

                    {/* Content */}
                    <div className="p-6 max-h-[600px] overflow-y-auto scrollbar-thin scrollbar-thumb-[#2f3335] scrollbar-track-transparent">
                        {loading ? (
                            <div className="flex items-center justify-center py-20">
                                <div className="w-8 h-8 border-4 border-[var(--color-primary)] border-t-transparent rounded-full animate-spin" />
                            </div>
                        ) : (
                            <>
                                {/* Tab 1: Members */}
                                {activeTab === 'members' && (
                                    <div className="space-y-4">
                                        <div className="flex items-center justify-between mb-6">
                                            <h3 className="text-lg font-bold text-white">Member Management</h3>
                                            <span className="text-sm text-[#958d73]">{members.length} members</span>
                                        </div>

                                        {members.map(member => (
                                            <div
                                                key={member.id}
                                                className="flex items-center justify-between p-4 bg-[#111111] border border-[#2f3335] rounded-xl hover:border-[var(--color-primary)]/30 transition-colors"
                                            >
                                                <div className="flex items-center gap-3">
                                                    <div className="w-10 h-10 rounded-full bg-[#1e1e1e] border border-[#2f3335] flex items-center justify-center">
                                                        <span className="text-sm font-bold text-[var(--color-primary)]">
                                                            {member.name.substring(0, 2).toUpperCase()}
                                                        </span>
                                                    </div>
                                                    <div>
                                                        <div className="flex items-center gap-2">
                                                            <p className="text-sm font-semibold text-white">{member.name}</p>
                                                            {member.id === groupData?.creatorId && (
                                                                <Crown className="w-3.5 h-3.5 text-yellow-500" />
                                                            )}
                                                            {member.isAdmin && member.id !== groupData?.creatorId && (
                                                                <Shield className="w-3.5 h-3.5 text-blue-500" />
                                                            )}
                                                        </div>
                                                        <p className="text-xs text-[#958d73]">{member.email}</p>
                                                        <div className="flex items-center gap-2 mt-1">
                                                            <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${
                                                                member.toxicCount >= 3
                                                                    ? 'bg-red-500/20 text-red-500'
                                                                    : member.toxicCount > 0
                                                                    ? 'bg-yellow-500/20 text-yellow-500'
                                                                    : 'bg-green-500/20 text-green-500'
                                                            }`}>
                                                                {member.toxicCount} toxic messages
                                                            </span>
                                                            {member.isTempBanned && (
                                                                <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-orange-500/20 text-orange-500">
                                                                    TEMP BAN
                                                                </span>
                                                            )}
                                                        </div>
                                                    </div>
                                                </div>

                                                {member.id !== groupData?.creatorId && member.id !== currentUser.id && (
                                                    <div className="flex items-center gap-2">
                                                        <button
                                                            onClick={() => handleResetToxicity(member.id, member.name)}
                                                            className="p-2 bg-blue-500/10 hover:bg-blue-500/20 text-blue-500 rounded-lg transition-colors"
                                                            title="Reset toxicity count"
                                                        >
                                                            <RotateCcw className="w-4 h-4" />
                                                        </button>
                                                        <button
                                                            onClick={() => handleKick(member.id)}
                                                            className="p-2 bg-yellow-500/10 hover:bg-yellow-500/20 text-yellow-500 rounded-lg transition-colors"
                                                            title="Kick from group"
                                                        >
                                                            <UserX className="w-4 h-4" />
                                                        </button>
                                                        <button
                                                            onClick={() => handleBan(member.id, member.name)}
                                                            className="p-2 bg-red-500/10 hover:bg-red-500/20 text-red-500 rounded-lg transition-colors"
                                                            title="Permanently ban"
                                                        >
                                                            <Ban className="w-4 h-4" />
                                                        </button>
                                                    </div>
                                                )}
                                            </div>
                                        ))}
                                    </div>
                                )}

                                {/* Tab 2: Moderation */}
                                {activeTab === 'moderation' && (
                                    <div className="space-y-6">
                                        {/* Top 5 Offenders */}
                                        <div>
                                            <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
                                                <TrendingUp className="w-5 h-5 text-red-500" />
                                                Leaderboard of Shame (Top 5)
                                            </h3>
                                            {topOffenders.length === 0 ? (
                                                <p className="text-sm text-[#958d73] text-center py-8">No toxic messages reported yet</p>
                                            ) : (
                                                <div className="space-y-2">
                                                    {topOffenders.map((member, index) => (
                                                        <div
                                                            key={member.id}
                                                            className="flex items-center justify-between p-3 bg-[#111111] border border-[#2f3335] rounded-lg"
                                                        >
                                                            <div className="flex items-center gap-3">
                                                                <span className={`text-lg font-bold ${
                                                                    index === 0 ? 'text-red-500' :
                                                                    index === 1 ? 'text-orange-500' :
                                                                    index === 2 ? 'text-yellow-500' :
                                                                    'text-[#958d73]'
                                                                }`}>
                                                                    #{index + 1}
                                                                </span>
                                                                <p className="text-sm font-semibold text-white">{member.name}</p>
                                                            </div>
                                                            <span className="text-sm font-bold text-red-500">
                                                                {member.toxicCount} toxic messages
                                                            </span>
                                                        </div>
                                                    ))}
                                                </div>
                                            )}
                                        </div>

                                        {/* Word Blacklist */}
                                        <div>
                                            <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
                                                <AlertTriangle className="w-5 h-5 text-yellow-500" />
                                                Word Blacklist
                                            </h3>
                                            <div className="flex gap-2 mb-4">
                                                <input
                                                    type="text"
                                                    value={blacklistWord}
                                                    onChange={(e) => setBlacklistWord(e.target.value)}
                                                    onKeyDown={(e) => e.key === 'Enter' && handleAddBlacklistWord()}
                                                    placeholder="Enter word to blacklist..."
                                                    className="flex-1 bg-[#111111] border border-[#2f3335] rounded-xl px-4 py-2.5 text-sm text-white focus:outline-none focus:border-[var(--color-primary)] transition-colors"
                                                />
                                                <button
                                                    onClick={handleAddBlacklistWord}
                                                    className="px-6 py-2.5 bg-[var(--color-primary)] hover:bg-[#2d9248] text-black font-bold rounded-xl transition-all"
                                                >
                                                    Add
                                                </button>
                                            </div>
                                            <div className="flex flex-wrap gap-2">
                                                {(groupData?.wordBlacklist || []).map((word, index) => (
                                                    <div
                                                        key={index}
                                                        className="flex items-center gap-2 px-3 py-1.5 bg-red-500/10 border border-red-500/30 rounded-lg"
                                                    >
                                                        <span className="text-sm text-red-500 font-semibold">{word}</span>
                                                        <button
                                                            onClick={() => handleRemoveBlacklistWord(word)}
                                                            className="hover:bg-red-500/20 rounded p-0.5 transition-colors"
                                                        >
                                                            <X className="w-3.5 h-3.5 text-red-500" />
                                                        </button>
                                                    </div>
                                                ))}
                                                {(!groupData?.wordBlacklist || groupData.wordBlacklist.length === 0) && (
                                                    <p className="text-sm text-[#958d73] py-2">No blacklisted words yet</p>
                                                )}
                                            </div>
                                        </div>
                                    </div>
                                )}

                                {/* Tab 3: Settings */}
                                {activeTab === 'settings' && (
                                    <div className="space-y-6">
                                        <h3 className="text-lg font-bold text-white mb-4">Group Settings</h3>

                                        <div>
                                            <label className="block text-sm font-semibold text-[#958d73] mb-2">Group Name</label>
                                            <input
                                                type="text"
                                                value={groupName}
                                                onChange={(e) => setGroupName(e.target.value)}
                                                className="w-full bg-[#111111] border border-[#2f3335] rounded-xl px-4 py-2.5 text-sm text-white focus:outline-none focus:border-[var(--color-primary)] transition-colors"
                                            />
                                        </div>

                                        <div>
                                            <label className="block text-sm font-semibold text-[#958d73] mb-2">Description</label>
                                            <textarea
                                                value={groupDescription}
                                                onChange={(e) => setGroupDescription(e.target.value)}
                                                rows={3}
                                                className="w-full bg-[#111111] border border-[#2f3335] rounded-xl px-4 py-2.5 text-sm text-white focus:outline-none focus:border-[var(--color-primary)] transition-colors resize-none"
                                            />
                                        </div>

                                        <div>
                                            <label className="block text-sm font-semibold text-[#958d73] mb-2">Avatar URL</label>
                                            <div className="flex gap-2">
                                                <input
                                                    type="text"
                                                    value={groupAvatar}
                                                    onChange={(e) => setGroupAvatar(e.target.value)}
                                                    placeholder="https://example.com/avatar.png"
                                                    className="flex-1 bg-[#111111] border border-[#2f3335] rounded-xl px-4 py-2.5 text-sm text-white focus:outline-none focus:border-[var(--color-primary)] transition-colors"
                                                />
                                                {groupAvatar && (
                                                    <div className="w-10 h-10 rounded-lg overflow-hidden border border-[#2f3335]">
                                                        <img src={groupAvatar} alt="Preview" className="w-full h-full object-cover" />
                                                    </div>
                                                )}
                                            </div>
                                        </div>

                                        <div className="flex items-center justify-between p-4 bg-[#111111] border border-[#2f3335] rounded-xl">
                                            <div>
                                                <div className="flex items-center gap-2">
                                                    <Shield className="w-4 h-4 text-[var(--color-primary)]" />
                                                    <p className="text-sm font-semibold text-white">Strict Mode (AI Auto-Mod)</p>
                                                </div>
                                                <p className="text-xs text-[#958d73] mt-1">
                                                    When enabled, AI automatically detects and penalizes toxic messages
                                                </p>
                                            </div>
                                            <button
                                                onClick={() => setStrictMode(!strictMode)}
                                                className={`relative w-12 h-6 rounded-full transition-colors ${
                                                    strictMode ? 'bg-[var(--color-primary)]' : 'bg-[#2f3335]'
                                                }`}
                                            >
                                                <div className={`absolute top-0.5 w-5 h-5 bg-white rounded-full transition-transform ${
                                                    strictMode ? 'translate-x-6' : 'translate-x-0.5'
                                                }`} />
                                            </button>
                                        </div>

                                        <button
                                            onClick={handleSaveSettings}
                                            className="w-full px-6 py-3 bg-[var(--color-primary)] hover:bg-[#2d9248] text-black font-bold rounded-xl transition-all"
                                        >
                                            Save Settings
                                        </button>
                                    </div>
                                )}
                            </>
                        )}
                    </div>
                </motion.div>
            </div>
        </AnimatePresence>
    );
}
