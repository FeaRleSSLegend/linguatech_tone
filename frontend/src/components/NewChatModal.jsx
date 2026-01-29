import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { User, Users, X, Search, Plus, Check } from 'lucide-react';
import { useChat } from '../context/ChatContext';

export default function NewChatModal({ isOpen, onClose }) {
    const { createChat, availableUsers } = useChat();
    const [view, setView] = useState('selection'); // 'selection', 'direct', 'group'
    const [searchQuery, setSearchQuery] = useState('');
    const [groupName, setGroupName] = useState('');
    const [selectedUsers, setSelectedUsers] = useState([]);

    // Reset state when modal closes
    useEffect(() => {
        if (!isOpen) {
            // Slight delay to allow animation to finish before resetting state
            const timer = setTimeout(() => {
                setView('selection');
                setSearchQuery('');
                setGroupName('');
                setSelectedUsers([]);
            }, 300);
            return () => clearTimeout(timer);
        }
    }, [isOpen]);

    const filteredUsers = availableUsers.filter(user =>
        user.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        user.username.toLowerCase().includes(searchQuery.toLowerCase())
    );

    const handleBack = () => {
        setView('selection');
        setSearchQuery('');
        setGroupName('');
        setSelectedUsers([]);
    };

    const handleCreateDirect = (user) => {
        createChat('direct', { name: user.name, online: user.online });
        onClose();
    };

    const handleCreateGroup = () => {
        if (!groupName.trim()) return;
        createChat('group', { name: groupName });
        onClose();
    };

    const toggleUserSelection = (userId) => {
        if (selectedUsers.includes(userId)) {
            setSelectedUsers(prev => prev.filter(id => id !== userId));
        } else {
            setSelectedUsers(prev => [...prev, userId]);
        }
    };

    return (
        <AnimatePresence>
            {isOpen && (
                <>
                    {/* Backdrop */}
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        onClick={onClose}
                        className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50"
                    />

                    {/* Modal */}
                    <motion.div
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.95 }}
                        className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-md z-50 overflow-hidden"
                    >
                        <div className="bg-[#1e1e1e] border border-white/10 rounded-xl shadow-2xl overflow-hidden">

                            {/* View 1: Selection */}
                            {view === 'selection' && (
                                <div className="p-0">
                                    <div className="flex items-center justify-between p-4 border-b border-white/5">
                                        <h2 className="text-[#e0ddd9] font-medium">New Chat</h2>
                                        <button onClick={onClose} className="text-[#958d73] hover:text-white transition-colors">
                                            <X className="w-5 h-5" />
                                        </button>
                                    </div>
                                    <div className="p-2 space-y-1">
                                        <button
                                            onClick={() => setView('direct')}
                                            className="w-full flex items-center gap-4 p-2.5 rounded-lg hover:bg-white/5 transition-colors text-left group"
                                        >
                                            <div className="w-10 h-10 rounded-full bg-[#2f3335] flex items-center justify-center text-[var(--color-primary)] group-hover:bg-[var(--color-primary)]/20 transition-colors">
                                                <User className="w-5 h-5" />
                                            </div>
                                            <div>
                                                <h3 className="text-[#e0ddd9] text-sm font-medium">Direct Message</h3>
                                                <p className="text-[#958d73] text-xs">Chat with one person</p>
                                            </div>
                                        </button>

                                        <button
                                            onClick={() => setView('group')}
                                            className="w-full flex items-center gap-4 p-2.5 rounded-lg hover:bg-white/5 transition-colors text-left group"
                                        >
                                            <div className="w-10 h-10 rounded-full bg-[#2f3335] flex items-center justify-center text-[var(--color-primary)] group-hover:bg-[var(--color-primary)]/20 transition-colors">
                                                <Users className="w-5 h-5" />
                                            </div>
                                            <div>
                                                <h3 className="text-[#e0ddd9] text-sm font-medium">Group Chat</h3>
                                                <p className="text-[#958d73] text-xs">Create a group with multiple people</p>
                                            </div>
                                        </button>
                                    </div>
                                </div>
                            )}

                            {/* View 2: Direct Message Search */}
                            {view === 'direct' && (
                                <div className="p-0">
                                    <div className="flex items-center justify-between p-4 border-b border-white/5">
                                        <h2 className="text-[#e0ddd9] font-medium">Start a Chat</h2>
                                        <button onClick={onClose} className="text-[#958d73] hover:text-white transition-colors">
                                            <X className="w-5 h-5" />
                                        </button>
                                    </div>
                                    <div className="p-4 space-y-4">
                                        <div>
                                            <label className="text-[#958d73] text-xs uppercase tracking-wider mb-2 block">Search for a user</label>
                                            <div className="relative">
                                                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[#958d73]" />
                                                <input
                                                    type="text"
                                                    value={searchQuery}
                                                    onChange={(e) => setSearchQuery(e.target.value)}
                                                    placeholder="Search by username..."
                                                    className="w-full bg-[#080808] border border-[#2f3335] rounded-lg py-2.5 pl-10 pr-4 text-[#e0ddd9] placeholder-[#958d73] focus:outline-none focus:border-[var(--color-primary)] transition-colors"
                                                    autoFocus
                                                />
                                            </div>
                                        </div>

                                        {/* Results List */}
                                        <div className="max-h-[200px] overflow-y-auto space-y-1">
                                            {searchQuery && filteredUsers.map(user => (
                                                <button
                                                    key={user.id}
                                                    onClick={() => handleCreateDirect(user)}
                                                    className="w-full flex items-center gap-3 p-2 rounded-lg hover:bg-white/5 transition-colors text-left"
                                                >
                                                    <div className="w-8 h-8 rounded-full bg-[#2f3335] flex items-center justify-center text-[#e0ddd9] text-xs font-medium">
                                                        {user.name.substring(0, 2).toUpperCase()}
                                                    </div>
                                                    <div>
                                                        <h4 className="text-[#e0ddd9] text-sm">{user.name}</h4>
                                                        <div className="flex items-center gap-2">
                                                            <span className="text-[#958d73] text-xs">{user.username}</span>
                                                            {user.online && <span className="w-1.5 h-1.5 rounded-full bg-[var(--color-primary)]"></span>}
                                                        </div>
                                                    </div>
                                                </button>
                                            ))}
                                            {searchQuery && filteredUsers.length === 0 && (
                                                <p className="text-center text-[#958d73] text-sm py-2">No users found</p>
                                            )}
                                        </div>

                                        <button
                                            onClick={handleBack}
                                            className="w-full border border-[#2f3335] text-[#e0ddd9] hover:bg-white/5 font-medium py-2.5 rounded-lg transition-colors"
                                        >
                                            Back
                                        </button>
                                    </div>
                                </div>
                            )}

                            {/* View 3: Group Chat Creation */}
                            {view === 'group' && (
                                <div className="p-0">
                                    <div className="flex items-center justify-between p-4 border-b border-white/5">
                                        <h2 className="text-[#e0ddd9] font-medium">Create Group</h2>
                                        <button onClick={onClose} className="text-[#958d73] hover:text-white transition-colors">
                                            <X className="w-5 h-5" />
                                        </button>
                                    </div>
                                    <div className="p-4 space-y-4">
                                        {/* Group Name */}
                                        <div>
                                            <label className="text-[#958d73] text-xs uppercase tracking-wider mb-2 block">Group Name</label>
                                            <input
                                                type="text"
                                                value={groupName}
                                                onChange={(e) => setGroupName(e.target.value)}
                                                placeholder="Enter group name..."
                                                className="w-full bg-[#080808] border border-[#2f3335] rounded-lg py-2.5 px-4 text-[#e0ddd9] placeholder-[#958d73] focus:outline-none focus:border-[var(--color-primary)] transition-colors"
                                                autoFocus
                                            />
                                        </div>

                                        {/* Members */}
                                        <div>
                                            <label className="text-[#958d73] text-xs uppercase tracking-wider mb-2 block">Add members</label>
                                            <div className="relative mb-2">
                                                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[#958d73]" />
                                                <input
                                                    type="text"
                                                    value={searchQuery}
                                                    onChange={(e) => setSearchQuery(e.target.value)}
                                                    placeholder="Search by username..."
                                                    className="w-full bg-[#080808] border border-[#2f3335] rounded-lg py-2.5 pl-10 pr-4 text-[#e0ddd9] placeholder-[#958d73] focus:outline-none focus:border-[var(--color-primary)] transition-colors"
                                                />
                                            </div>

                                            {/* Results / Selection */}
                                            <div className="max-h-[150px] overflow-y-auto space-y-1">
                                                {/* Show selected first */}
                                                {selectedUsers.length > 0 && (
                                                    <div className="flex flex-wrap gap-2 mb-2">
                                                        {availableUsers.filter(u => selectedUsers.includes(u.id)).map(u => (
                                                            <span key={u.id} className="bg-[#2f3335] text-[#e0ddd9] text-xs px-2 py-1 rounded-full flex items-center gap-1">
                                                                {u.name}
                                                                <X className="w-3 h-3 cursor-pointer hover:text-white" onClick={() => toggleUserSelection(u.id)} />
                                                            </span>
                                                        ))}
                                                    </div>
                                                )}

                                                {searchQuery && filteredUsers.map(user => (
                                                    <button
                                                        key={user.id}
                                                        onClick={() => toggleUserSelection(user.id)}
                                                        className={`w-full flex items-center justify-between p-2 rounded-lg transition-colors text-left ${selectedUsers.includes(user.id) ? 'bg-[var(--color-primary)]/10 border border-[var(--color-primary)]/30' : 'hover:bg-white/5'
                                                            }`}
                                                    >
                                                        <div className="flex items-center gap-3">
                                                            <div className="w-8 h-8 rounded-full bg-[#2f3335] flex items-center justify-center text-[#e0ddd9] text-xs font-medium">
                                                                {user.name.substring(0, 2).toUpperCase()}
                                                            </div>
                                                            <div>
                                                                <h4 className="text-[#e0ddd9] text-sm">{user.name}</h4>
                                                                <span className="text-[#958d73] text-xs block">{user.username}</span>
                                                            </div>
                                                        </div>
                                                        {selectedUsers.includes(user.id) && <Check className="w-4 h-4 text-[var(--color-primary)]" />}
                                                    </button>
                                                ))}
                                            </div>
                                        </div>

                                        <div className="flex gap-3 pt-2">
                                            <button
                                                onClick={handleBack}
                                                className="flex-1 border border-[#2f3335] text-[#e0ddd9] hover:bg-white/5 font-medium py-2.5 rounded-lg transition-colors"
                                            >
                                                Back
                                            </button>
                                            <button
                                                onClick={handleCreateGroup}
                                                disabled={!groupName.trim()}
                                                className="flex-1 bg-[var(--color-primary)] hover:bg-[var(--color-primary-hover)] text-white font-medium py-2.5 rounded-lg transition-colors flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
                                            >
                                                <Plus className="w-4 h-4" />
                                                Create
                                            </button>
                                        </div>
                                    </div>
                                </div>
                            )}

                        </div>
                    </motion.div>
                </>
            )}
        </AnimatePresence>
    );
}
