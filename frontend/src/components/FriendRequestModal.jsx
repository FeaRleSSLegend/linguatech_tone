import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Search, UserPlus, Check, XCircle, Loader } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import {
    searchUserByEmail,
    sendFriendRequest,
    subscribeToFriendRequests,
    acceptFriendRequest,
    rejectFriendRequest
} from '../lib/firebase';

const springTransition = {
    type: "spring",
    stiffness: 100,
    damping: 20
};

export default function FriendRequestModal({ isOpen, onClose }) {
    const { user } = useAuth();
    const [activeTab, setActiveTab] = useState('send'); // 'send' or 'received'
    const [searchEmail, setSearchEmail] = useState('');
    const [searchResult, setSearchResult] = useState(null);
    const [isSearching, setIsSearching] = useState(false);
    const [isSending, setIsSending] = useState(false);
    const [friendRequests, setFriendRequests] = useState([]);
    const [searchError, setSearchError] = useState('');

    // Subscribe to friend requests ONLY when modal is open
    useEffect(() => {
        // Don't log or subscribe if modal is closed
        if (!isOpen) return;

        if (!user) {
            console.log('🚫 [FriendRequestModal] No user authenticated');
            return;
        }

        console.log('🎯 [FriendRequestModal] Starting listener for user:', user.id);

        const unsubscribe = subscribeToFriendRequests(user.id, (requests) => {
            console.log('📥 [FriendRequestModal] Received requests:', requests);
            setFriendRequests(requests);
        });

        return () => {
            console.log('🔚 [FriendRequestModal] Unsubscribing listener');
            unsubscribe();
        };
    }, [user, isOpen]);

    const handleSearch = async () => {
        if (!searchEmail.trim()) return;

        setIsSearching(true);
        setSearchError('');
        setSearchResult(null);

        try {
            const foundUser = await searchUserByEmail(searchEmail.trim());

            if (!foundUser) {
                setSearchError('No user found with this email');
            } else if (foundUser.id === user.id) {
                setSearchError('Cannot send request to yourself');
            } else {
                setSearchResult(foundUser);
            }
        } catch (error) {
            console.error('Search error:', error);
            setSearchError('Search failed. Please try again.');
        } finally {
            setIsSearching(false);
        }
    };

    const handleSendRequest = async () => {
        if (!searchResult) return;

        setIsSending(true);
        try {
            await sendFriendRequest(user.id, searchResult.id);
            alert('Friend request sent!');
            setSearchEmail('');
            setSearchResult(null);
        } catch (error) {
            console.error('Send request error:', error);
            if (error.message.includes('already sent')) {
                alert('You already sent a request to this user');
            } else if (error.message.includes('already sent you')) {
                alert('This user already sent you a request. Check "Received" tab.');
            } else {
                alert('Failed to send friend request');
            }
        } finally {
            setIsSending(false);
        }
    };

    const handleAccept = async (requestId, fromUserId) => {
        try {
            await acceptFriendRequest(requestId, fromUserId, user.id);
            alert('Friend request accepted! Chat created.');
        } catch (error) {
            console.error('Accept error:', error);
            alert('Failed to accept request');
        }
    };

    const handleReject = async (requestId) => {
        try {
            await rejectFriendRequest(requestId);
        } catch (error) {
            console.error('Reject error:', error);
            alert('Failed to reject request');
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
                                    <UserPlus className="w-5 h-5 text-[var(--color-primary)]" />
                                    Friend Requests
                                </h2>
                                <button
                                    onClick={onClose}
                                    className="text-[#958d73] hover:text-white transition-colors p-2 hover:bg-[#1a1a1a] rounded-lg"
                                >
                                    <X className="w-5 h-5" />
                                </button>
                            </div>

                            {/* Tabs */}
                            <div className="flex border-b border-[#2f3335]">
                                <button
                                    onClick={() => setActiveTab('send')}
                                    className={`flex-1 py-3 px-4 text-sm font-medium transition-colors relative ${activeTab === 'send'
                                        ? 'text-white bg-[#1a1a1a]'
                                        : 'text-[#958d73] hover:text-white hover:bg-[#1a1a1a]'
                                        }`}
                                >
                                    Send Request
                                    {activeTab === 'send' && (
                                        <motion.div
                                            layoutId="activeTab"
                                            className="absolute bottom-0 left-0 right-0 h-0.5 bg-[var(--color-primary)]"
                                            transition={springTransition}
                                        />
                                    )}
                                </button>
                                <button
                                    onClick={() => setActiveTab('received')}
                                    className={`flex-1 py-3 px-4 text-sm font-medium transition-colors relative ${activeTab === 'received'
                                        ? 'text-white bg-[#1a1a1a]'
                                        : 'text-[#958d73] hover:text-white hover:bg-[#1a1a1a]'
                                        }`}
                                >
                                    Received
                                    {friendRequests.length > 0 && (
                                        <span className="ml-2 bg-red-500 text-white text-xs font-bold rounded-full px-2 py-0.5 shadow-[0_0_8px_rgba(239,68,68,0.5)]">
                                            {friendRequests.length > 9 ? '9+' : friendRequests.length}
                                        </span>
                                    )}
                                    {activeTab === 'received' && (
                                        <motion.div
                                            layoutId="activeTab"
                                            className="absolute bottom-0 left-0 right-0 h-0.5 bg-[var(--color-primary)]"
                                            transition={springTransition}
                                        />
                                    )}
                                </button>
                            </div>

                            {/* Content */}
                            <div className="p-6 max-h-96 overflow-y-auto">
                                <AnimatePresence mode="wait">
                                    {activeTab === 'send' ? (
                                        <motion.div
                                            key="send"
                                            initial={{ opacity: 0, x: -20 }}
                                            animate={{ opacity: 1, x: 0 }}
                                            exit={{ opacity: 0, x: 20 }}
                                            transition={springTransition}
                                            className="space-y-4"
                                        >
                                            <p className="text-sm text-[#958d73]">
                                                Search for a user by email to send a friend request
                                            </p>

                                            <div className="relative">
                                                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[#958d73]" />
                                                <input
                                                    type="email"
                                                    placeholder="user@example.com"
                                                    value={searchEmail}
                                                    onChange={(e) => setSearchEmail(e.target.value)}
                                                    onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                                                    className="w-full bg-[#1a1a1a] border border-[#2f3335] focus:border-[var(--color-primary)] rounded-lg pl-10 pr-4 py-3 text-sm text-white placeholder-[#5a5a5a] focus:outline-none transition-colors"
                                                />
                                            </div>

                                            <button
                                                onClick={handleSearch}
                                                disabled={isSearching || !searchEmail.trim()}
                                                className={`w-full py-2.5 rounded-lg font-medium transition-all flex items-center justify-center gap-2 ${!searchEmail.trim()
                                                    ? 'bg-[#2f3335] text-[#5a5a5a] cursor-not-allowed'
                                                    : 'bg-[var(--color-primary)] text-black hover:bg-[#2d9248]'
                                                    }`}
                                            >
                                                {isSearching ? (
                                                    <>
                                                        <Loader className="w-4 h-4 animate-spin" />
                                                        Searching...
                                                    </>
                                                ) : (
                                                    <>
                                                        <Search className="w-4 h-4" />
                                                        Search
                                                    </>
                                                )}
                                            </button>

                                            {searchError && (
                                                <motion.div
                                                    initial={{ opacity: 0, y: -10 }}
                                                    animate={{ opacity: 1, y: 0 }}
                                                    className="bg-red-500/10 border border-red-500/30 rounded-lg p-3 text-sm text-red-400"
                                                >
                                                    {searchError}
                                                </motion.div>
                                            )}

                                            {searchResult && (
                                                <motion.div
                                                    initial={{ opacity: 0, y: -10 }}
                                                    animate={{ opacity: 1, y: 0 }}
                                                    transition={springTransition}
                                                    className="bg-[#1a1a1a] border border-[#2f3335] rounded-lg p-4 space-y-3"
                                                >
                                                    <div className="flex items-center gap-3">
                                                        <div className="w-12 h-12 rounded-full bg-[#2f3335] flex items-center justify-center text-white font-medium">
                                                            {searchResult.name?.substring(0, 2).toUpperCase() || 'U'}
                                                        </div>
                                                        <div className="flex-1">
                                                            <h3 className="text-white font-medium">
                                                                {searchResult.name || 'Unknown User'}
                                                            </h3>
                                                            <p className="text-sm text-[#958d73]">{searchResult.email}</p>
                                                        </div>
                                                    </div>

                                                    <button
                                                        onClick={handleSendRequest}
                                                        disabled={isSending}
                                                        className={`w-full py-2.5 rounded-lg font-medium transition-all flex items-center justify-center gap-2 ${isSending
                                                            ? 'bg-[#2f3335] text-[#5a5a5a] cursor-not-allowed'
                                                            : 'bg-[var(--color-primary)] text-black hover:bg-[#2d9248]'
                                                            }`}
                                                    >
                                                        {isSending ? (
                                                            <>
                                                                <Loader className="w-4 h-4 animate-spin" />
                                                                Sending...
                                                            </>
                                                        ) : (
                                                            <>
                                                                <UserPlus className="w-4 h-4" />
                                                                Send Friend Request
                                                            </>
                                                        )}
                                                    </button>
                                                </motion.div>
                                            )}
                                        </motion.div>
                                    ) : (
                                        <motion.div
                                            key="received"
                                            initial={{ opacity: 0, x: 20 }}
                                            animate={{ opacity: 1, x: 0 }}
                                            exit={{ opacity: 0, x: -20 }}
                                            transition={springTransition}
                                            className="space-y-3"
                                        >
                                            {friendRequests.length === 0 ? (
                                                <div className="text-center py-8">
                                                    <UserPlus className="w-12 h-12 text-[#2f3335] mx-auto mb-3" />
                                                    <p className="text-[#958d73] text-sm">No pending requests</p>
                                                </div>
                                            ) : (
                                                friendRequests.map((request) => (
                                                    <motion.div
                                                        key={request.id}
                                                        initial={{ opacity: 0, y: -10 }}
                                                        animate={{ opacity: 1, y: 0 }}
                                                        transition={springTransition}
                                                        className="bg-[#1a1a1a] border border-[#2f3335] rounded-lg p-4"
                                                    >
                                                        <div className="flex items-center gap-3 mb-3">
                                                            <div className="w-12 h-12 rounded-full bg-[#2f3335] flex items-center justify-center text-white font-medium">
                                                                {request.fromUser?.name?.substring(0, 2).toUpperCase() || 'U'}
                                                            </div>
                                                            <div className="flex-1">
                                                                <h3 className="text-white font-medium">
                                                                    {request.fromUser?.name || 'Unknown User'}
                                                                </h3>
                                                                <p className="text-sm text-[#958d73]">
                                                                    {request.fromUser?.email}
                                                                </p>
                                                            </div>
                                                        </div>

                                                        <div className="flex gap-2">
                                                            <button
                                                                onClick={() => handleAccept(request.id, request.fromUserId)}
                                                                className="flex-1 bg-[var(--color-primary)] hover:bg-[#2d9248] text-black font-medium py-2 rounded-lg transition-all flex items-center justify-center gap-2"
                                                            >
                                                                <Check className="w-4 h-4" />
                                                                Accept
                                                            </button>
                                                            <button
                                                                onClick={() => handleReject(request.id)}
                                                                className="flex-1 bg-transparent border border-red-500/50 hover:bg-red-500/10 text-red-400 font-medium py-2 rounded-lg transition-all flex items-center justify-center gap-2"
                                                            >
                                                                <XCircle className="w-4 h-4" />
                                                                Reject
                                                            </button>
                                                        </div>
                                                    </motion.div>
                                                ))
                                            )}
                                        </motion.div>
                                    )}
                                </AnimatePresence>
                            </div>
                        </div>
                    </motion.div>
                </>
            )}
        </AnimatePresence>
    );
}
