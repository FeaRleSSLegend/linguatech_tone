import { useState, useRef, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

import {
    MessageSquare,
    ArrowLeft,
    Search,
    Send,
    X,
    Shield,
    Plus,
    MoreVertical,
    Video,
    Phone,
    UserX,
    Trash2,
    Clock,
    Ban,
    ShieldOff
} from 'lucide-react';

import MessageBubble from './MessageBubble';
import SuggestionCard from './SuggestionCard';
import { useAnalyze } from '../hooks/useAnalyze';
import { useChat } from '../context/ChatContext';
import { useAuth } from '../context/AuthContext';
import { useSocket } from '../context/SocketContext';

export default function ChatInterface({ onAnalysisUpdate, onBackToChats }) {
    const { onlineUsers } = useSocket();
    const {
        messages,
        sendMessage,
        activeChatId,
        chats,
        setActiveChatId,
        typingUsers,
        blockUser,
        unblockUser,
        deleteChat,
        emitTyping
    } = useChat();

    const { user } = useAuth();
    
    const [inputValue, setInputValue] = useState('');
    const [showSuggestion, setShowSuggestion] = useState(false);
    const [pendingMessage, setPendingMessage] = useState(null);
    const [isSearching, setIsSearching] = useState(false);
    const [searchQuery, setSearchQuery] = useState('');
    const [isActionsMenuOpen, setIsActionsMenuOpen] = useState(false);
    const [replyToMessage, setReplyToMessage] = useState(null);

    const typingTimeoutRef = useRef(null);
    const messagesEndRef = useRef(null);
    const actionsMenuRef = useRef(null);

    const { analysis, isAnalyzing } = useAnalyze(inputValue, 500);

    /* -------------------- Typing Indicator -------------------- */
    useEffect(() => {
        if (!inputValue.trim()) {
            emitTyping(false);
            return;
        }

        emitTyping(true);

        if (typingTimeoutRef.current) {
            clearTimeout(typingTimeoutRef.current);
        }

        typingTimeoutRef.current = setTimeout(() => {
            emitTyping(false);
        }, 3000);

        return () => {
            if (typingTimeoutRef.current) {
                clearTimeout(typingTimeoutRef.current);
            }
        };
    }, [inputValue, emitTyping]);

    /* -------------------- Active Chat -------------------- */
    const activeChat = useMemo(
        () => chats.find(c => c.id === activeChatId),
        [chats, activeChatId]
    );
    

    /* -------------------- Toxic Message Count -------------------- */
    const toxicMessageCount = useMemo(() => {
        if (!activeChatId) return 0;
        return messages.filter(m =>
            m.chatId === activeChatId &&
            (m.analysis?.toxicity?.label === 'toxic' ||
             m.analysis?.toxicity?.label === 'very toxic' ||
             m.analysis?.sentiment?.label === 'negative')
        ).length;
    }, [messages, activeChatId]);

    /* -------------------- Ban Status Check -------------------- */
    const isTempBanned = useMemo(() => {
        if (!activeChat || !user || activeChat.type !== 'group') return false;
        const tempBannedUsers = activeChat.tempBannedUsers || [];
        const banRecord = tempBannedUsers.find(b => b.userId === user.id);
        if (!banRecord) return false;
        // Check if ban is still active
        return Date.now() < banRecord.bannedUntil;
    }, [activeChat, user]);

    const isPermBanned = useMemo(() => {
        if (!activeChat || !user || activeChat.type !== 'group') return false;
        const permBannedUsers = activeChat.permBannedUsers || [];
        return permBannedUsers.includes(user.id);
    }, [activeChat, user]);

    /* -------------------- Block Status Check -------------------- */
    const blockStatus = useMemo(() => {
        if (!activeChat || !user || activeChat.type !== 'direct') {
            return { isBlocked: false, iBlockedThem: false, theyBlockedMe: false };
        }

        const otherUserId = activeChat.participants?.find(id => id !== user.id);
        const iBlockedThem = activeChat.blockedBy?.includes(user.id) || false;
        const theyBlockedMe = activeChat.blockedBy?.includes(otherUserId) || false;

        return {
            isBlocked: iBlockedThem || theyBlockedMe,
            iBlockedThem,
            theyBlockedMe
        };
    }, [activeChat, user]);

    const { isBlocked, iBlockedThem, theyBlockedMe } = blockStatus;

    /* -------------------- Messages (Search Optimized) -------------------- */
    const activeMessages = useMemo(() => {
        return messages.filter(m => {
            if (m.chatId !== activeChatId) return false;
            if (!isSearching || !searchQuery) return true;
            return m.text.toLowerCase().includes(searchQuery.toLowerCase());
        });
    }, [messages, activeChatId, isSearching, searchQuery]);

    /* -------------------- Analysis Propagation -------------------- */
    useEffect(() => {
        if (onAnalysisUpdate) {
            onAnalysisUpdate(analysis, isAnalyzing);
        }
    }, [analysis, isAnalyzing, onAnalysisUpdate]);

    /* -------------------- Scroll -------------------- */
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [activeMessages]);

    /* -------------------- Close Actions Menu on Outside Click -------------------- */
    useEffect(() => {
        if (!isActionsMenuOpen) return;

        const handleClickOutside = (event) => {
            if (actionsMenuRef.current && !actionsMenuRef.current.contains(event.target)) {
                setIsActionsMenuOpen(false);
            }
        };

        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, [isActionsMenuOpen]);

    /* -------------------- Action Handlers -------------------- */
    const handleBlockUser = async () => {
        if (!activeChat || activeChat.type !== 'direct') return;

        // Determine the other user's ID in a direct chat
        const otherUserId = activeChat.participants?.find(id => id !== user.id);

        if (window.confirm(`Are you sure you want to block this user? You will no longer be able to send or receive messages.`)) {
            try {
                await blockUser(activeChat.id, otherUserId);
                setIsActionsMenuOpen(false);
            } catch (error) {
                console.error("Failed to block user:", error);
                alert("Failed to block user. Please try again.");
            }
        }
    };

    const handleUnblockUser = async () => {
        if (!activeChat || activeChat.type !== 'direct') return;

        // Determine the other user's ID in a direct chat
        const otherUserId = activeChat.participants?.find(id => id !== user.id);

        if (window.confirm(`Unblock this user? You will be able to send and receive messages again.`)) {
            try {
                await unblockUser(activeChat.id, otherUserId);
                setIsActionsMenuOpen(false);
            } catch (error) {
                console.error("Failed to unblock user:", error);
                alert("Failed to unblock user. Please try again.");
            }
        }
    };

    const handleDeleteChat = async () => {
        if (!activeChat) return;

        if (window.confirm("Reset this chat? This will permanently delete all message history. The chat will reappear if a new message is sent.")) {
            try {
                await deleteChat(activeChat.id);
                setActiveChatId(null);
            } catch (error) {
                console.error("Failed to reset chat:", error);
                alert("Failed to reset chat. Please try again.");
            }
        }
    };

    /* -------------------- Send Message -------------------- */
    const handleSend = () => {
        if (!inputValue.trim() || isAnalyzing) return;

        if (analysis?.should_warn) {
            setPendingMessage(inputValue);
            setShowSuggestion(true);
            return;
        }

        // Include replyTo reference if replying
        const messageData = {
            ...(analysis?.toxicity || { label: 'safe', confidence: 0 }),
            ...(replyToMessage && { replyTo: { id: replyToMessage.id, text: replyToMessage.text, sender: replyToMessage.sender.name } })
        };

        sendMessage(inputValue, messageData);
        setInputValue('');
        setSearchQuery('');
        setReplyToMessage(null);
    };

    /* -------------------- Reply Handler -------------------- */
    const handleReply = (message) => {
        setReplyToMessage(message);
    };

    return (
        <div className="flex flex-col h-full bg-[#050505] relative">
            {/* Header */}
            <header className="h-20 border-b border-[#2f3335] flex items-center justify-between px-6 bg-[#080808]/50 backdrop-blur-xl shrink-0 relative z-10">
                <div className="flex items-center gap-4">
                    <button
                        onClick={() => {
                            if (onBackToChats) {
                                onBackToChats();
                            } else {
                                setActiveChatId(null);
                            }
                        }}
                        className="md:hidden text-[#958d73] hover:text-[#e0ddd9] transition-colors"
                        title="Back to chats"
                    >
                        <ArrowLeft className="w-6 h-6" />
                    </button>
                    {activeChat ? (
                        <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-full bg-[#1e1e1e] flex items-center justify-center text-[#958d73] border border-[#2f3335]">
                                {(() => {
                                    // For direct chats, show other participant's initials
                                    if (activeChat.type === 'direct' && activeChat.participantNames && activeChat.participants) {
                                        const otherUserId = activeChat.participants.find(id => id !== user?.id);
                                        const otherName = activeChat.participantNames[otherUserId];
                                        return otherName ? otherName.substring(0, 2).toUpperCase() : activeChat.name.substring(0, 2).toUpperCase();
                                    }
                                    return activeChat.name.substring(0, 2).toUpperCase();
                                })()}
                            </div>
                            <div>
                                <h2 className="text-sm font-bold text-[#e0ddd9] flex items-center gap-2">
                                    {(() => {
                                        // For direct chats, show only other participant's name
                                        if (activeChat.type === 'direct' && activeChat.participantNames && activeChat.participants) {
                                            const otherUserId = activeChat.participants.find(id => id !== user?.id);
                                            return activeChat.participantNames[otherUserId] || activeChat.name;
                                        }
                                        return activeChat.name;
                                    })()}
                                </h2>
                                <p className="text-[10px] text-[#958d73] flex items-center gap-1.5 uppercase tracking-wider font-semibold">
                                    {activeChat.type === 'direct' ? (
                                        (() => {
                                            // 1. Identify the person you're chatting with
                                            const otherUserId = activeChat.participants?.find(id => id !== user?.id);
                                            // 2. Check if they are in the real-time online list
                                            const isOnline = onlineUsers.includes(otherUserId);
                                            
                                            return (
                                                <>
                                                    <span className={`w-1.5 h-1.5 rounded-full ${
                                                        isOnline 
                                                        ? 'bg-[#34a853] shadow-[0_0_8px_#34a853]' 
                                                        : 'bg-[#2f3335]'
                                                    }`} />
                                                    {isOnline ? 'Online' : 'Offline'}
                                                </>
                                            );
                                        })()
                                    ) : (
                                        <>
                                            <MessageSquare className="w-3 h-3" />
                                            Collaborative Channel
                                        </>
                                    )}
                                </p>
                            </div>
                        </div>
                    ) : (
                        <h2 className="text-sm font-bold text-[#e0ddd9]">Tone Chat</h2>
                    )}
                </div>

                <div className="flex items-center gap-2 relative">
                    <button
                        onClick={() => setIsSearching(!isSearching)}
                        className={`p-2 rounded-lg transition-colors ${isSearching ? 'text-[var(--color-primary)] bg-[var(--color-primary)]/10' : 'text-[#958d73] hover:text-[#e0ddd9]'}`}
                        title="Search messages"
                    >
                        <Search className="w-5 h-5" />
                    </button>

                    {activeChat && (
                        <div className="relative" ref={actionsMenuRef}>
                            <button
                                onClick={() => setIsActionsMenuOpen(!isActionsMenuOpen)}
                                className="p-2 rounded-lg transition-colors text-[#958d73] hover:text-[#e0ddd9] hover:bg-[#1e1e1e]"
                                title="Chat actions"
                            >
                                <MoreVertical className="w-5 h-5" />
                            </button>

                            <AnimatePresence>
                                {isActionsMenuOpen && (
                                    <motion.div
                                        initial={{ opacity: 0, y: -10, scale: 0.95 }}
                                        animate={{ opacity: 1, y: 0, scale: 1 }}
                                        exit={{ opacity: 0, y: -10, scale: 0.95 }}
                                        transition={{ duration: 0.15 }}
                                        className="absolute right-0 top-full mt-2 w-48 bg-[#0d0d0d] border border-[#2f3335] rounded-xl shadow-2xl overflow-hidden z-50 origin-top-right"
                                    >
                                        <button
                                            onClick={() => { alert('Video call coming soon'); setIsActionsMenuOpen(false); }}
                                            className="w-full px-4 py-3 text-left text-sm text-[#e0ddd9] hover:bg-[#1a1a1a] transition-colors flex items-center gap-3"
                                        >
                                            <Video className="w-4 h-4 text-[var(--color-primary)]" />
                                            Video Call
                                        </button>
                                        <button
                                            disabled
                                            className="w-full px-4 py-3 text-left text-sm text-[#5a5a5a] cursor-not-allowed flex items-center gap-3"
                                        >
                                            <Phone className="w-4 h-4" />
                                            Voice Call
                                            <span className="ml-auto text-[10px] bg-[#2f3335] px-2 py-0.5 rounded">Soon</span>
                                        </button>
                                        <div className="h-px bg-[#2f3335] my-1" />
                                        {activeChat.type === 'direct' && (
                                            isBlocked ? (
                                                <button
                                                    onClick={handleUnblockUser}
                                                    className="w-full px-4 py-3 text-left text-sm text-green-500 hover:bg-green-500/10 transition-colors flex items-center gap-3"
                                                >
                                                    <ShieldOff className="w-4 h-4" />
                                                    Unblock User
                                                </button>
                                            ) : (
                                                <button
                                                    onClick={handleBlockUser}
                                                    className="w-full px-4 py-3 text-left text-sm text-yellow-500 hover:bg-yellow-500/10 transition-colors flex items-center gap-3"
                                                >
                                                    <UserX className="w-4 h-4" />
                                                    Block User
                                                </button>
                                            )
                                        )}
                                        <button
                                            onClick={handleDeleteChat}
                                            className="w-full px-4 py-3 text-left text-sm text-red-500 hover:bg-red-500/10 transition-colors flex items-center gap-3"
                                        >
                                            <Trash2 className="w-4 h-4" />
                                            Reset Chat
                                        </button>
                                    </motion.div>
                                )}
                            </AnimatePresence>
                        </div>
                    )}
                </div>

                {/* Inline Search Bar */}
                <AnimatePresence>
                    {isSearching && (
                        <motion.div
                            initial={{ opacity: 0, y: -10 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -10 }}
                            className="absolute top-20 left-0 right-0 h-14 bg-[#0d0d0d] border-b border-[#2f3335] px-6 flex items-center gap-3 z-20 shadow-xl"
                        >
                            <Search className="w-4 h-4 text-[#958d73]" />
                            <input
                                type="text"
                                placeholder="Search message history..."
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                                className="flex-1 bg-transparent text-sm focus:outline-none text-[#e0ddd9] placeholder-[#5a5a5a]"
                                autoFocus
                            />
                            <button onClick={() => { setIsSearching(false); setSearchQuery(''); }} className="text-[#958d73] hover:text-[#e0ddd9]">
                                <X className="w-4 h-4" />
                            </button>
                        </motion.div>
                    )}
                </AnimatePresence>
            </header>

            {/* Content Area */}
            <div className="flex-1 overflow-hidden flex flex-col md:flex-row">
                {/* Messages Section */}
                <div className="flex-1 flex flex-col h-full">
                    {!activeChatId ? (
                        <div className="flex-1 flex flex-col items-center justify-center text-center p-8 space-y-6">
                            <div className="w-20 h-20 rounded-3xl bg-[#0d0d0d] border border-[#1a1a1a] flex items-center justify-center shadow-2xl relative">
                                <MessageSquare className="w-10 h-10 text-[var(--color-primary)]" />
                                <div className="absolute -top-1 -right-1 w-4 h-4 bg-[var(--color-primary)] rounded-full blur-md opacity-50" />
                            </div>
                            <div className="max-w-md">
                                <h3 className="text-xl font-bold text-white mb-2">Select a chat to start</h3>
                                <p className="text-[#958d73] text-sm leading-relaxed">
                                    Choose a conversation from the sidebar or send a friend request to connect with someone new.
                                </p>
                            </div>
                        </div>
                    ) : (
                        <>
                            {/* Message List */}
                            <div className={`flex-1 overflow-y-auto px-4 py-6 scrollbar-thin scrollbar-thumb-[#2f3335] scrollbar-track-transparent transition-all duration-300 ${isSearching ? 'pt-20' : 'pt-6'}`}>
                                <div className="max-w-4xl mx-auto space-y-6">
                                    <AnimatePresence mode="popLayout">
                                        {activeMessages.map((m, idx) => (
                                            <motion.div
                                                key={m.id || idx}
                                                initial={{ opacity: 0, scale: 0.95, y: 10 }}
                                                animate={{ opacity: 1, scale: 1, y: 0 }}
                                                layout
                                            >
                                                <MessageBubble
                                                    message={m}
                                                    isOwn={m.sender.id === user?.id}
                                                    onReply={handleReply}
                                                />
                                            </motion.div>
                                        ))}
                                    </AnimatePresence>

                                    {/* Typing Indicator */}
                                    {typingUsers.length > 0 && (
                                        <div className="flex items-center gap-3 text-[#958d73] text-[10px] italic">
                                            <div className="flex gap-1 bg-[#1a1a1a] px-2 py-1.5 rounded-full border border-white/5">
                                                <span className="w-1 h-1 bg-[#958d73] rounded-full animate-bounce [animation-delay:-0.3s]" />
                                                <span className="w-1 h-1 bg-[#958d73] rounded-full animate-bounce [animation-delay:-0.15s]" />
                                                <span className="w-1 h-1 bg-[#958d73] rounded-full animate-bounce" />
                                            </div>
                                            <span>
                                                {typingUsers.length === 1
                                                    ? `${typingUsers[0]} is refining thoughts...`
                                                    : `${typingUsers.join(', ')} are collaborating...`}
                                            </span>
                                        </div>
                                    )}
                                    <div ref={messagesEndRef} />
                                </div>
                            </div>

                            {/* Input Area */}
                            <div className="p-4 md:p-6 shrink-0 bg-[#080808]/80 backdrop-blur-xl border-t border-[#2f3335]">
                                <div className="max-w-3xl mx-auto">
                                    {/* REPLY BOX */}
                                    <AnimatePresence>
                                        {replyToMessage && (
                                            <motion.div
                                                initial={{ opacity: 0, y: 10 }}
                                                animate={{ opacity: 1, y: 0 }}
                                                exit={{ opacity: 0, y: 10 }}
                                                className="mb-3 px-4 py-2 bg-[#1a1a1a] border border-[#2f3335] rounded-xl flex items-center gap-3"
                                            >
                                                <div className="flex-1">
                                                    <p className="text-[10px] text-[var(--color-primary)] font-semibold mb-0.5">
                                                        Replying to {replyToMessage.sender.name}
                                                    </p>
                                                    <p className="text-xs text-[#958d73] truncate">
                                                        {replyToMessage.text}
                                                    </p>
                                                </div>
                                                <button
                                                    onClick={() => setReplyToMessage(null)}
                                                    className="text-[#958d73] hover:text-white transition-colors"
                                                >
                                                    <X className="w-4 h-4" />
                                                </button>
                                            </motion.div>
                                        )}
                                    </AnimatePresence>

                                    {/* LIVE ANALYSIS BAR */}
                                    <AnimatePresence>
                                        {inputValue.trim() && (
                                            <motion.div
                                                initial={{ opacity: 0, y: 10 }}
                                                animate={{ opacity: 1, y: 0 }}
                                                exit={{ opacity: 0, y: 10 }}
                                                className="mb-3 px-4 py-2 bg-[#0d0d0d] border border-white/5 rounded-xl flex items-center gap-4 overflow-hidden"
                                            >
                                                <div className="text-xl shrink-0">
                                                    {isAnalyzing ? '🤔' : analysis?.emoji || '😐'}
                                                </div>
                                                <div className="flex-1 space-y-1.5">
                                                    <div className="flex items-center justify-between">
                                                        <span className="text-[10px] uppercase tracking-widest font-bold text-[#958d73]">
                                                            {isAnalyzing ? 'Analyzing Resonance...' : (analysis?.feedback || 'Steady Tone')}
                                                        </span>
                                                        {!isAnalyzing && (
                                                            <span className={`text-[10px] font-bold uppercase ${analysis?.toxicity?.label === 'toxic' ? 'text-red-500' :
                                                                analysis?.toxicity?.label === 'warning' ? 'text-yellow-500' : 'text-[#34a853]'
                                                                }`}>
                                                                {analysis?.toxicity?.label === 'safe' ? 'Safe' : analysis?.toxicity?.label}
                                                            </span>
                                                        )}
                                                    </div>
                                                    <div className="h-1 bg-[#1a1a1a] rounded-full overflow-hidden">
                                                        <motion.div
                                                            initial={{ width: 0 }}
                                                            animate={{ width: isAnalyzing ? '30%' : `${(analysis?.sentiment?.confidence || 0.5) * 100}%` }}
                                                            className={`h-full transition-colors duration-500 ${isAnalyzing ? 'bg-[#34a853] animate-pulse' :
                                                                analysis?.toxicity?.label === 'toxic' ? 'bg-red-500' :
                                                                    analysis?.toxicity?.label === 'warning' ? 'bg-yellow-500' : 'bg-[#34a853]'
                                                                }`}
                                                        />
                                                    </div>
                                                </div>
                                            </motion.div>
                                        )}
                                    </AnimatePresence>

                                    {isTempBanned ? (
                                        <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-2xl p-6 text-center">
                                            <Clock className="w-8 h-8 text-yellow-500 mx-auto mb-3" />
                                            <h3 className="text-lg font-bold text-yellow-500 mb-2">Temporarily Banned</h3>
                                            <p className="text-sm text-[#958d73]">You have been temporarily banned from sending messages in this group. Your ban will expire soon.</p>
                                        </div>
                                    ) : isPermBanned ? (
                                        <div className="bg-red-500/10 border border-red-500/30 rounded-2xl p-6 text-center">
                                            <Ban className="w-8 h-8 text-red-500 mx-auto mb-3" />
                                            <h3 className="text-lg font-bold text-red-500 mb-2">Permanently Banned</h3>
                                            <p className="text-sm text-[#958d73]">You have been permanently banned from this group and cannot send messages.</p>
                                        </div>
                                    ) : (
                                        <div className={`bg-[#111111] border border-[#2f3335] rounded-2xl p-2 relative group transition-colors shadow-2xl ${!isBlocked && 'focus-within:border-[var(--color-primary)]'}`}>
                                            <div className="flex items-end gap-2 px-3 py-2">
                                                <textarea
                                                    rows="1"
                                                    placeholder={
                                                        isBlocked
                                                            ? (iBlockedThem ? "You blocked this contact" : "You were blocked by this contact")
                                                            : `Message ${activeChat?.name}...`
                                                    }
                                                    value={inputValue}
                                                    onChange={(e) => setInputValue(e.target.value)}
                                                    onKeyDown={(e) => {
                                                        if (e.key === 'Enter' && !e.shiftKey) {
                                                            e.preventDefault();
                                                            handleSend();
                                                        }
                                                    }}
                                                    disabled={isBlocked}
                                                    className={`flex-1 bg-transparent border-none focus:ring-0 text-[#e0ddd9] placeholder-[#5a5a5a] text-sm resize-none py-2 max-h-40 ${isBlocked && 'cursor-not-allowed opacity-50'}`}
                                                />
                                                <button
                                                    onClick={handleSend}
                                                    disabled={!inputValue.trim() || isBlocked}
                                                    className={`p-2.5 rounded-xl transition-all shadow-lg ${inputValue.trim() && !isBlocked
                                                        ? 'bg-[#34a853] text-white hover:bg-[#2d9248] shadow-emerald-900/40'
                                                        : 'bg-[#1a1a1a] text-[#5a5a5a] cursor-not-allowed'
                                                        }`}
                                                >
                                                    <Send className="w-5 h-5" />
                                                </button>
                                            </div>

                                        {/* Status Indicators */}
                                        <div className="flex items-center justify-between px-3 pb-2 pt-1 border-t border-white/5 mt-1">
                                            <div className="flex items-center gap-4">
                                                <div className="flex items-center gap-1.5 min-w-[100px]">
                                                    <Shield className={`w-3 h-3 ${analysis?.toxicity?.label === 'toxic' ? 'text-red-500' : 'text-[#34a853]'}`} />
                                                    <span className="text-[10px] text-[#958d73] uppercase tracking-widest font-bold">Resonance Shield Active</span>
                                                </div>
                                            </div>
                                            <span className="text-[9px] text-[#5a5a5a] font-mono">
                                                {inputValue.length} CHARS
                                            </span>
                                        </div>
                                        </div>
                                    )}
                                    {!isTempBanned && !isPermBanned && (
                                        <p className="mt-3 text-[10px] text-center text-[#5a5a5a]">
                                            Tone uses AI to refine communication. <a href="/terms" className="hover:text-[#958d73] underline">Beta Feature</a>
                                        </p>
                                    )}
                                </div>
                            </div>
                        </>
                    )}
                </div>
            </div>

            {/* Safety Suggestion Overlay */}
            <SuggestionCard
                isVisible={showSuggestion}
                originalMessage={pendingMessage}
                suggestion={analysis?.rephrase}
                reason={analysis?.rephrase?.reason}
                onUseSuggestion={(s) => {
                    setInputValue(s);
                    setShowSuggestion(false);
                    setPendingMessage(null);
                }}
                onEdit={() => {
                    setShowSuggestion(false);
                    setPendingMessage(null);
                }}
                onSendAnyway={() => {
                    sendMessage(pendingMessage, analysis?.toxicity);
                    setInputValue('');
                    setShowSuggestion(false);
                    setPendingMessage(null);
                }}
                onClose={() => {
                    setShowSuggestion(false);
                    setPendingMessage(null);
                }}
            />
        </div>
    );
}