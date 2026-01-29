import { useState, useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
import { Search, LogOut, Shield, UserPlus, MessageSquare, Plus } from 'lucide-react';
import { useChat } from '../context/ChatContext';
import { useAuth } from '../context/AuthContext';
import FriendRequestModal from './FriendRequestModal';
import ProfileView from './ProfileView';
import CreateGroupModal from './CreateGroupModal';
import JoinGroupModal from './JoinGroupModal';
import Logo from './Logo';
import NotificationBell from './NotificationBell';
import NotificationDropdown from './NotificationDropdown';
import { subscribeToUserPresence, subscribeToFriendRequests, acceptFriendRequest, rejectFriendRequest } from '../lib/firebase';

function stringToColor(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
        hash = str.charCodeAt(i) + ((hash << 5) - hash);
    }
    const hue = hash % 360;
    return `hsl(${hue}, 65%, 50%)`;
}

function getDisplayName(chat, currentUserId) {
    // For groups, show group name
    if (chat.type === 'group' || chat.isGroup) {
        return chat.name;
    }

    // For direct chats, show OTHER participant's name
    if (chat.participantNames && chat.participants) {
        const otherUserId = chat.participants.find(id => id !== currentUserId);
        if (otherUserId && chat.participantNames[otherUserId]) {
            return chat.participantNames[otherUserId];
        }
    }

    // Fallback: parse the "A & B" format
    if (chat.name && chat.name.includes(' & ')) {
        const names = chat.name.split(' & ');
        // Return first name as fallback
        return names[0];
    }

    // Final fallback
    return chat.name || 'Unknown';
}

function getDisplayInitials(chat, currentUserId) {
    const displayName = getDisplayName(chat, currentUserId);
    return displayName.substring(0, 2).toUpperCase();
}

export default function Sidebar({ isCollapsed, onToggle, onViewChange, pendingRequestsCount = 0 }) {
    const { chats, messages, activeChatId, setActiveChatId, unreadCounts } = useChat();
    const { logout, user } = useAuth();
    const [isFriendRequestModalOpen, setIsFriendRequestModalOpen] = useState(false);
    const [isProfileOpen, setIsProfileOpen] = useState(false);
    const [searchQuery, setSearchQuery] = useState('');
    const [activeTab, setActiveTab] = useState('direct');
    const [onlineStatuses, setOnlineStatuses] = useState({});
    const [hasUnreadMessages, setHasUnreadMessages] = useState(false);
    const [isNotificationDropdownOpen, setIsNotificationDropdownOpen] = useState(false);
    const [friendRequests, setFriendRequests] = useState([]);
    const [isCreateGroupModalOpen, setIsCreateGroupModalOpen] = useState(false);
    const [isJoinGroupModalOpen, setIsJoinGroupModalOpen] = useState(false);
    const bellRef = useRef(null);

    const groupCount = chats.filter(c => c.type === 'group').length;

    useEffect(() => {
        const unsubscribers = [];

        chats.forEach(chat => {
            if (chat.type === 'direct' && chat.participants) {
                const otherUserId = chat.participants.find(p => p !== user?.id);
                if (otherUserId) {
                    const unsubscribe = subscribeToUserPresence(otherUserId, (userData) => {
                        setOnlineStatuses(prev => ({
                            ...prev,
                            [chat.id]: userData?.isOnline || false
                        }));
                    });
                    unsubscribers.push(unsubscribe);
                }
            }
        });

        return () => unsubscribers.forEach(unsub => unsub && unsub());
    }, [chats, user]);

    // Real-time friend request listener (for dropdown)
    useEffect(() => {
        if (!user) return;

        const unsubscribe = subscribeToFriendRequests(user.id, (requests) => {
            setFriendRequests(requests);
        });

        return () => unsubscribe();
    }, [user]);

    // Real-time unread message detection (GREEN dot)
    useEffect(() => {
        const totalUnread = Object.values(unreadCounts).reduce((sum, count) => sum + count, 0);
        setHasUnreadMessages(totalUnread > 0);
    }, [unreadCounts]);

    const handleAcceptRequest = async (requestId, fromUserId) => {
        try {
            await acceptFriendRequest(requestId, fromUserId, user.id);
            console.log('✅ [Sidebar] Friend request accepted');
        } catch (error) {
            console.error('❌ [Sidebar] Accept error:', error);
            alert('Failed to accept request');
        }
    };

    const handleRejectRequest = async (requestId) => {
        try {
            await rejectFriendRequest(requestId);
            console.log('✅ [Sidebar] Friend request rejected');
        } catch (error) {
            console.error('❌ [Sidebar] Reject error:', error);
            alert('Failed to reject request');
        }
    };

    const getLastMessageData = (chatId) => {
        const chatMessages = messages.filter(m => m.chatId === chatId);
        if (chatMessages.length === 0) return null;
        return [...chatMessages].sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp)).pop();
    };

    const formatTime = (timestamp) => {
        if (!timestamp) return '';
        const date = new Date(timestamp);
        const now = new Date();
        const diff = now - date;
        const minutes = Math.floor(diff / 60000);

        if (minutes < 1) return 'Just now';
        if (minutes < 60) return `${minutes}m`;
        const hours = Math.floor(minutes / 60);
        if (hours < 24) return `${hours}h`;
        return `${Math.floor(hours / 24)}d`;
    };

    const filteredChats = chats.filter(chat => {
        // Filter by active tab
        if (chat.type !== activeTab) return false;

        // Filter by search query
        if (!chat.name.toLowerCase().includes(searchQuery.toLowerCase())) return false;

        // Hide groups where user is permanently banned
        if (chat.type === 'group' && chat.permBannedUsers && chat.permBannedUsers.includes(user?.id)) {
            return false;
        }

        return true;
    }).sort((a, b) => (b.lastActivity || 0) - (a.lastActivity || 0));

    const avatarColor = user?.id ? stringToColor(user.id) : '#958d73';
    const userInitials = user?.name ? user.name.substring(0, 2).toUpperCase() : 'U';

    return (
        <>
            {/* Desktop Sidebar */}
            <div className="hidden md:flex w-full h-full flex-col bg-[#080808]">
                {/* Header */}
                <div className="p-4 flex items-center justify-between border-b border-[#2f3335]">
                    <div className="flex items-center gap-3">
                        <Logo className="w-8 h-8" showText={true} />
                        <div className="h-6 w-px bg-[#2f3335]" />
                        <div className="relative">
                            <NotificationBell
                                ref={bellRef}
                                count={pendingRequestsCount}
                                isActive={isNotificationDropdownOpen}
                                onClick={() => setIsNotificationDropdownOpen(!isNotificationDropdownOpen)}
                            />
                            <NotificationDropdown
                                isOpen={isNotificationDropdownOpen}
                                onClose={() => setIsNotificationDropdownOpen(false)}
                                requests={friendRequests}
                                onAccept={handleAcceptRequest}
                                onReject={handleRejectRequest}
                                anchorRef={bellRef}
                            />
                        </div>
                    </div>

                    <div className="flex items-center gap-2">
                        <button
                            onClick={() => setIsFriendRequestModalOpen(true)}
                            className="text-[#958d73] hover:text-[var(--color-primary)] transition-colors p-2 hover:bg-[var(--color-primary)]/10 rounded-lg"
                            title="Send Friend Request"
                        >
                            <UserPlus className="w-5 h-5" />
                        </button>

                        {user?.role === 'admin' || user?.role === 'super_admin' ? (
                            <Link
                                to="/admin/reports"
                                className="bg-red-500/10 hover:bg-red-500/20 text-red-500 transition-all p-2 rounded-lg"
                                title="Admin Dashboard"
                            >
                                <Shield className="w-5 h-5" />
                            </Link>
                        ) : null}

                        <button
                            onClick={logout}
                            className="text-[#958d73] hover:text-red-400 transition-colors p-2 hover:bg-red-500/10 rounded-lg"
                            title="Logout"
                        >
                            <LogOut className="w-5 h-5" />
                        </button>
                    </div>
                </div>

                {/* Search */}
                <div className="px-4 pt-4 pb-2">
                    <div className="relative group">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[#958d73] group-focus-within:text-[#e0ddd9] transition-colors" />
                        <input
                            type="text"
                            placeholder="Search chats..."
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            className="w-full bg-[#111111] border border-transparent focus:border-[#2f3335] rounded-xl py-2.5 pl-9 pr-4 text-sm text-[#e0ddd9] placeholder-[#5a5a5a] focus:outline-none transition-all"
                        />
                    </div>
                </div>

                {/* Tabs */}
                <div className="px-4 flex items-center gap-1 mb-2">
                    <button
                        onClick={() => setActiveTab('direct')}
                        className={`text-xs font-semibold py-2 px-4 rounded-xl transition-colors flex-1 relative ${activeTab === 'direct' ? 'bg-[#1e1e1e] text-white' : 'text-[#958d73] hover:text-[#e0ddd9]'
                            }`}
                    >
                        Direct
                        {(() => {
                            const directUnreadCount = chats
                                .filter(c => c.type === 'direct')
                                .reduce((sum, c) => sum + (unreadCounts[c.id] || 0), 0);
                            return directUnreadCount > 0 ? (
                                <span className="ml-2 bg-red-500 text-white text-[10px] font-bold rounded-full px-1.5 py-0.5 min-w-[18px] inline-block text-center">
                                    {directUnreadCount > 9 ? '9+' : directUnreadCount}
                                </span>
                            ) : null;
                        })()}
                    </button>
                    <button
                        onClick={() => setActiveTab('group')}
                        className={`text-xs font-semibold py-2 px-4 rounded-xl transition-colors flex-1 relative ${activeTab === 'group' ? 'bg-[#1e1e1e] text-white' : 'text-[#958d73] hover:text-[#e0ddd9]'
                            }`}
                    >
                        Groups ({groupCount})
                        {(() => {
                            const groupUnreadCount = chats
                                .filter(c => c.type === 'group')
                                .reduce((sum, c) => sum + (unreadCounts[c.id] || 0), 0);
                            return groupUnreadCount > 0 ? (
                                <span className="ml-2 bg-red-500 text-white text-[10px] font-bold rounded-full px-1.5 py-0.5 min-w-[18px] inline-block text-center">
                                    {groupUnreadCount > 9 ? '9+' : groupUnreadCount}
                                </span>
                            ) : null;
                        })()}
                    </button>
                </div>

                {/* Group Actions (shown only in groups tab) */}
                {activeTab === 'group' && (
                    <div className="px-4 mb-2 space-y-2">
                        <button
                            onClick={() => setIsCreateGroupModalOpen(true)}
                            className="w-full py-2.5 px-4 bg-[var(--color-primary)] hover:bg-[#2d9248] text-black font-medium rounded-xl transition-all flex items-center justify-center gap-2"
                        >
                            <Plus className="w-4 h-4" />
                            Create Group
                        </button>
                        <button
                            onClick={() => setIsJoinGroupModalOpen(true)}
                            className="w-full py-2.5 px-4 bg-[#1a1a1a] border border-[#2f3335] hover:bg-[#2f3335] text-[#e0ddd9] font-medium rounded-xl transition-all flex items-center justify-center gap-2"
                        >
                            <UserPlus className="w-4 h-4" />
                            Join with Code
                        </button>
                    </div>
                )}

                {/* Chat List */}
                <div className="flex-1 overflow-y-auto px-2 space-y-1">
                    {filteredChats.length === 0 && activeTab === 'direct' && (
                        <div className="text-center py-12 px-4">
                            <MessageSquare className="w-16 h-16 text-[#2f3335] mx-auto mb-4 opacity-50" />
                            <p className="text-[#958d73] text-sm mb-4 leading-relaxed">
                                No connections yet—search for a friend to start chatting
                            </p>
                            <button
                                onClick={() => setIsFriendRequestModalOpen(true)}
                                className="text-xs text-[var(--color-primary)] hover:text-[#2d9248] transition-colors font-medium"
                            >
                                Find friends
                            </button>
                        </div>
                    )}

                    {filteredChats.map(chat => {
                        const lastMsg = getLastMessageData(chat.id);
                        const isOnline = onlineStatuses[chat.id] || false;
                        const displayName = getDisplayName(chat, user?.id);
                        const displayInitials = getDisplayInitials(chat, user?.id);
                        return (
                            <button
                                key={chat.id}
                                onClick={() => setActiveChatId(chat.id)}
                                className={`w-full flex items-center gap-3 p-2.5 rounded-lg transition-colors text-left group ${activeChatId === chat.id ? 'bg-[#2f3335]' : 'hover:bg-[#1e1e1e]'
                                    }`}
                            >
                                <div className="w-10 h-10 rounded-full bg-[#1e1e1e] flex items-center justify-center text-[#958d73] border border-[#2f3335] group-hover:border-white/10 relative shrink-0">
                                    {chat.type === 'direct' && isOnline && (
                                        <span className="absolute bottom-0 right-0 w-2.5 h-2.5 bg-[var(--color-primary)] border-2 border-[#080808] rounded-full"></span>
                                    )}
                                    <span className="text-sm font-medium">{displayInitials}</span>
                                    {unreadCounts[chat.id] > 0 && (
                                        <span className="absolute -top-1 -right-1 w-5 h-5 bg-red-500 text-white text-[10px] font-bold rounded-full flex items-center justify-center">
                                            {unreadCounts[chat.id]}
                                        </span>
                                    )}
                                </div>

                                <div className="flex-1 min-w-0">
                                    <div className="flex items-center justify-between">
                                        <h3 className={`text-sm font-medium truncate ${activeChatId === chat.id ? 'text-white' : 'text-[#e0ddd9]'}`}>
                                            {displayName}
                                        </h3>
                                        <span className="text-[10px] text-[#958d73]">
                                            {lastMsg ? formatTime(lastMsg.timestamp) : ''}
                                        </span>
                                    </div>
                                    <p className="text-xs text-[#958d73] truncate">
                                        {lastMsg ? lastMsg.text : 'No messages yet'}
                                    </p>
                                </div>
                            </button>
                        )
                    })}
                </div>

                {/* Profile Avatar - Bottom Left */}
                <button
                    onClick={() => setIsProfileOpen(true)}
                    className="p-4 border-t border-[#2f3335] flex items-center gap-3 hover:bg-[#1a1a1a] transition-colors group"
                >
                    <div
                        className="w-10 h-10 rounded-full flex items-center justify-center text-white font-bold text-sm relative overflow-hidden group-hover:scale-105 transition-transform"
                        style={{
                            background: user?.profilePictureUrl
                                ? `url(${user.profilePictureUrl}) center/cover`
                                : avatarColor
                        }}
                    >
                        {!user?.profilePictureUrl && userInitials}
                    </div>
                    <div className="flex-1 text-left">
                        <p className="text-sm font-medium text-white truncate">{user?.name || 'User'}</p>
                        <p className="text-xs text-[#958d73] truncate">View profile</p>
                    </div>
                </button>
            </div>

            {/* Mobile Bottom Bar */}
            <div className="md:hidden fixed bottom-0 left-0 right-0 bg-[#080808] border-t border-[#2f3335] z-40">
                <div className="flex items-center justify-around px-2 py-3">
                    <button
                        onClick={() => {
                            setActiveTab('direct');
                            if (activeChatId) setActiveChatId(null);
                            if (onViewChange) onViewChange('chats');
                        }}
                        className={`flex flex-col items-center gap-1 px-3 py-1 rounded-lg transition-colors relative ${activeTab === 'direct' && !activeChatId ? 'text-[var(--color-primary)]' : 'text-[#958d73]'
                            }`}
                    >
                        <MessageSquare className="w-5 h-5" />
                        <span className="text-[10px] font-medium">Chats</span>
                        {hasUnreadMessages && (
                            <span className="absolute top-1 right-1 w-2 h-2 bg-[#34a853] rounded-full shadow-[0_0_8px_#34a853]"></span>
                        )}
                    </button>

                    <button
                        onClick={() => {
                            setIsFriendRequestModalOpen(true);
                        }}
                        className="flex flex-col items-center gap-1 px-3 py-1 rounded-lg text-[#958d73] hover:text-[var(--color-primary)] transition-colors relative"
                    >
                        <UserPlus className="w-5 h-5" />
                        <span className="text-[10px] font-medium">Friends</span>
                        {pendingRequestsCount > 0 && (
                            <span className="absolute top-0 right-0 w-4 h-4 bg-red-500 text-white text-[8px] font-bold rounded-full flex items-center justify-center shadow-[0_0_8px_rgba(239,68,68,0.5)]">
                                {pendingRequestsCount}
                            </span>
                        )}
                    </button>

                    <button
                        onClick={() => {
                            setIsProfileOpen(true);
                            if (onViewChange) onViewChange('profile');
                        }}
                        className="flex flex-col items-center gap-1 px-3 py-1 rounded-lg text-[#958d73] transition-colors"
                    >
                        <div
                            className="w-6 h-6 rounded-full flex items-center justify-center text-white font-bold text-[10px]"
                            style={{
                                background: user?.profilePictureUrl
                                    ? `url(${user.profilePictureUrl}) center/cover`
                                    : avatarColor
                            }}
                        >
                            {!user?.profilePictureUrl && userInitials}
                        </div>
                        <span className="text-[10px] font-medium">Profile</span>
                    </button>
                </div>
            </div>

            {/* Modals */}
            <FriendRequestModal isOpen={isFriendRequestModalOpen} onClose={() => setIsFriendRequestModalOpen(false)} />
            <CreateGroupModal isOpen={isCreateGroupModalOpen} onClose={() => setIsCreateGroupModalOpen(false)} />
            <JoinGroupModal isOpen={isJoinGroupModalOpen} onClose={() => setIsJoinGroupModalOpen(false)} />
            <ProfileView isOpen={isProfileOpen} onClose={() => setIsProfileOpen(false)} />
        </>
    );
}
