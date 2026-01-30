import { useState, useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
import { Search, LogOut, Shield, UserPlus, MessageSquare, Plus, Users } from 'lucide-react';
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
import { useSocket } from '../context/SocketContext';

function stringToColor(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
        hash = str.charCodeAt(i) + ((hash << 5) - hash);
    }
    const hue = hash % 360;
    return `hsl(${hue}, 65%, 50%)`;
}

/**
 * Get display name for a chat (Shows OTHER person's name in direct chats)
 */
function getDisplayName(chat, currentUserId, currentUserName) {
    if (chat.type === 'group' || chat.isGroup) {
        return chat.name;
    }

    if (chat.participants && chat.participantNames) {
        const otherUserId = chat.participants.find(id => id !== currentUserId);
        if (otherUserId && chat.participantNames[otherUserId]) {
            return chat.participantNames[otherUserId];
        }
    }

    if (chat.name && chat.name.includes(' & ')) {
        const names = chat.name.split(' & ');
        const otherName = names.find(n => n.trim() !== currentUserName);
        return otherName || names[0];
    }

    return chat.name || 'Unknown';
}

function getDisplayInitials(chat, currentUserId, currentUserName) {
    const displayName = getDisplayName(chat, currentUserId, currentUserName);
    return displayName.substring(0, 2).toUpperCase();
}

function formatMessageTime(timestamp) {
    if (!timestamp) return '';
    
    let msgDate = timestamp?.toDate ? timestamp.toDate() : new Date(timestamp);
    const now = new Date();

    // Create "start of today" to compare properly
    const today = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
    const msgTime = msgDate.getTime();

    if (msgTime >= today) {
        return msgDate.toLocaleTimeString([], { 
            hour: '2-digit', 
            minute: '2-digit',
            hour12: true 
        });
    }

    const yesterday = today - 86400000; // 24 hours in ms
    if (msgTime >= yesterday) {
        return 'Yesterday';
    }

    return msgDate.toLocaleDateString([], { month: 'short', day: 'numeric' });
}

function ChatListSkeleton() {
    return (
        <div className="px-2 space-y-1 animate-pulse">
            {[1, 2, 3, 4, 5].map((i) => (
                <div key={i} className="flex items-center gap-3 p-2.5 rounded-lg bg-[#1a1a1a]">
                    <div className="w-10 h-10 rounded-full bg-[#2f3335]" />
                    <div className="flex-1 space-y-2">
                        <div className="h-3 bg-[#2f3335] rounded w-3/4" />
                        <div className="h-2 bg-[#2f3335] rounded w-1/2" />
                    </div>
                </div>
            ))}
        </div>
    );
}

export default function Sidebar({ isCollapsed, onToggle, onViewChange, pendingRequestsCount = 0 }) {
    const { chats, activeChatId, setActiveChatId, unreadCounts } = useChat();
    const { logout, user } = useAuth();    
    const [isFriendRequestModalOpen, setIsFriendRequestModalOpen] = useState(false);
    const [isProfileOpen, setIsProfileOpen] = useState(false);
    const [searchQuery, setSearchQuery] = useState('');
    const [activeTab, setActiveTab] = useState('direct');
    const [onlineStatuses, setOnlineStatuses] = useState({});
    const [isNotificationDropdownOpen, setIsNotificationDropdownOpen] = useState(false);
    const [friendRequests, setFriendRequests] = useState([]);
    const [isCreateGroupModalOpen, setIsCreateGroupModalOpen] = useState(false);
    const [isJoinGroupModalOpen, setIsJoinGroupModalOpen] = useState(false);
    const [chatsLoading, setChatsLoading] = useState(true);
    const bellRef = useRef(null);

    const groupCount = chats.filter(c => (c.type || '').toLowerCase() === 'group' || c.isGroup).length;
    // ADD THIS LINE HERE
    const { onlineUsers } = useSocket();

    // This state doesn't need to do anything other than change to trigger a refresh
    const [, setTick] = useState(0);

    useEffect(() => {
        // Set up a timer to run every 60 seconds (60000 ms)
        const timer = setInterval(() => {
            setTick(t => t + 1);
        }, 60000);

        return () => clearInterval(timer); // Clean up on unmount
    }, []);

    useEffect(() => {
        const timer = setTimeout(() => setChatsLoading(false), 2000);
        if (chats.length > 0) setChatsLoading(false);
        return () => clearTimeout(timer);
    }, [chats]);

    

    useEffect(() => {
        if (!user) return;
        const unsubscribe = subscribeToFriendRequests(user.id, (requests) => {
            setFriendRequests(requests);
        });
        return () => unsubscribe();
    }, [user]);

    const handleAcceptRequest = async (requestId, fromUserId) => {
        try {
            await acceptFriendRequest(requestId, fromUserId, user.id);
        } catch (error) {
            console.error('Accept error:', error);
        }
    };

    const handleRejectRequest = async (requestId) => {
        try {
            await rejectFriendRequest(requestId);
        } catch (error) {
            console.error('Reject error:', error);
        }
    };

    /**
     * 🔥 FIX #3: Read lastMessage directly from chat object (not from messages array)
     * This ensures ALL chats show previews, not just the active one
     */
    const getLastMessagePreview = (chat) => {
        if (chat.lastMessage) {
            return {
                text: chat.lastMessage,
                // Keep it as the raw Firestore object so formatMessageTime can use .toDate()
                timestamp: chat.lastActivity 
            };
        }
        return null;
    };

    /**
     * 🔥 FIX #5: Sort by lastActivity (most recent first)
     * Case-insensitive type matching
     */
    // Inside Sidebar.jsx, update the filteredChats constant:
    const filteredChats = chats.filter(chat => {
        const chatType = (chat.type || '').toLowerCase();
        const tabType = activeTab.toLowerCase();
        
        if (chatType !== tabType) return false;

        // 🔥 NEW: Filter out chats where the other participant is BLOCKED
        const otherUserId = chat.participants?.find(id => id !== user?.id);
        if (user?.blockedUsers?.includes(otherUserId)) return false; // Hide blocked
        
        const displayName = getDisplayName(chat, user?.id, user?.name);
        if (!displayName.toLowerCase().includes(searchQuery.toLowerCase())) return false;
        
        if ((chat.type === 'group' || chat.isGroup) && chat.permBannedUsers?.includes(user?.id)) return false;
        
        return true;
    }).sort((a, b) => {
        // Standardize sorting by milliseconds
        const timeA = a.lastActivity?.toMillis?.() || new Date(a.lastActivity).getTime() || 0;
        const timeB = b.lastActivity?.toMillis?.() || new Date(b.lastActivity).getTime() || 0;
        return timeB - timeA;
    });

    const avatarColor = user?.id ? stringToColor(user.id) : '#958d73';
    const userInitials = user?.name ? user.name.substring(0, 2).toUpperCase() : 'U';

    return (
        <>
            {/* Desktop Sidebar - Fixed width to prevent collapse */}
            <div className="hidden md:flex w-[320px] lg:w-[380px] h-full flex-col bg-[#080808] shrink-0 border-r border-[#2f3335] relative z-30">
                {/* 🔥 FIX #2: Improved Header with stable notification bell */}
                <div className="h-16 px-4 flex items-center justify-between border-b border-[#2f3335] shrink-0 bg-[#0a0a0a]">
                    <div className="flex items-center gap-3">
                        <Logo className="w-8 h-8" showText={true} />
                    </div>

                    <div className="flex items-center gap-2">
                        {/* 🔥 FIX #2: Notification bell with proper positioning */}
                        <div className="relative">
                            <NotificationBell
                                ref={bellRef}
                                count={friendRequests.length}
                                isActive={isNotificationDropdownOpen}
                                onClick={() => setIsNotificationDropdownOpen(!isNotificationDropdownOpen)}
                            />
                            {/* Dropdown uses absolute positioning to float above UI */}
                            <NotificationDropdown
                                isOpen={isNotificationDropdownOpen}
                                onClose={() => setIsNotificationDropdownOpen(false)}
                                requests={friendRequests}
                                onAccept={handleAcceptRequest}
                                onReject={handleRejectRequest}
                                anchorRef={bellRef}
                            />
                        </div>

                        <button 
                            onClick={() => setIsFriendRequestModalOpen(true)} 
                            className="text-[#958d73] hover:text-[var(--color-primary)] p-2 hover:bg-[var(--color-primary)]/10 rounded-lg transition-colors"
                            title="Send friend request"
                        >
                            <UserPlus className="w-5 h-5" />
                        </button>

                        {(user?.role === 'admin' || user?.role === 'super_admin') && (
                            <Link 
                                to="/admin/reports" 
                                className="bg-red-500/10 hover:bg-red-500/20 text-red-500 p-2 rounded-lg transition-all"
                                title="Admin Reports"
                            >
                                <Shield className="w-5 h-5" />
                            </Link>
                        )}

                        <button 
                            onClick={logout} 
                            className="text-[#958d73] hover:text-red-400 p-2 hover:bg-red-500/10 rounded-lg transition-colors"
                            title="Logout"
                        >
                            <LogOut className="w-5 h-5" />
                        </button>
                    </div>
                </div>

                {/* Search */}
                <div className="px-4 pt-4 pb-2 shrink-0">
                    <div className="relative group">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[#958d73]" />
                        <input
                            type="text"
                            placeholder="Search chats..."
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            className="w-full bg-[#111111] border border-transparent focus:border-[#2f3335] rounded-xl py-2.5 pl-9 pr-4 text-sm text-[#e0ddd9] focus:outline-none transition-all placeholder-[#5a5a5a]"
                        />
                    </div>
                </div>

                {/* Tabs */}
                <div className="px-4 flex items-center gap-1 mb-2 shrink-0">
                    {['direct', 'group'].map((tab) => {
                        const count = chats.filter(c => (c.type || '').toLowerCase() === tab.toLowerCase()).reduce((sum, c) => sum + (unreadCounts[c.id] || 0), 0);
                        return (
                            <button
                                key={tab}
                                onClick={() => setActiveTab(tab)}
                                className={`text-xs font-semibold py-2 px-4 rounded-xl transition-colors flex-1 relative ${activeTab === tab ? 'bg-[#1e1e1e] text-white' : 'text-[#958d73] hover:text-[#e0ddd9]'}`}
                            >
                                {tab.charAt(0).toUpperCase() + tab.slice(1)} {tab === 'group' ? `(${groupCount})` : ''}
                                {count > 0 && (
                                    <span className="ml-2 bg-[#00a884] text-black text-[10px] font-bold rounded-full px-1.5 py-0.5 min-w-[18px] inline-block shadow-[0_0_8px_rgba(0,168,132,0.3)]">
                                        {count > 9 ? '9+' : count}
                                    </span>
                                )}
                            </button>
                        );
                    })}
                </div>

                {/* 🔥 FIX #1: Group Actions with BOTH Create and Join buttons */}
                {activeTab === 'group' && (
                    <div className="px-4 mb-2 space-y-2 shrink-0">
                        <button 
                            onClick={() => setIsCreateGroupModalOpen(true)} 
                            className="w-full py-2.5 px-4 bg-[var(--color-primary)] hover:bg-[#2d9248] text-black font-bold rounded-xl transition-all flex items-center justify-center gap-2"
                        >
                            <Plus className="w-4 h-4" /> Create Group
                        </button>
                        <button 
                            onClick={() => setIsJoinGroupModalOpen(true)} 
                            className="w-full py-2.5 px-4 bg-[#1a1a1a] hover:bg-[#2a2a2a] border border-[#2f3335] hover:border-[var(--color-primary)] text-[#e0ddd9] font-bold rounded-xl transition-all flex items-center justify-center gap-2"
                        >
                            <Users className="w-4 h-4" /> Join Group
                        </button>
                    </div>
                )}

                {/* Chat List */}
                <div className="flex-1 overflow-y-auto px-2 space-y-1 scrollbar-thin scrollbar-thumb-[#2f3335] scrollbar-track-transparent">
                    {chatsLoading ? (
                        <ChatListSkeleton />
                    ) : filteredChats.length === 0 ? (
                        <div className="text-center py-12">
                            <MessageSquare className="w-12 h-12 text-[#2f3335] mx-auto mb-2 opacity-50" />
                            <p className="text-[#958d73] text-xs">
                                {activeTab === 'group' ? 'No groups yet. Create or join one!' : 'No conversations found'}
                            </p>
                        </div>
                    ) : (
                        filteredChats.map(chat => {
                            const lastMsgData = getLastMessagePreview(chat);
                            const isOnline = chat.type === 'direct' && chat.participants?.some(id => id !== user?.id && onlineUsers.includes(id));                            const displayName = getDisplayName(chat, user?.id, user?.name);
                            const displayInitials = getDisplayInitials(chat, user?.id, user?.name);
                            const unreadCount = unreadCounts[chat.id] || 0;
                            const hasUnread = unreadCount > 0;

                            // Check if EITHER user has blocked (hides online status and shows blocked message)
                            const otherUserId = chat.participants?.find(id => id !== user?.id);
                            const iBlockedThem = chat.blockedBy?.includes(user?.id) || false;
                            const theyBlockedMe = chat.blockedBy?.includes(otherUserId) || false;
                            const isBlocked = chat.type === 'direct' && (iBlockedThem || theyBlockedMe);

                            return (
                                <button
                                    key={chat.id}
                                    onClick={() => setActiveChatId(chat.id)}
                                    className={`w-full flex items-center gap-3 p-3 rounded-xl transition-all text-left group ${
                                        activeChatId === chat.id 
                                            ? 'bg-[#2f3335] ring-1 ring-[var(--color-primary)]/30' 
                                            : 'hover:bg-[#151515]'
                                    }`}
                                >
                                    <div className="relative shrink-0">
                                        <div className="w-12 h-12 rounded-full bg-[#1e1e1e] flex items-center justify-center text-[#958d73] border border-[#2f3335] group-hover:border-[var(--color-primary)]/40 transition-all">
                                            {/* This will now only show if the ID is in the onlineUsers array from Socket.io */}
                                            {chat.type === 'direct' && isOnline && !isBlocked && (
                                                <span className="absolute bottom-0.5 right-0.5 w-3 h-3 bg-[#34a853] border-2 border-[#080808] rounded-full z-10 shadow-[0_0_6px_rgba(52,168,83,0.6)]"></span>
                                            )}
                                            <span className="text-sm font-bold">{displayInitials}</span>
                                        </div>
                                    </div>

                                    <div className="flex-1 min-w-0">
                                        <div className="flex items-center justify-between mb-0.5">
                                            <h3 className={`text-sm truncate ${hasUnread ? 'font-black text-white' : 'font-medium text-[#e0ddd9]'}`}>
                                                {displayName}
                                            </h3>
                                            <span className={`text-[10px] shrink-0 ml-2 ${hasUnread ? 'text-[#00a884] font-bold' : 'text-[#5a5a5a]'}`}>
                                                {lastMsgData ? formatMessageTime(lastMsgData.timestamp) : ''}
                                            </span>
                                        </div>
                                        <div className="flex items-center justify-between gap-2">
                                            <p className={`text-xs truncate ${hasUnread ? 'font-bold text-[#e0ddd9]' : 'text-[#958d73]'}`}>
                                                {isBlocked ? (iBlockedThem ? 'You blocked this contact' : 'You were blocked by this contact') : (lastMsgData ? lastMsgData.text : 'No messages yet')}
                                            </p>
                                            {/* 🔥 FIX #4: WhatsApp-style unread badge */}
                                            {hasUnread && !isBlocked && (
                                                <span className="shrink-0 min-w-[20px] h-[20px] bg-[#00a884] text-black text-[11px] font-black rounded-full flex items-center justify-center px-1.5 shadow-[0_0_10px_rgba(0,168,132,0.5)]">
                                                    {unreadCount > 9 ? '9+' : unreadCount}
                                                </span>
                                            )}
                                        </div>
                                    </div>
                                </button>
                            );
                        })
                    )}
                </div>

                {/* Profile Avatar */}
                <button 
                    onClick={() => setIsProfileOpen(true)} 
                    className="h-16 px-4 border-t border-[#2f3335] flex items-center gap-3 hover:bg-[#1a1a1a] transition-colors group shrink-0"
                >
                    <div 
                        className="w-10 h-10 rounded-full flex items-center justify-center text-white font-bold text-sm overflow-hidden border-2 border-[#2f3335]"
                        style={user?.profilePictureUrl ? { 
                            backgroundImage: `url(${user.profilePictureUrl})`, 
                            backgroundSize: 'cover',
                            backgroundPosition: 'center'
                        } : { 
                            backgroundColor: avatarColor 
                        }}
                    >
                        {!user?.profilePictureUrl && userInitials}
                    </div>
                    <div className="flex-1 text-left min-w-0">
                        <p className="text-sm font-bold text-white truncate">{user?.name || 'User'}</p>
                        <p className="text-[10px] text-[#958d73]">Settings & Profile</p>
                    </div>
                </button>
            </div>

            {/* Mobile Bottom Bar */}
            <div className="md:hidden fixed bottom-0 left-0 right-0 h-16 bg-[#080808] border-t border-[#2f3335] z-50">
                <div className="flex items-center justify-around h-full px-2">
                    <button 
                        onClick={() => { 
                            setActiveTab('direct'); 
                            setActiveChatId(null); 
                            onViewChange?.('chats'); 
                        }} 
                        className={`flex flex-col items-center gap-1 ${activeTab === 'direct' && !activeChatId ? 'text-[var(--color-primary)]' : 'text-[#958d73]'}`}
                    >
                        <div className="relative">
                            <MessageSquare className="w-6 h-6" />
                            {Object.values(unreadCounts).some(c => c > 0) && (
                                <span className="absolute -top-1 -right-1 w-2.5 h-2.5 bg-[#00a884] rounded-full border-2 border-[#080808]"></span>
                            )}
                        </div>
                        <span className="text-[10px] font-bold">Chats</span>
                    </button>
                    <button 
                        onClick={() => setIsFriendRequestModalOpen(true)} 
                        className="flex flex-col items-center gap-1 text-[#958d73]"
                    >
                        <div className="relative">
                            <UserPlus className="w-6 h-6" />
                            {friendRequests.length > 0 && (
                                <span className="absolute -top-1 -right-1 bg-red-500 text-white text-[8px] rounded-full w-4 h-4 flex items-center justify-center font-bold">
                                    {friendRequests.length > 9 ? '9+' : friendRequests.length}
                                </span>
                            )}
                        </div>
                        <span className="text-[10px] font-bold">Friends</span>
                    </button>
                    <button 
                        onClick={() => setIsProfileOpen(true)} 
                        className="flex flex-col items-center gap-1 text-[#958d73]"
                    >
                        <div className="w-6 h-6 rounded-full bg-[#2f3335] flex items-center justify-center text-white text-[10px] font-bold">
                            {userInitials}
                        </div>
                        <span className="text-[10px] font-bold">Profile</span>
                    </button>
                </div>
            </div>

            {/* Modals */}
            <FriendRequestModal 
                isOpen={isFriendRequestModalOpen} 
                onClose={() => setIsFriendRequestModalOpen(false)} 
            />
            <CreateGroupModal 
                isOpen={isCreateGroupModalOpen} 
                onClose={() => setIsCreateGroupModalOpen(false)} 
            />
            <JoinGroupModal 
                isOpen={isJoinGroupModalOpen} 
                onClose={() => setIsJoinGroupModalOpen(false)} 
            />
            <ProfileView 
                isOpen={isProfileOpen} 
                onClose={() => setIsProfileOpen(false)} 
            />
        </>
    );
}