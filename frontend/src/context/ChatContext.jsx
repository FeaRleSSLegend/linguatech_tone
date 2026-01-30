import { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { useAuth } from './AuthContext';
import { subscribeToChats, subscribeToMessages } from '../lib/firebase';
import { io } from 'socket.io-client';
import { useNotification } from './NotificationContext';
import { doc, deleteDoc, updateDoc, collection, query, where, getDocs, writeBatch, arrayUnion, arrayRemove } from 'firebase/firestore';
import { db } from "../lib/firebase";
const ChatContext = createContext(null);

const SOCKET_URL = import.meta.env.VITE_API_URL || 'http://localhost:3001';

export const useChat = () => useContext(ChatContext);

export const ChatProvider = ({ children }) => {
    const { user } = useAuth();
    const [messages, setMessages] = useState([]);
    const [chats, setChats] = useState([]);
    const [activeChatId, setActiveChatId] = useState(null);
    const [socket, setSocket] = useState(null);
    const [typingUsers, setTypingUsers] = useState({});
    const [unreadCounts, setUnreadCounts] = useState({});
    const { addNotification } = useNotification();

    /**
     * Reset conversation: Delete all messages but keep chat document for searchability
     */
    const deleteChat = async (chatId) => {
        // 1. Validate user is authenticated
        if (!user || !user.id) {
            console.error("❌ [ChatContext] User not authenticated");
            throw new Error("You must be logged in to reset a chat");
        }

        // 2. Validate user is a participant in this chat
        const chat = chats.find(c => c.id === chatId);
        if (!chat) {
            console.error("❌ [ChatContext] Chat not found");
            throw new Error("Chat not found");
        }

        if (!chat.participants?.includes(user.id)) {
            console.error("❌ [ChatContext] User not a participant");
            throw new Error("You are not authorized to reset this chat");
        }

        try {
            console.log(`🗑️ [ChatContext] Resetting chat ${chatId} for user ${user.id}`);

            // 3. Delete all messages in the messages sub-collection
            const messagesQuery = query(collection(db, 'messages'), where('chatId', '==', chatId));
            const messagesSnapshot = await getDocs(messagesQuery);

            const batch = writeBatch(db);
            messagesSnapshot.docs.forEach((messageDoc) => {
                batch.delete(messageDoc.ref);
            });
            await batch.commit();

            // 4. Clear lastMessage and updatedAt to make chat disappear from sidebar
            const chatRef = doc(db, 'chats', chatId);
            await updateDoc(chatRef, {
                lastMessage: '',
                lastMessageSenderId: '',
                updatedAt: null
            });

            // 5. If it was the active chat, clear it
            if (activeChatId === chatId) {
                setActiveChatId(null);
            }

            console.log(`✅ [ChatContext] Chat ${chatId} reset successfully (${messagesSnapshot.docs.length} messages deleted)`);
        } catch (error) {
            console.error("❌ [ChatContext] Error resetting chat:", error);
            throw error;
        }
    };

    /**
     * Block a user in a direct chat
     */
    const blockUser = async (chatId, otherUserId) => {
        if (!user || !user.id) {
            throw new Error("You must be logged in to block a user");
        }

        try {
            console.log(`🚫 [ChatContext] Blocking user ${otherUserId} in chat ${chatId}`);

            // Update the chat document to add current user to blockedBy array
            const chatRef = doc(db, 'chats', chatId);
            await updateDoc(chatRef, {
                blockedBy: arrayUnion(user.id)
            });

            console.log(`✅ [ChatContext] User ${otherUserId} blocked successfully`);
        } catch (error) {
            console.error("❌ [ChatContext] Error blocking user:", error);
            throw error;
        }
    };

    /**
     * Unblock a user in a direct chat
     */
    const unblockUser = async (chatId, otherUserId) => {
        if (!user || !user.id) {
            throw new Error("You must be logged in to unblock a user");
        }

        try {
            console.log(`✅ [ChatContext] Unblocking user ${otherUserId} in chat ${chatId}`);

            // Update the chat document to remove current user from blockedBy array
            const chatRef = doc(db, 'chats', chatId);
            await updateDoc(chatRef, {
                blockedBy: arrayRemove(user.id)
            });

            console.log(`✅ [ChatContext] User ${otherUserId} unblocked successfully`);
        } catch (error) {
            console.error("❌ [ChatContext] Error unblocking user:", error);
            throw error;
        }
    };

    // Initialize Socket
    useEffect(() => {
        if (!user) return;

        console.log('🔌 [Socket] Connecting to:', SOCKET_URL);
        const newSocket = io(SOCKET_URL);
        setSocket(newSocket);

        newSocket.on('connect', () => {
            console.log('✅ [Socket] Connected:', newSocket.id);
        });

        newSocket.on('receive_message', (data) => {
            console.log('📨 [Socket] Received message:', data);

            // Update Messages
            setMessages((prev) => {
                if (prev.some(m => m.id === data.id)) return prev;
                return [...prev, data];
            });

            // Handle Unread Counts & Notifications
            setActiveChatId((currentActiveId) => {
                if (data.chatId !== currentActiveId && data.sender.id !== user.id) {
                    setUnreadCounts((prev) => ({
                        ...prev,
                        [data.chatId]: (prev[data.chatId] || 0) + 1
                    }));

                    setChats((currentChats) => {
                        const chat = currentChats.find(c => c.id === data.chatId);
                        const chatName = chat?.name || 'New Message';
                        
                        addNotification({
                            type: 'message',
                            id: data.id,
                            chatId: data.chatId,
                            title: chatName,
                            message: `${data.sender.name}: ${data.text.substring(0, 50)}${data.text.length > 50 ? '...' : ''}`,
                        });
                        
                        return currentChats;
                    });
                }
                return currentActiveId;
            });
        });

        newSocket.on('user_typing', (data) => {
            setTypingUsers((prev) => {
                const users = prev[data.chatId] || [];
                if (!users.includes(data.userName)) {
                    return { ...prev, [data.chatId]: [...users, data.userName] };
                }
                return prev;
            });
        });

        newSocket.on('user_stop_typing', (data) => {
            setTypingUsers((prev) => {
                const users = prev[data.chatId] || [];
                return { ...prev, [data.chatId]: users.filter(u => u !== data.userName) };
            });
        });

        return () => {
            console.log('🔌 [Socket] Cleaning up');
            newSocket.close();
        };
    }, [user, addNotification]);

    // Join room when activeChatId changes
    useEffect(() => {
        if (socket && activeChatId && user) {
            socket.emit('join_chat', { chatId: activeChatId, userId: user.id });
        }
    }, [socket, activeChatId, user]);

    /**
     * 🔥 CRITICAL FIX: Subscribe to chats using EXACT Firestore index
     * Index: participants (array-contains) + updatedAt (desc)
     */
    useEffect(() => {
        if (!user) {
            console.log('❌ [Chats] No user, skipping subscription');
            return;
        }

        console.log('📡 [Chats] Subscribing for user:', user.id);
        
        const unsubscribe = subscribeToChats(user.id, (updatedChats) => {
            console.log('✅ [Chats] Received:', {
                count: updatedChats.length,
                samples: updatedChats.slice(0, 3).map(c => ({
                    id: c.id,
                    name: c.name,
                    type: c.type,
                    lastMessage: c.lastMessage || 'No preview'
                }))
            });
            
            // Chats already sorted by updatedAt DESC from Firestore
            setChats(updatedChats);
        });

        return () => {
            console.log('🔌 [Chats] Unsubscribing');
            unsubscribe();
        };
    }, [user]);

    /**
     * Subscribe to messages for active chat
     */
    useEffect(() => {
        if (!user || !activeChatId) {
            setMessages([]);
            return;
        }

        console.log('📡 [Messages] Subscribing for chat:', activeChatId);
        
        const unsubscribe = subscribeToMessages(activeChatId, user.id, (updatedMessages) => {
            console.log('✅ [Messages] Received:', updatedMessages.length);
            setMessages(updatedMessages);
        });

        return () => {
            console.log('🔌 [Messages] Unsubscribing');
            unsubscribe();
        };
    }, [user, activeChatId]);

    const handleSendMessage = async (text, toxicityAnalysis) => {
        if (!user || !activeChatId) return;

        // HARD RESTRICTION: Check if chat is blocked by EITHER user
        const activeChat = chats.find(c => c.id === activeChatId);
        if (activeChat?.blockedBy && activeChat.blockedBy.length > 0) {
            const otherUserId = activeChat.participants?.find(id => id !== user.id);
            const iBlockedThem = activeChat.blockedBy.includes(user.id);
            const theyBlockedMe = activeChat.blockedBy.includes(otherUserId);

            if (iBlockedThem || theyBlockedMe) {
                console.warn("⚠️ [Send] Cannot send message - chat has active block", {
                    iBlockedThem,
                    theyBlockedMe,
                    blockedBy: activeChat.blockedBy
                });
                throw new Error(iBlockedThem ? "You have blocked this contact" : "You were blocked by this contact");
            }
        }

        try {
            const { sendMessage } = await import('../services/api');

            const sender = {
                id: user.id,
                name: user.name || user.email || 'Guest',
                email: user.email
            };

            console.log('📤 [Send] Sending message');
            const newMessage = await sendMessage(activeChatId, text, sender, toxicityAnalysis);

            // Update local state
            setMessages(prev => {
                if (prev.some(m => m.id === newMessage.id)) return prev;
                return [...prev, newMessage];
            });

            // Broadcast via socket
            if (socket && socket.connected) {
                socket.emit('send_message', newMessage);
                socket.emit('stop_typing', { chatId: activeChatId, userName: user.name });
            }
        } catch (err) {
            console.error("❌ [Send] Failed:", err);
            throw err;
        }
    };

    const handleTyping = useCallback((isTyping) => {
        if (socket && socket.connected && activeChatId && user) {
            // Don't emit typing if chat is blocked
            const activeChat = chats.find(c => c.id === activeChatId);
            if (activeChat?.blockedBy && activeChat.blockedBy.length > 0) {
                console.log('🚫 [Typing] Blocked - not emitting typing indicator');
                return;
            }

            socket.emit(isTyping ? 'typing' : 'stop_typing', {
                chatId: activeChatId,
                userName: user.name
            });
        }
    }, [socket, activeChatId, user, chats]);

    const handleSetActiveChatId = useCallback((id) => {
        console.log('🎯 [Chat] Setting active:', id);
        setActiveChatId(id);
        
        // Clear unread count
        if (id) {
            setUnreadCounts(prev => {
                if (prev[id] > 0) {
                    console.log('✅ [Unread] Cleared for chat:', id);
                    return { ...prev, [id]: 0 };
                }
                return prev;
            });
        }
    }, []);

    const value = {
        messages,
        chats,
        activeChatId,
        setActiveChatId: handleSetActiveChatId,
        unreadCounts,
        sendMessage: handleSendMessage,
        typingUsers: typingUsers[activeChatId] || [],
        emitTyping: handleTyping,
        deleteChat,
        blockUser,
        unblockUser,
        socket
    };

    return (
        <ChatContext.Provider value={value}>
            {children}
        </ChatContext.Provider>
    );
};