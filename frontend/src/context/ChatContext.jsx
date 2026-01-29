import { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { useAuth } from './AuthContext';
import { getChats, createChat, getMessages, sendMessage } from '../services/api';
import { subscribeToChats, subscribeToMessages } from '../lib/firebase';
import { io } from 'socket.io-client';
import { useNotification } from './NotificationContext';

const ChatContext = createContext(null);

const SOCKET_URL = import.meta.env.VITE_API_URL || 'http://localhost:3001';

export const useChat = () => useContext(ChatContext);

export const ChatProvider = ({ children }) => {
    const { user } = useAuth();
    const [messages, setMessages] = useState([]);
    const [chats, setChats] = useState([]);
    const [activeChatId, setActiveChatId] = useState(null); // Null means empty state
    const [socket, setSocket] = useState(null);
    const [typingUsers, setTypingUsers] = useState({}); // { chatId: [username1, username2] }
    const [unreadCounts, setUnreadCounts] = useState({}); // { chatId: count }
    const { addNotification } = useNotification();

    // Initialize Socket
    useEffect(() => {
        if (!user) return;

        const newSocket = io(SOCKET_URL);
        setSocket(newSocket);

        newSocket.on('receive_message', (data) => {
            console.log('Socket received message:', data);

            // 1. Update Messages
            setMessages((prev) => {
                // Deduplicate by ID
                if (prev.some(m => m.id === data.id)) return prev;
                return [...prev, data];
            });

            // 2. Update Chat List (Bring to top like WhatsApp)
            setChats((prev) => {
                const chatIndex = prev.findIndex(c => c.id === data.chatId);
                if (chatIndex === -1) return prev;
                const updatedChats = [...prev];
                const chat = { ...updatedChats.splice(chatIndex, 1)[0], lastActivity: Date.now() };
                return [chat, ...updatedChats];
            });

            // 3. Handle Active Chat vs Unread/Notifications
            setActiveChatId((currentActiveId) => {
                if (data.chatId !== currentActiveId) {
                    // Increment unread count
                    setUnreadCounts((prev) => ({
                        ...prev,
                        [data.chatId]: (prev[data.chatId] || 0) + 1
                    }));

                    // Trigger notification
                    const chatName = chats.find(c => c.id === data.chatId)?.name || 'New Message';
                    addNotification({
                        type: 'message',
                        id: data.id,
                        chatId: data.chatId,
                        title: chatName,
                        message: `${data.sender.name}: ${data.text.substring(0, 50)}${data.text.length > 50 ? '...' : ''}`,
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

        return () => newSocket.close();
    }, [user, activeChatId]);

    // Join room when activeChatId changes
    useEffect(() => {
        if (socket && activeChatId && user) {
            socket.emit('join_chat', { chatId: activeChatId, userId: user.id });
        }
    }, [socket, activeChatId]);

    // Real-time chats subscription with privacy filter
    useEffect(() => {
        if (!user) return;

        console.log('📡 Subscribing to chats for user:', user.id);
        const unsubscribe = subscribeToChats(user.id, (updatedChats) => {
            console.log('✅ Chats updated:', updatedChats.length);
            setChats(updatedChats);
        });

        return () => {
            console.log('🔌 Unsubscribing from chats');
            unsubscribe();
        };
    }, [user]);

    // Real-time messages subscription with privacy filter
    useEffect(() => {
        if (!user || !activeChatId) {
            setMessages([]);
            return;
        }

        console.log('📡 Subscribing to messages for chat:', activeChatId);
        const unsubscribe = subscribeToMessages(activeChatId, user.id, (updatedMessages) => {
            console.log('✅ Messages updated:', updatedMessages.length);
            setMessages(updatedMessages);
        });

        return () => {
            console.log('🔌 Unsubscribing from messages');
            unsubscribe();
        };
    }, [user, activeChatId]);

    const sendMessageToApi = async (text, toxicityAnalysis) => {
        if (!user || !activeChatId) return;

        try {
            const sender = {
                id: user.id || 'guest',
                name: user.name || user.email || 'Guest',
                email: user.email
            };
            const newMessage = await sendMessage(activeChatId, text, sender, toxicityAnalysis);

            // Update local state (already logic in socket listener, but we append here for immediate feedback)
            setMessages(prev => {
                if (prev.some(m => m.id === newMessage.id)) return prev;
                return [...prev, newMessage];
            });

            // Broadcast via socket
            if (socket) {
                console.log('Emitting message to room:', activeChatId);
                socket.emit('send_message', newMessage);
                socket.emit('stop_typing', { chatId: activeChatId, userName: user.name });
            }
        } catch (err) {
            console.error("Failed to send message", err);
        }
    };

    // Re-naming to avoid conflict with service
    const handleSendMessage = (text, toxicityAnalysis) => {
        sendMessageToApi(text, toxicityAnalysis);
    };

    const handleTyping = useCallback((isTyping) => {
        if (socket && activeChatId && user) {
            socket.emit(isTyping ? 'typing' : 'stop_typing', {
                chatId: activeChatId,
                userName: user.name
            });
        }
    }, [socket, activeChatId, user]);

    const handleCreateChat = async (type, details = {}) => {
        let name = details.name;
        if (!name) {
            name = type === 'direct' ? `User ${Math.floor(Math.random() * 1000)}` : `Group ${Math.floor(Math.random() * 100)}`;
        }

        try {
            // Prepare participants
            const participants = type === 'direct' && details.userId
                ? [user.id, details.userId]
                : [user.id];

            const newChat = await createChat(name, type, user?.id, participants);

            setChats(prev => {
                const exists = prev.find(c => c.id === newChat.id);
                if (exists) return prev;
                return [newChat, ...prev];
            });

            setActiveChatId(newChat.id);
            return newChat;
        } catch (err) {
            console.error("Failed to create chat", err);
        }
    };

    const value = {
        messages,
        chats,
        activeChatId,
        setActiveChatId: (id) => {
            setActiveChatId(id);
            setUnreadCounts(prev => ({ ...prev, [id]: 0 }));
        },
        unreadCounts,
        sendMessage: handleSendMessage,
        createChat: handleCreateChat,
        typingUsers: typingUsers[activeChatId] || [],
        emitTyping: handleTyping,
        socket
    };

    return (
        <ChatContext.Provider value={value}>
            {children}
        </ChatContext.Provider>
    );
};
