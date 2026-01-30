import { motion, AnimatePresence } from 'framer-motion';
import { useState, useEffect } from 'react';
import { useChat } from '../context/ChatContext';
import Sidebar from './Sidebar';
import ChatInterface from './ChatInterface';

const springTransition = {
    type: "spring",
    stiffness: 100,
    damping: 20
};

/**
 * MobileChatLayout Component
 * Handles mobile-specific navigation between chat list and chat interface
 * Uses a slide animation pattern similar to WhatsApp/Telegram
 */
export default function MobileChatLayout({ onAnalysisUpdate, pendingRequestsCount }) {
    const { activeChatId, setActiveChatId } = useChat();
    const [currentView, setCurrentView] = useState('chats'); // 'chats' or 'chat-room'

    // Sync currentView with activeChatId
    useEffect(() => {
        if (activeChatId) {
            setCurrentView('chat-room');
        } else {
            setCurrentView('chats');
        }
    }, [activeChatId]);

    const handleBackToChats = () => {
        setActiveChatId(null);
        setCurrentView('chats');
    };

    return (
        <div className="w-full h-full pb-16 relative overflow-hidden bg-[#080808]">
            <AnimatePresence mode="wait" initial={false}>
                {currentView === 'chats' && (
                    <motion.div
                        key="chats"
                        initial={{ x: -300, opacity: 0 }}
                        animate={{ x: 0, opacity: 1 }}
                        exit={{ x: -300, opacity: 0 }}
                        transition={springTransition}
                        className="absolute inset-0 pb-16"
                    >
                        <Sidebar 
                            onViewChange={setCurrentView} 
                            pendingRequestsCount={pendingRequestsCount} 
                        />
                    </motion.div>
                )}

                {currentView === 'chat-room' && (
                    <motion.div
                        key="chat-room"
                        initial={{ x: 300, opacity: 0 }}
                        animate={{ x: 0, opacity: 1 }}
                        exit={{ x: 300, opacity: 0 }}
                        transition={springTransition}
                        className="absolute inset-0"
                    >
                        <ChatInterface
                            onAnalysisUpdate={onAnalysisUpdate}
                            onBackToChats={handleBackToChats}
                        />
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}