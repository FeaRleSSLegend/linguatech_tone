import { useState, useCallback, useEffect } from 'react';
import { Routes, Route } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import ChatInterface from './components/ChatInterface';
import Sidebar from './components/Sidebar';
import LoginPage from './pages/LoginPage';
import SignupPage from './pages/SignupPage';
import ProtectedRoute from './components/ProtectedRoute';
import AdminRoute from './components/AdminRoute';
import { useChat } from './context/ChatContext';
import { useAuth } from './context/AuthContext';
import { subscribeToFriendRequests } from './lib/firebase';
import NotificationContainer from './components/NotificationToast';

import LandingPage from './pages/LandingPage';
import PrivacyPolicy from './pages/PrivacyPolicy';
import TermsOfService from './pages/TermsOfService';
import AdminReports from './pages/AdminReports';
import GroupAdminReports from './pages/GroupAdminReports';

const springTransition = {
  type: "spring",
  stiffness: 100,
  damping: 20
};

function App() {
  const [analysis, setAnalysis] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [currentView, setCurrentView] = useState('chats'); // 'chats', 'chat-room', 'profile'
  const { activeChatId, setActiveChatId } = useChat();
  const { user } = useAuth();
  const [pendingRequestsCount, setPendingRequestsCount] = useState(0);

  // Global friend request listener for notification badge
  useEffect(() => {
    if (!user) {
      console.log('🚫 [App.jsx] Global listener not started - no user');
      return;
    }

    console.log('🌐 [App.jsx] Starting global friend request listener for user:', user.id);

    const unsubscribe = subscribeToFriendRequests(user.id, (requests) => {
      console.log('🔔 [App.jsx] Global notification update:', requests.length, 'pending requests');
      setPendingRequestsCount(requests.length);
    });

    return () => {
      console.log('🔚 [App.jsx] Unsubscribing global listener');
      unsubscribe();
    };
  }, [user]);

  // Sync currentView with activeChatId
  useEffect(() => {
    if (activeChatId) {
      setCurrentView('chat-room');
    } else {
      setCurrentView('chats');
    }
  }, [activeChatId]);

  const handleAnalysisUpdate = useCallback((newAnalysis, analyzing) => {
    setAnalysis(newAnalysis);
    setIsAnalyzing(analyzing);
  }, []);

  const handleBackToChats = () => {
    setActiveChatId(null);
    setCurrentView('chats');
  };

  return (
    <>
      <Routes>
        {/* Public Routes */}
        <Route path="/" element={<LandingPage />} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/signup" element={<SignupPage />} />
        <Route path="/privacy" element={<PrivacyPolicy />} />
        <Route path="/terms" element={<TermsOfService />} />

        {/* Protected Chat Route */}
        <Route
          path="/app"
          element={
            <ProtectedRoute>
              <div className="min-h-screen bg-[#080808] flex h-screen overflow-hidden relative">
                {/* Desktop Sidebar */}
                <aside className="hidden md:block w-[300px] h-full border-r border-[#2f3335]">
                  <Sidebar onViewChange={setCurrentView} pendingRequestsCount={pendingRequestsCount} />
                </aside>

                {/* Mobile: Conditional rendering based on currentView */}
                <div className="md:hidden w-full h-full pb-16 relative overflow-hidden">
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
                        <Sidebar onViewChange={setCurrentView} pendingRequestsCount={pendingRequestsCount} />
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
                          onAnalysisUpdate={handleAnalysisUpdate}
                          onBackToChats={handleBackToChats}
                        />
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>

                {/* Desktop Chat Section */}
                <main className="hidden md:flex flex-1 flex-col h-full relative">
                  <ChatInterface onAnalysisUpdate={handleAnalysisUpdate} />
                </main>

                {/* Mobile Bottom Bar (rendered by Sidebar) */}
              </div>
            </ProtectedRoute>
          }
        />
        <Route
          path="/admin/reports"
          element={
            <AdminRoute>
              <AdminReports />
            </AdminRoute>
          }
        />
        <Route
          path="/admin/groups/:groupId"
          element={
            <AdminRoute>
              <GroupAdminReports />
            </AdminRoute>
          }
        />
      </Routes>
      <NotificationContainer />
    </>
  );
}

export default App;
