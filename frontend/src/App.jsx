import { useState, useCallback, useEffect } from 'react';
import { Routes, Route } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import ChatInterface from './components/ChatInterface';
import Sidebar from './components/Sidebar';
import MobileChatLayout from './components/MobileChatLayout';
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

  const handleAnalysisUpdate = useCallback((newAnalysis, analyzing) => {
    setAnalysis(newAnalysis);
    setIsAnalyzing(analyzing);
  }, []);

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
                {/* Desktop Layout */}
                <aside className="hidden md:block h-full border-r border-[#2f3335] shrink-0 overflow-hidden">
                  <Sidebar pendingRequestsCount={pendingRequestsCount} />
                </aside>

                {/* Ensure the main chat takes the rest of the space and handles its own overflow */}
                <main className="hidden md:flex flex-1 flex-col h-full relative overflow-hidden min-w-0">
                  <ChatInterface onAnalysisUpdate={handleAnalysisUpdate} />
                </main>

                {/* Mobile Layout - Fixed blank screen issue */}
                <div className="md:hidden w-full h-full">
                  <MobileChatLayout 
                    onAnalysisUpdate={handleAnalysisUpdate}
                    pendingRequestsCount={pendingRequestsCount}
                  />
                </div>
              </div>
            </ProtectedRoute>
          }
        />

        {/* Admin Routes */}
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