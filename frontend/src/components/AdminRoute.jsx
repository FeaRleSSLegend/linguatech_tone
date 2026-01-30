import { Navigate, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export default function AdminRoute({ children }) {
    const { user, loading, profileLoading } = useAuth();
    const location = useLocation();

    // 1. Handle Loading States
    // We check both the Auth loading and the Firestore Profile loading
    if (loading || profileLoading) {
        return (
            <div className="min-h-screen bg-[#080808] flex items-center justify-center">
                <div className="flex flex-col items-center gap-4">
                    <div className="w-8 h-8 border-4 border-[#958d73] border-t-transparent rounded-full animate-spin" />
                    <span className="text-[10px] text-[#5a5a5a] uppercase tracking-widest font-mono">
                        Verifying Authority...
                    </span>
                </div>
            </div>
        );
    }

    // 2. Role-Based Access Control
    const isAdmin = user?.role === 'admin' || user?.role === 'super_admin';

    if (!user || !isAdmin) {
        // Log the unauthorized attempt for debugging
        console.warn(`Access Denied: User ${user?.email} attempted to reach ${location.pathname}`);
        return <Navigate to="/app" state={{ from: location }} replace />;
    }

    // 3. Authorization Success
    return children;
}