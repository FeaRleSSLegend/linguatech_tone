import { createContext, useContext, useState, useEffect } from 'react';
import { auth, db } from '../lib/firebase'; // Ensure these are exported from your firebase config
import { onAuthStateChanged } from 'firebase/auth';
import { doc, getDoc } from 'firebase/firestore';
import { loginUser, registerUser, logoutUser } from '../services/api';

const AuthContext = createContext(null);

export const useAuth = () => useContext(AuthContext);

export const AuthProvider = ({ children }) => {
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);

    // This is the "Magic" part: Real-time Auth Sync
    useEffect(() => {
        const unsubscribe = onAuthStateChanged(auth, async (firebaseUser) => {
            try {
                if (firebaseUser) {
                    // 1. Get the latest profile from Firestore (where the role lives)
                    const userDoc = await getDoc(doc(db, 'users', firebaseUser.uid));
                    const profileData = userDoc.exists() ? userDoc.data() : {};

                    const fullUser = {
                        id: firebaseUser.uid,
                        email: firebaseUser.email,
                        ...profileData
                    };

                    console.log('👤 [AuthContext] Authority Verified:', fullUser.role);
                    setUser(fullUser);
                    localStorage.setItem('tone_user', JSON.stringify(fullUser));
                } else {
                    setUser(null);
                    localStorage.removeItem('tone_user');
                }
            } catch (error) {
                console.error("❌ [AuthContext] Sync Error:", error);
                setUser(null);
            } finally {
                setLoading(false);
            }
        });

        return () => unsubscribe(); // Cleanup listener on unmount
    }, []);

    const login = async (email, password) => {
        // api.js handles the Firebase sign-in logic
        const data = await loginUser(email, password);
        // Note: setUser is handled by onAuthStateChanged listener above
        return data.user;
    };

    const signup = async (name, email, password, adminSecret) => {
        const data = await registerUser(name, email, password, adminSecret);
        return data.user;
    };

    const logout = async () => {
        try {
            await logoutUser();
            setUser(null);
            localStorage.removeItem('tone_user');
        } catch (error) {
            console.error('Logout error:', error);
        }
    };

    const value = {
        user,
        login,
        signup,
        logout,
        loading,
        isAdmin: user?.role === 'admin' || user?.role === 'super_admin'
    };

    return (
        <AuthContext.Provider value={value}>
            {!loading && children}
        </AuthContext.Provider>
    );
};