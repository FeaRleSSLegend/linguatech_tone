import { createContext, useContext, useState, useEffect } from 'react';
import { loginUser, registerUser } from '../services/api';

const AuthContext = createContext(null);

export const useAuth = () => useContext(AuthContext);

export const AuthProvider = ({ children }) => {
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        // Check localStorage for existing session with safety
        try {
            const storedUser = localStorage.getItem('tone_user');
            if (storedUser) {
                const parsedUser = JSON.parse(storedUser);
                console.log('👤 [AuthContext] Session restored:', {
                    userId: parsedUser.id,
                    email: parsedUser.email,
                    name: parsedUser.name
                });
                setUser(parsedUser);
            } else {
                console.log('🔓 [AuthContext] No stored session found');
            }
        } catch (e) {
            console.error("❌ [AuthContext] Session recovery failed", e);
            localStorage.removeItem('tone_user'); // Clear corrupt data
        } finally {
            setLoading(false);
        }
    }, []);

    const login = async (email, password) => {
        try {
            const data = await loginUser(email, password);
            console.log('✅ [AuthContext] Login successful:', {
                userId: data.user.id,
                email: data.user.email,
                name: data.user.name
            });
            setUser(data.user);
            localStorage.setItem('tone_user', JSON.stringify(data.user));
            return data.user;
        } catch (error) {
            console.error('❌ [AuthContext] Login error:', error);
            throw error;
        }
    };

    const signup = async (name, email, password, adminSecret) => {
        try {
            const data = await registerUser(name, email, password, adminSecret);
            console.log('✅ [AuthContext] Signup successful:', {
                userId: data.user.id,
                email: data.user.email,
                name: data.user.name
            });
            setUser(data.user);
            localStorage.setItem('tone_user', JSON.stringify(data.user));
            return data.user;
        } catch (error) {
            console.error('❌ [AuthContext] Signup error:', error);
            throw error;
        }
    };

    const logout = () => {
        console.log('🚪 [AuthContext] User logging out');
        setUser(null);
        localStorage.removeItem('tone_user');
    };

    const updateUser = (updatedUser) => {
        setUser(updatedUser);
        localStorage.setItem('tone_user', JSON.stringify(updatedUser));
    };

    const value = {
        user,
        login,
        signup,
        logout,
        updateUser,
        loading
    };

    return (
        <AuthContext.Provider value={value}>
            {!loading && children}
        </AuthContext.Provider>
    );
};
