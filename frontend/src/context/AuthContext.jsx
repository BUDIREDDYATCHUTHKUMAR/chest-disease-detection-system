import React, { createContext, useState, useEffect, useContext } from 'react';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
    const [token, setToken] = useState(localStorage.getItem('token'));
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        console.log("AuthContext: Token changed:", token);
        if (token) {
            localStorage.setItem('token', token);
            fetchUser();
        } else {
            console.log("AuthContext: No token, clearing user");
            localStorage.removeItem('token');
            setUser(null);
            setLoading(false);
        }
    }, [token]);

    const fetchUser = async () => {
        console.log("AuthContext: Fetching user...");
        try {
            const response = await fetch('/api/users/me', {
                headers: { Authorization: `Bearer ${token}` }
            });
            console.log("AuthContext: Fetch user response status:", response.status);

            if (response.ok) {
                const data = await response.json();
                console.log("AuthContext: User fetched successfully:", data);
                setUser(data);
            } else {
                console.error("AuthContext: Fetch user failed");
                logout();
            }
        } catch (error) {
            console.error("AuthContext: Network error fetching user:", error);
            logout();
        } finally {
            setLoading(false);
        }
    };

    const login = (newToken) => {
        console.log("AuthContext: Login called with token:", newToken);
        setToken(newToken);
    };

    const logout = () => {
        console.log("AuthContext: Logout called");
        setToken(null);
        setUser(null);
        localStorage.removeItem('token');
    };

    return (
        <AuthContext.Provider value={{ token, user, login, logout, loading }}>
            {loading ? (
                <div className="min-h-screen bg-[#0B1120] flex items-center justify-center">
                    <div className="flex flex-col items-center gap-4">
                        <div className="w-12 h-12 border-4 border-cyan-500/30 border-t-cyan-500 rounded-full animate-spin" />
                        <p className="text-cyan-400 font-mono text-sm tracking-wider animate-pulse">INITIALIZING SECURITY PROTOCOLS...</p>
                    </div>
                </div>
            ) : children}
        </AuthContext.Provider>
    );
};

export const useAuth = () => useContext(AuthContext);
