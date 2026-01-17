import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { AnimatePresence, motion } from 'framer-motion';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';

// Components
import Navbar from './components/Navbar';
import Hero from './components/Hero';
import UploadZone from './components/UploadZone';
import ResultsPanel from './components/ResultsPanel';
import History from './components/History';
import { AuthProvider, useAuth } from './context/AuthContext';
import { Login } from './components/Login';
import { Register } from './components/Register';

function AppContent() {
    const { user, token, loading } = useAuth();
    const [authView, setAuthView] = useState('login');
    const [activeTab, setActiveTab] = useState('upload');
    const [files, setFiles] = useState([]);
    const [scanning, setScanning] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    const [booted, setBooted] = useState(false);
    const [history, setHistory] = useState([]);

    // System Boot Effect & Load History
    useEffect(() => {
        setTimeout(() => setBooted(true), 800);
        const saved = localStorage.getItem('chestCareHistory');
        if (saved) {
            try {
                setHistory(JSON.parse(saved));
            } catch (e) {
                console.error("Failed to load history", e);
            }
        }
    }, [user]);

    if (loading) return null; // Let AuthProvider handle main loading screen

    if (!user) {
        return (
            <div className={`min-h-screen bg-primary transition-opacity duration-1000 ${booted ? 'opacity-100' : 'opacity-0'}`}>
                {authView === 'login'
                    ? <Login onSwitchToRegister={() => setAuthView('register')} />
                    : <Register onSwitchToLogin={() => setAuthView('login')} />}
            </div>
        );
    }

    // Save History
    const addToHistory = (newResult, filePreview) => {
        const newEntry = {
            id: Date.now(),
            result: newResult,
            preview: filePreview,
            timestamp: new Date().toISOString()
        };
        const updated = [newEntry, ...history];
        setHistory(updated);
        localStorage.setItem('chestCareHistory', JSON.stringify(updated));
    };

    const clearHistory = () => {
        setHistory([]);
        localStorage.removeItem('chestCareHistory');
    };

    const handleAnalysis = async () => {
        if (files.length === 0) return;

        setScanning(true);
        setError(null);
        setResult(null);

        // Use the last uploaded file for analysis
        const fileToAnalyze = files[files.length - 1].file;
        const filePreview = files[files.length - 1].preview; // Capture preview for history
        const formData = new FormData();
        formData.append('file', fileToAnalyze);

        try {
            // simulated "deep scan" delay for UX
            await new Promise(resolve => setTimeout(resolve, 2500));

            const response = await axios.post('/api/predict', formData, {
                headers: { Authorization: `Bearer ${token}` }
            });

            setResult(response.data);
            addToHistory(response.data, filePreview); // Save to history with preview
            setActiveTab('analysis'); // Auto-switch to analysis tab
        } catch (err) {
            console.error(err);
            setError("Analysis Failed. Please check connection.");
        } finally {
            setScanning(false);
        }
    };

    return (
        <div className={`min-h-screen bg-primary text-slate-200 font-sans selection:bg-accent/30 selection:text-white transition-opacity duration-1000 ${booted ? 'opacity-100' : 'opacity-0'}`}>

            {/* Global Background Grid */}
            <div className="fixed inset-0 bg-grid-pattern opacity-[0.05] pointer-events-none z-0"></div>

            <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />

            <main className="relative z-10 pt-24 pb-20 px-4 md:px-8 max-w-7xl mx-auto">

                <AnimatePresence mode="wait">
                    {activeTab === 'upload' && (
                        <motion.div
                            key="upload"
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -10 }}
                            transition={{ duration: 0.3 }}
                        >
                            <Hero />

                            <div id="upload-section">
                                <UploadZone
                                    files={files}
                                    setFiles={setFiles}
                                    onUpload={handleAnalysis}
                                    scanning={scanning}
                                />
                            </div>

                            {/* Error Toast */}
                            {error && (
                                <div className="mt-8 max-w-md mx-auto bg-red-500/10 border border-red-500/50 p-4 rounded-lg text-center text-red-200 text-sm">
                                    {error}
                                </div>
                            )}
                        </motion.div>
                    )}

                    {activeTab === 'analysis' && (
                        <motion.div
                            key="analysis"
                            initial={{ opacity: 0, scale: 0.95 }}
                            animate={{ opacity: 1, scale: 1 }}
                            exit={{ opacity: 0, scale: 1.05 }}
                            transition={{ duration: 0.4 }}
                        >
                            {result ? (
                                <ResultsPanel result={result} />
                            ) : (
                                <div className="flex flex-col items-center justify-center min-h-[50vh] text-slate-500">
                                    <p>No analysis data available.</p>
                                    <button
                                        onClick={() => setActiveTab('upload')}
                                        className="mt-4 text-accent hover:underline mb-8"
                                    >
                                        Upload an X-ray first
                                    </button>
                                </div>
                            )}
                        </motion.div>
                    )}

                    {activeTab === 'history' && (
                        <motion.div
                            key="history"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                        >
                            <History
                                history={history}
                                onClear={clearHistory}
                                onSelect={(item) => {
                                    setResult(item.result);
                                    setActiveTab('analysis');
                                }}
                            />
                        </motion.div>
                    )}
                </AnimatePresence>

            </main>

            {/* Footer */}
            <footer className="fixed bottom-0 left-0 right-0 p-4 text-center text-[10px] text-slate-600 font-mono z-0 pointer-events-none">
                AI CHEST DIAGNOSTIC SYSTEM V.3.0 // POWERED BY DEEP LEARNING
            </footer>
        </div>
    );
}

function App() {
    return (
        <Router>
            <AuthProvider>
                <AppContent />
            </AuthProvider>
        </Router>
    );
}

export default App;
