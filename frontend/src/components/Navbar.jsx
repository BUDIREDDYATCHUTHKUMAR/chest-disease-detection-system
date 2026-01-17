import { useAuth } from '../context/AuthContext';
import { Activity, LayoutDashboard, History, LogOut } from 'lucide-react';

const Navbar = ({ activeTab, setActiveTab }) => {
    const { logout, user } = useAuth();
    const tabs = [
        { id: 'upload', label: 'Upload X-ray', icon: LayoutDashboard },
        { id: 'analysis', label: 'Disease Analysis', icon: Activity },
        { id: 'history', label: 'History', icon: History },
    ];

    return (
        <nav className="fixed top-0 left-0 right-0 z-50 px-6 py-4 bg-primary/80 backdrop-blur-md border-b border-white/5">
            <div className="max-w-7xl mx-auto flex items-center justify-between">
                {/* Logo Section */}
                <div className="flex items-center gap-3">
                    <div className="relative">
                        <div className="absolute inset-0 bg-accent/20 blur-xl rounded-full"></div>
                        <Activity className="w-8 h-8 text-accent relative z-10" />
                    </div>
                    <div>
                        <h1 className="text-2xl font-bold font-sans tracking-tight text-white">
                            AI Chest<span className="text-accent">Care</span>
                        </h1>
                        <p className="text-[10px] text-slate-400 font-mono tracking-[0.2em] uppercase">
                            Medical Diagnostic System
                        </p>
                    </div>
                </div>

                {/* Navigation Tabs */}
                <div className="hidden md:flex items-center gap-1 bg-surface/50 p-1 rounded-full border border-white/5">
                    {tabs.map((tab) => (
                        <button
                            key={tab.id}
                            onClick={() => setActiveTab(tab.id)}
                            className={`
                flex items-center gap-2 px-6 py-2.5 rounded-full text-sm font-medium transition-all duration-300
                ${activeTab === tab.id
                                    ? 'bg-accent/10 text-accent shadow-[0_0_10px_rgba(6,182,212,0.15)] border border-accent/20'
                                    : 'text-slate-400 hover:text-white hover:bg-white/5'}
              `}
                        >
                            <tab.icon className="w-4 h-4" />
                            {tab.label}
                        </button>
                    ))}
                </div>

                {/* System Status / Profile */}
                <div className="flex items-center gap-4">
                    <div className="hidden md:flex items-center gap-2 text-sm text-slate-400">
                        <span className="text-accent">{user?.username}</span>
                    </div>

                    <button
                        onClick={logout}
                        className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-red-500/10 border border-red-500/20 text-red-400 hover:bg-red-500/20 transition-all text-xs font-mono tracking-wide"
                        title="Disconnect Session"
                    >
                        <LogOut className="w-3 h-3" />
                        LOGOUT
                    </button>

                    <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/20">
                        <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></div>
                        <span className="text-xs font-mono text-emerald-500 tracking-wide">SYSTEM ONLINE</span>
                    </div>
                </div>
            </div>
        </nav>
    );
};

export default Navbar;
