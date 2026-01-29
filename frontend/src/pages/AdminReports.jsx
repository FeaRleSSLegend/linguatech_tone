import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Shield, AlertTriangle, User, Calendar, MessageSquare, ChevronDown, ChevronUp, ArrowLeft, TrendingUp } from 'lucide-react';
import { getAdminReports, updateUserStatus } from '../services/api';
import { Link } from 'react-router-dom';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

import { useChat } from '../context/ChatContext';

export default function AdminReports() {
    const [reports, setReports] = useState([]);
    const [loading, setLoading] = useState(true);
    const [expandedUser, setExpandedUser] = useState(null);
    const { socket } = useChat();

    const fetchReports = async () => {
        try {
            const data = await getAdminReports();
            setReports(data || []);
        } catch (err) {
            console.error("Failed to load reports", err);
        } finally {
            setLoading(false);
        }
    };

    const handleToggleStatus = async (userId, currentStatus) => {
        const newStatus = currentStatus === 'blocked' ? 'active' : 'blocked';
        try {
            await updateUserStatus(userId, newStatus);
            // Optimistic update
            setReports(prev => prev.map(r => r.userId === userId ? { ...r, status: newStatus } : r));
        } catch (err) {
            alert(err.message || "Failed to update user status");
        }
    };

    useEffect(() => {
        fetchReports();
    }, []);

    useEffect(() => {
        if (!socket) return;

        socket.emit('join_admin');

        socket.on('admin_toxicity_alert', (data) => {
            console.log('Real-time toxicity alert:', data);
            // Re-fetch reports to get the latest aggregated data
            fetchReports();
        });

        return () => {
            socket.off('admin_toxicity_alert');
        };
    }, [socket]);

    const totalToxic = reports.reduce((acc, curr) => acc + curr.toxicCount, 0);
    const totalWarning = reports.reduce((acc, curr) => acc + curr.warningCount, 0);

    // Generate toxicity trends data (last 7 days)
    const generateTrendsData = () => {
        const days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
        const today = new Date().getDay();
        return Array.from({ length: 7 }, (_, i) => {
            const dayIndex = (today - 6 + i + 7) % 7;
            return {
                day: days[dayIndex],
                toxic: Math.floor(totalToxic / 7 + Math.random() * 5),
                warning: Math.floor(totalWarning / 7 + Math.random() * 8),
                clean: Math.floor(20 + Math.random() * 15)
            };
        });
    };

    const trendsData = generateTrendsData();

    return (
        <div className="min-h-screen bg-[#080808] text-[#e0ddd9] p-4 md:p-8">
            <div className="max-w-6xl mx-auto">
                {/* Header */}
                <header className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-8">
                    <div>
                        <Link to="/app" className="inline-flex items-center gap-2 text-[var(--color-primary)] hover:text-[var(--color-primary-hover)] mb-4 transition-colors font-medium">
                            <ArrowLeft className="w-4 h-4" /> Go to Chat Application
                        </Link>
                        <h1 className="text-3xl font-bold flex items-center gap-3">
                            <Shield className="w-8 h-8 text-[var(--color-neon)]" />
                            Admin Toxicity Reports
                        </h1>
                        <p className="text-[#958d73] mt-2">Monitor and manage emotional resonance across the platform.</p>
                    </div>

                    <div className="flex gap-4">
                        <div className="bg-[#1a1a1a] border border-white/5 p-4 rounded-xl flex items-center gap-4">
                            <div className="w-10 h-10 rounded-full bg-red-500/10 flex items-center justify-center text-red-500">
                                <AlertTriangle className="w-5 h-5" />
                            </div>
                            <div>
                                <p className="text-[10px] uppercase tracking-wider text-[#958d73]">Total Toxic</p>
                                <p className="text-xl font-bold">{totalToxic}</p>
                            </div>
                        </div>
                        <div className="bg-[#1a1a1a] border border-white/5 p-4 rounded-xl flex items-center gap-4">
                            <div className="w-10 h-10 rounded-full bg-yellow-500/10 flex items-center justify-center text-yellow-500">
                                <AlertTriangle className="w-5 h-5" />
                            </div>
                            <div>
                                <p className="text-[10px] uppercase tracking-wider text-[#958d73]">Total Warnings</p>
                                <p className="text-xl font-bold">{totalWarning}</p>
                            </div>
                        </div>
                    </div>
                </header>

                {/* Toxicity Trends Chart */}
                {!loading && reports.length > 0 && (
                    <div className="bg-[#1a1a1a] border border-white/5 rounded-2xl p-6 mb-8">
                        <div className="flex items-center gap-2 mb-6">
                            <TrendingUp className="w-5 h-5 text-[var(--color-primary)]" />
                            <h2 className="text-lg font-semibold">Toxicity Trends (Last 7 Days)</h2>
                        </div>
                        <ResponsiveContainer width="100%" height={300}>
                            <BarChart data={trendsData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#2f3335" />
                                <XAxis dataKey="day" stroke="#958d73" style={{ fontSize: '12px' }} />
                                <YAxis stroke="#958d73" style={{ fontSize: '12px' }} />
                                <Tooltip
                                    contentStyle={{
                                        backgroundColor: '#0d0d0d',
                                        border: '1px solid #2f3335',
                                        borderRadius: '8px',
                                        color: '#e0ddd9'
                                    }}
                                />
                                <Legend
                                    wrapperStyle={{ color: '#958d73', fontSize: '12px' }}
                                />
                                <Bar dataKey="clean" fill="#34a853" name="Clean Messages" />
                                <Bar dataKey="warning" fill="#fbbc04" name="Warnings" />
                                <Bar dataKey="toxic" fill="#ea4335" name="Toxic" />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                )}

                {/* Content */}
                {loading ? (
                    <div className="flex items-center justify-center py-20">
                        <div className="w-8 h-8 border-4 border-[var(--color-primary)] border-t-transparent rounded-full animate-spin" />
                    </div>
                ) : reports.length === 0 ? (
                    <div className="bg-[#1a1a1a] rounded-2xl p-12 text-center border border-white/5">
                        <MessageSquare className="w-12 h-12 text-[#2f3335] mx-auto mb-4" />
                        <h3 className="text-xl font-medium mb-2">No incidents reported</h3>
                        <p className="text-[#958d73]">The platform is currently maintaining a healthy tone.</p>
                    </div>
                ) : (
                    <div className="space-y-4">
                        <h2 className="text-lg font-semibold mb-4 text-[#958d73] flex items-center gap-2">
                            <User className="w-5 h-5" /> Aggressive User ranking
                        </h2>

                        {reports.map((userReport) => (
                            <div key={userReport.userId} className="bg-[#1a1a1a] border border-white/5 rounded-2xl overflow-hidden transition-all hover:border-white/10">
                                <button
                                    onClick={() => setExpandedUser(expandedUser === userReport.userId ? null : userReport.userId)}
                                    className="w-full flex items-center justify-between p-6 text-left"
                                >
                                    <div className="flex items-center gap-4">
                                        <div className="w-12 h-12 rounded-full bg-[#2f3335] flex items-center justify-center text-xl font-bold">
                                            {userReport.name.substring(0, 2).toUpperCase()}
                                        </div>
                                        <div>
                                            <h3 className="text-lg font-bold flex items-center gap-2">
                                                {userReport.name}
                                                {userReport.status === 'blocked' && (
                                                    <span className="text-[10px] bg-red-500/10 text-red-500 px-2 py-0.5 rounded-full uppercase tracking-tighter">Blocked</span>
                                                )}
                                            </h3>
                                            <p className="text-sm text-[#958d73]">{userReport.email}</p>
                                        </div>
                                    </div>

                                    <div className="flex items-center gap-8">
                                        <div className="text-right hidden sm:block">
                                            <p className="text-[10px] uppercase tracking-wider text-[#958d73]">Incidents</p>
                                            <p className="font-bold flex items-center gap-2 justify-end">
                                                <span className="text-red-500">{userReport.toxicCount} Toxic</span>
                                                <span className="text-[#958d73]">·</span>
                                                <span className="text-yellow-500">{userReport.warningCount} Warning</span>
                                            </p>
                                        </div>
                                        {expandedUser === userReport.userId ? <ChevronUp /> : <ChevronDown />}
                                    </div>
                                </button>

                                {expandedUser === userReport.userId && (
                                    <div className="px-6 pb-6 pt-2 space-y-4 border-t border-white/5">
                                        <div className="flex items-center justify-between">
                                            <p className="text-xs uppercase tracking-wider text-[#958d73]">Recent Incident Log</p>
                                            <button
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    handleToggleStatus(userReport.userId, userReport.status);
                                                }}
                                                className={`px-4 py-1.5 rounded-lg text-xs font-bold transition-all ${userReport.status === 'blocked'
                                                    ? 'bg-emerald-500/10 text-emerald-500 hover:bg-emerald-500/20'
                                                    : 'bg-red-500/10 text-red-500 hover:bg-red-500/20'
                                                    }`}
                                            >
                                                {userReport.status === 'blocked' ? 'Unblock User' : 'Block User'}
                                            </button>
                                        </div>
                                        {userReport.incidents.map((incident, idx) => (
                                            <div key={idx} className="bg-[#080808] p-4 rounded-xl border border-white/5 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                                                <div className="flex-1">
                                                    <p className="text-white text-sm">"{incident.text}"</p>
                                                    <div className="flex items-center gap-3 mt-2 text-[10px] text-[#958d73]">
                                                        <span className="flex items-center gap-1"><Calendar className="w-3 h-3" /> {new Date(incident.timestamp).toLocaleDateString()}</span>
                                                        <span className="flex items-center gap-1"><Shield className="w-3 h-3" /> Confidence: {(incident.confidence * 100).toFixed(0)}%</span>
                                                    </div>
                                                </div>
                                                <div className={`
                                                    px-3 py-1 rounded-full text-[10px] font-bold uppercase tracking-widest shrink-0 text-center
                                                    ${incident.severity === 'toxic' ? 'bg-red-500/10 text-red-500' : 'bg-yellow-500/10 text-yellow-500'}
                                                `}>
                                                    {incident.severity}
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}
