import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Shield, AlertTriangle, User, Calendar, MessageSquare, ChevronDown, ChevronUp, ArrowLeft, BarChart3, Users, Clock, Ban } from 'lucide-react';
import { useParams, Link } from 'react-router-dom';
import { getGroupReports, updateGroupBan } from '../services/api';
import { useChat } from '../context/ChatContext';
import { doc, updateDoc } from 'firebase/firestore';
import { db } from '../lib/firebase';

export default function GroupAdminReports() {
    const { groupId } = useParams();
    const [reportData, setReportData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [expandedUser, setExpandedUser] = useState(null);
    const { socket } = useChat();

    const fetchReports = async () => {
        try {
            const data = await getGroupReports(groupId);
            setReportData(data);
        } catch (err) {
            console.error("Failed to load group reports", err);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchReports();
    }, [groupId]);

    useEffect(() => {
        if (!socket) return;

        socket.on('admin_toxicity_alert', (data) => {
            if (data.chatId === groupId) {
                fetchReports();
            }
        });

        socket.on('group_ban_updated', (data) => {
            fetchReports();
        });

        return () => {
            socket.off('admin_toxicity_alert');
            socket.off('group_ban_updated');
        };
    }, [socket, groupId]);

    const handleToggleBan = async (userId, isCurrentlyBanned) => {
        try {
            await updateGroupBan(groupId, userId, !isCurrentlyBanned);
            fetchReports();
        } catch (err) {
            alert(err.message || "Failed to update member status");
        }
    };

    const handleTempBan = async (userId, offenderName) => {
        if (!confirm(`Temporarily ban ${offenderName} for 24 hours?`)) return;
        try {
            const chatRef = doc(db, 'chats', groupId);
            const tempBannedUsers = reportData.tempBannedUsers || [];
            await updateDoc(chatRef, {
                tempBannedUsers: [...tempBannedUsers.filter(u => u.userId !== userId), {
                    userId,
                    bannedUntil: Date.now() + (24 * 60 * 60 * 1000) // 24 hours
                }]
            });
            alert(`${offenderName} has been temporarily banned for 24 hours`);
            fetchReports();
        } catch (err) {
            alert('Failed to temp ban user: ' + err.message);
        }
    };

    const handlePermBan = async (userId, offenderName) => {
        if (!confirm(`Permanently ban ${offenderName} from this group? They will not be able to see or access this group.`)) return;
        try {
            const chatRef = doc(db, 'chats', groupId);
            const permBannedUsers = reportData.permBannedUsers || [];
            await updateDoc(chatRef, {
                permBannedUsers: [...permBannedUsers, userId]
            });
            alert(`${offenderName} has been permanently banned from this group`);
            fetchReports();
        } catch (err) {
            alert('Failed to perm ban user: ' + err.message);
        }
    };

    if (loading) {
        return (
            <div className="min-h-screen bg-[#080808] flex items-center justify-center">
                <div className="w-8 h-8 border-4 border-[var(--color-primary)] border-t-transparent rounded-full animate-spin" />
            </div>
        );
    }

    if (!reportData) {
        return (
            <div className="min-h-screen bg-[#080808] text-[#e0ddd9] flex flex-col items-center justify-center p-8 text-center">
                <Shield className="w-16 h-16 text-[#2f3335] mb-4" />
                <h2 className="text-2xl font-bold mb-2">Group dashboard not available</h2>
                <p className="text-[#958d73] mb-6">You might not have admin privileges for this group.</p>
                <Link to="/app" className="bg-[#1a1a1a] px-6 py-2 rounded-lg border border-white/5 hover:border-white/10 transition-colors">
                    Back to App
                </Link>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-[#080808] text-[#e0ddd9] p-4 md:p-8">
            <div className="max-w-6xl mx-auto">
                {/* Header */}
                <header className="mb-8">
                    <Link to="/app" className="inline-flex items-center gap-2 text-[#958d73] hover:text-white mb-6 transition-colors">
                        <ArrowLeft className="w-4 h-4" /> Back to Chat
                    </Link>

                    <div className="flex flex-col md:flex-row md:items-end justify-between gap-6">
                        <div>
                            <div className="flex items-center gap-3 mb-2">
                                <span className="bg-emerald-500/10 text-emerald-500 text-[10px] font-bold uppercase tracking-widest px-2 py-0.5 rounded border border-emerald-500/20">Group Admin</span>
                                <h1 className="text-3xl font-bold">{reportData.groupName}</h1>
                            </div>
                            <p className="text-[#958d73]">Behavioral insights and moderation for this channel.</p>
                        </div>

                        <div className="flex gap-4">
                            <div className="bg-[#1a1a1a] border border-white/5 p-4 rounded-2xl flex items-center gap-4">
                                <div className="w-10 h-10 rounded-full bg-[var(--color-primary)]/10 flex items-center justify-center text-[var(--color-primary)]">
                                    <MessageSquare className="w-5 h-5" />
                                </div>
                                <div>
                                    <p className="text-[10px] uppercase tracking-wider text-[#958d73]">Activity</p>
                                    <p className="text-xl font-bold">{reportData.stats.totalMessages}</p>
                                </div>
                            </div>
                            <div className="bg-[#1a1a1a] border border-white/5 p-4 rounded-2xl flex items-center gap-4">
                                <div className="w-10 h-10 rounded-full bg-red-500/10 flex items-center justify-center text-red-500">
                                    <AlertTriangle className="w-5 h-5" />
                                </div>
                                <div>
                                    <p className="text-[10px] uppercase tracking-wider text-[#958d73]">Incidents</p>
                                    <p className="text-xl font-bold">{reportData.stats.toxicCount}</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </header>

                {/* Offenders List */}
                <div className="space-y-4">
                    <div className="flex items-center justify-between mb-4">
                        <h2 className="text-lg font-semibold text-[#958d73] flex items-center gap-2">
                            <BarChart3 className="w-5 h-5" />
                            Top Behavioral Risks
                        </h2>
                        <span className="text-xs text-[#5a5a5a]">{reportData.offenders.length} flagged members</span>
                    </div>

                    {reportData.offenders.length === 0 ? (
                        <div className="bg-[#1a1a1a] rounded-2xl p-12 text-center border border-white/5">
                            <CheckCircle className="w-12 h-12 text-emerald-500/30 mx-auto mb-4" />
                            <p className="text-[#958d73]">No behavioral concerns detected in this group yet.</p>
                        </div>
                    ) : reportData.offenders.map((offender) => (
                        <div key={offender.userId} className="bg-[#1a1a1a] border border-white/5 rounded-2xl overflow-hidden hover:border-white/10 transition-all">
                            <button
                                onClick={() => setExpandedUser(expandedUser === offender.userId ? null : offender.userId)}
                                className="w-full flex items-center justify-between p-6 text-left"
                            >
                                <div className="flex items-center gap-4">
                                    <div className="w-12 h-12 rounded-full bg-[#2f3335] flex items-center justify-center text-xl font-bold border border-white/5 relative">
                                        {offender.name.substring(0, 2).toUpperCase()}
                                        {offender.isBannedFromGroup && (
                                            <div className="absolute -top-1 -right-1 bg-red-500 text-white rounded-full p-0.5">
                                                <X className="w-3 h-3" />
                                            </div>
                                        )}
                                    </div>
                                    <div>
                                        <h3 className="text-lg font-bold flex items-center gap-2">
                                            {offender.name}
                                            {offender.isBannedFromGroup && (
                                                <span className="text-[9px] bg-red-500/10 text-red-500 px-2 py-0.5 rounded-full uppercase tracking-tighter border border-red-500/20">Banned from Group</span>
                                            )}
                                        </h3>
                                        <p className="text-sm text-[#958d73]">{offender.email}</p>
                                    </div>
                                </div>

                                <div className="flex items-center gap-8">
                                    <div className="text-right hidden sm:block">
                                        <p className="text-[10px] uppercase tracking-wider text-[#958d73]">Aggression Level</p>
                                        <p className="font-bold flex items-center gap-2 justify-end">
                                            <span className="text-red-500">{offender.toxicCount} Flagged</span>
                                        </p>
                                    </div>
                                    {expandedUser === offender.userId ? <ChevronUp /> : <ChevronDown />}
                                </div>
                            </button>

                            {expandedUser === offender.userId && (
                                <div className="px-6 pb-6 pt-2 space-y-4 border-t border-white/5">
                                    <div className="flex items-center justify-between flex-wrap gap-2">
                                        <p className="text-xs uppercase tracking-wider text-[#958d73]">Moderation Actions</p>
                                        <div className="flex gap-2 flex-wrap">
                                            <button
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    handleTempBan(offender.userId, offender.name);
                                                }}
                                                className="px-4 py-1.5 rounded-lg text-xs font-bold bg-yellow-500/10 text-yellow-500 hover:bg-yellow-500/20 border border-yellow-500/20 transition-all flex items-center gap-2"
                                            >
                                                <Clock className="w-3 h-3" />
                                                Temp Ban (24h)
                                            </button>
                                            <button
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    handlePermBan(offender.userId, offender.name);
                                                }}
                                                className="px-4 py-1.5 rounded-lg text-xs font-bold bg-red-500/10 text-red-500 hover:bg-red-500/20 border border-red-500/20 transition-all flex items-center gap-2"
                                            >
                                                <Ban className="w-3 h-3" />
                                                Perm Ban
                                            </button>
                                        </div>
                                    </div>

                                    <div className="space-y-2">
                                        {offender.incidents.map((incident, idx) => (
                                            <div key={idx} className="bg-[#080808] p-4 rounded-xl border border-white/5 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                                                <div className="flex-1">
                                                    <p className="text-white text-sm">"{incident.text}"</p>
                                                    <div className="flex items-center gap-3 mt-2 text-[10px] text-[#958d73]">
                                                        <span className="flex items-center gap-1"><Calendar className="w-3 h-3" /> {new Date(incident.timestamp).toLocaleDateString()}</span>
                                                        <span className="flex items-center gap-1"><Shield className="w-3 h-3" /> {(incident.confidence * 100).toFixed(0)}% Confidence</span>
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
                                </div>
                            )}
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}

function CheckCircle({ className }) {
    return (
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
    )
}
