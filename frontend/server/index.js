import express from 'express';
import cors from 'cors';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import http from 'http';
import { Server } from 'socket.io';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DB_DIR = path.join(__dirname, 'db');
const USERS_FILE = path.join(DB_DIR, 'users.json');
const CHATS_FILE = path.join(DB_DIR, 'chats.json');
const MESSAGES_FILE = path.join(DB_DIR, 'messages.json');

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
    cors: {
        origin: "*",
        methods: ["GET", "POST"]
    }
});

const PORT = process.env.PORT || 3001;
const ADMIN_SECRET = "ADMIN_SECRET_2026"; // In production, use env variables

app.use(cors());
app.use(express.json());

// Initialize DB files
async function initDB() {
    try {
        await fs.mkdir(DB_DIR, { recursive: true });
        for (const file of [USERS_FILE, CHATS_FILE, MESSAGES_FILE]) {
            try {
                await fs.access(file);
            } catch {
                await fs.writeFile(file, JSON.stringify([]));
            }
        }
    } catch (err) {
        console.error('DB Init Error:', err);
    }
}

initDB();

async function getData(file) {
    const content = await fs.readFile(file, 'utf-8');
    return JSON.parse(content);
}

async function saveData(file, data) {
    await fs.writeFile(file, JSON.stringify(data, null, 2));
}

app.get('/', (req, res) => {
    res.send('Tone Backend is Live 🚀');
});

// --- Socket.io Logic ---
const userSocketMap = {};  // Add this line

io.on('connection', (socket) => {
    console.log('User connected:', socket.id);

    // 1. Map the connecting user - ADD THIS
    const userId = socket.handshake.query.userId;
    if (userId && userId !== "undefined") {
        userSocketMap[userId] = socket.id;
    }

    // 2. Tell everyone who is currently online - ADD THIS
    io.emit("getOnlineUsers", Object.keys(userSocketMap));

    socket.on('join_chat', async ({ chatId, userId }) => {
        const chats = await getData(CHATS_FILE);
        const chat = chats.find(c => c.id === chatId);

        if (chat && chat.bans && chat.bans.includes(userId)) {
            console.log(`Banned user ${userId} attempted to join chat: ${chatId}`);
            socket.emit('error', { message: 'You are banned from this group' });
            return;
        }

        socket.join(chatId);
        console.log(`User ${userId} joined chat: ${chatId}`);
    });

    socket.on('join_admin', () => {
        socket.join('admin_channel');
        console.log(`User ${socket.id} joined admin channel`);
    });

    socket.on('send_message', async (data) => {
        const users = await getData(USERS_FILE);
        const chats = await getData(CHATS_FILE);
        const sender = users.find(u => u.id === data.sender.id);
        const chat = chats.find(c => c.id === data.chatId);

        if (sender && sender.status === 'blocked') {
            console.log(`Blocked user ${data.sender.id} attempted to send message`);
            return;
        }

        if (chat && chat.type === 'group' && chat.bans && chat.bans.includes(data.sender.id)) {
            console.log(`Banned user ${data.sender.id} attempted to send message to group ${data.chatId}`);
            return;
        }

        // Broadcast to everyone in the room (including other tabs/sessions for the same user)
        console.log(`Broadcasting message from ${data.sender.name} to chat ${data.chatId}`);
        io.to(data.chatId).emit('receive_message', data);

        // Notify admins if message is aggressive
        if (data.analysis?.toxicity?.label === 'toxic' || data.analysis?.toxicity?.label === 'warning') {
            console.log(`Alerting admins about toxic message in chat ${data.chatId}`);
            io.to('admin_channel').emit('admin_toxicity_alert', data);
        }
    });

    socket.on('typing', (data) => {
        socket.to(data.chatId).emit('user_typing', data);
    });

    socket.on('stop_typing', (data) => {
        socket.to(data.chatId).emit('user_stop_typing', data);
    });

    socket.on('disconnect', () => {
        console.log('User disconnected:', socket.id);
        // Remove user from map and broadcast new list - ADD THIS
        if (userId) {
            delete userSocketMap[userId];
            io.emit("getOnlineUsers", Object.keys(userSocketMap));
        }
    });
});

/**
 * Heuristic-based Tone Analysis
 * (In a real-world app, this would call an LLM or ML model)
 */
function analyzeTone(message) {
    const lowerMessage = message.toLowerCase().trim();
    if (!lowerMessage) return null;

    // Dictionary: Explicitly Toxic/Hostile
    const toxicWords = ['fuck', 'shit', 'bitch', 'asshole', 'kill', 'die', 'bastard', 'cunt', 'idiot'];
    // Dictionary: Warning/Unkind
    const warningWords = ['stupid', 'dumb', 'hate', 'ugly', 'shut up', 'useless', 'trash', 'annoying'];
    // Dictionary: Positive/Empathetic
    const positiveWords = ['love', 'great', 'awesome', 'good', 'thanks', 'appreciate', 'wonderful', 'kind', 'happy', 'help'];

    let toxicScore = 0;
    toxicWords.forEach(word => {
        if (lowerMessage.includes(word)) toxicScore = 0.9;
    });

    let warningScore = 0;
    warningWords.forEach(word => {
        if (lowerMessage.includes(word)) warningScore = 0.6;
    });

    let positiveScore = 0;
    positiveWords.forEach(word => {
        if (lowerMessage.includes(word)) positiveScore = 0.8;
    });

    let toxicityLabel = 'safe';
    let toxicityConfidence = Math.max(toxicScore, warningScore * 0.7);
    let shouldWarn = false;
    let feedback = 'Balanced and objective tone.';
    let emoji = '😐';
    let sentimentLabel = 'neutral';
    let rephrase = null;

    if (toxicScore >= 0.9) {
        toxicityLabel = 'toxic';
        feedback = 'Highly toxic content detected. Please reconsider.';
        shouldWarn = true;
        emoji = '😡';
        sentimentLabel = 'negative';
    } else if (warningScore >= 0.6) {
        toxicityLabel = 'toxic';
        feedback = 'This message seems unkind or aggressive.';
        shouldWarn = true;
        emoji = '😠';
        sentimentLabel = 'negative';
    } else if (warningScore >= 0.4) {
        toxicityLabel = 'warning';
        feedback = 'Try using more constructive language.';
        emoji = '⚠️';
        sentimentLabel = 'negative';
    } else if (positiveScore >= 0.7) {
        sentimentLabel = 'positive';
        feedback = 'Friendly and supportive message!';
        emoji = '😊';
    }

    if (shouldWarn) {
        rephrase = {
            suggestions: [
                'I have a different perspective on this. Let\'s talk.',
                'I\'m feeling a bit frustrated. Can we check back later?',
                'This is difficult for me to discuss calmly right now.'
            ],
            reason: 'Categorized as Hostile/Aggressive language'
        };
    }

    return {
        sentiment: { label: sentimentLabel, confidence: positiveScore || 0.5 },
        toxicity: { label: toxicityLabel, confidence: toxicityConfidence },
        emoji,
        feedback,
        should_warn: shouldWarn,
        rephrase
    };
}

app.post('/analyze', (req, res) => {
    const { message } = req.body;

    if (!message) {
        return res.status(400).json({ error: 'Message is required' });
    }

    const analysis = analyzeTone(message);
    res.json(analysis || {
        sentiment: { label: 'neutral', confidence: 0.5 },
        toxicity: { label: 'safe', confidence: 0 },
        emoji: '😐',
        feedback: 'No significant tone detected.',
        should_warn: false
    });
});

// --- Auth Endpoints ---

app.post('/api/auth/register', async (req, res) => {
    const { name, email, password, adminSecret } = req.body;
    console.log(`Registration attempt for: ${email}`);
    const users = await getData(USERS_FILE);

    if (users.find(u => u.email === email)) {
        return res.status(400).json({ error: 'Email already exists' });
    }

    let role = 'user';
    if (adminSecret === ADMIN_SECRET) {
        role = 'super_admin';
    } else if (users.length === 0) {
        role = 'admin'; // Legacy support for first user
    }

    const newUser = {
        id: Date.now().toString(),
        name,
        email,
        password,
        role,
        status: 'active'
    };
    users.push(newUser);
    await saveData(USERS_FILE, users);

    res.status(201).json({ user: { id: newUser.id, name: newUser.name, email: newUser.email, role: newUser.role, status: newUser.status } });
});

app.post('/api/auth/login', async (req, res) => {
    const { email, password } = req.body;
    console.log(`Login attempt for: ${email}`);
    const users = await getData(USERS_FILE);
    const user = users.find(u => u.email === email && u.password === password);

    if (!user) {
        return res.status(401).json({ error: 'Invalid credentials' });
    }

    if (user.status === 'blocked') {
        return res.status(403).json({ error: 'Your account has been blocked due to toxicity.' });
    }

    res.json({ user: { id: user.id, name: user.name, email: user.email, role: user.role || 'user', status: user.status || 'active' } });
});

// --- Admin Middlewares ---
const requireAdmin = (req, res, next) => {
    const role = req.headers['x-user-role'];
    const userId = req.headers['x-user-id'];

    if (!userId || (role !== 'admin' && role !== 'super_admin')) {
        console.warn(`Unauthorized admin access attempt by user ${userId} with role ${role}`);
        return res.status(403).json({ error: 'Access denied. Admin privileges required.' });
    }
    next();
};

// --- Admin Endpoints ---

app.get('/api/admin/reports', requireAdmin, async (req, res) => {
    try {
        const messages = await getData(MESSAGES_FILE);
        const users = await getData(USERS_FILE);

        // Filter toxic/warning messages
        const toxicMessages = messages.filter(m =>
            m.analysis?.toxicity?.label === 'toxic' ||
            m.analysis?.toxicity?.label === 'warning'
        );

        // Group by user
        const report = {};

        toxicMessages.forEach(m => {
            const senderId = m.sender.id;
            if (!report[senderId]) {
                const user = users.find(u => u.id === senderId);
                report[senderId] = {
                    userId: senderId,
                    name: user?.name || m.sender.name || 'Unknown',
                    email: user?.email || m.sender.email || 'Unknown',
                    status: user?.status || 'active',
                    toxicCount: 0,
                    warningCount: 0,
                    incidents: []
                };
            }

            if (m.analysis.toxicity.label === 'toxic') {
                report[senderId].toxicCount++;
            } else {
                report[senderId].warningCount++;
            }

            report[senderId].incidents.push({
                text: m.text,
                chatId: m.chatId,
                timestamp: m.timestamp,
                severity: m.analysis.toxicity.label,
                confidence: m.analysis.toxicity.confidence
            });
        });

        res.json(Object.values(report).sort((a, b) => b.toxicCount - a.toxicCount));
    } catch (err) {
        console.error('Reports Error:', err);
        res.status(500).json({ error: 'Failed to generate reports' });
    }
});

app.post('/api/admin/users/:userId/status', requireAdmin, async (req, res) => {
    const { userId } = req.params;
    const { status } = req.body;

    try {
        const users = await getData(USERS_FILE);
        const userIndex = users.findIndex(u => u.id === userId);

        if (userIndex === -1) {
            return res.status(404).json({ error: 'User not found' });
        }

        // Don't allow blocking admins
        if (users[userIndex].role === 'super_admin' || users[userIndex].role === 'admin') {
            return res.status(403).json({ error: 'Cannot change status of an administrator' });
        }

        users[userIndex].status = status;
        await saveData(USERS_FILE, users);

        // Notify via socket to force log out or update UI
        io.emit('user_status_changed', { userId, status });

        res.json({ success: true, status });
    } catch (err) {
        console.error('Update Status Error:', err);
        res.status(500).json({ error: 'Failed to update user status' });
    }
});

// --- Group Admin Endpoints ---

app.get('/api/admin/groups/:groupId/reports', requireAdmin, async (req, res) => {
    const { groupId } = req.params;
    try {
        const messages = await getData(MESSAGES_FILE);
        const users = await getData(USERS_FILE);
        const chats = await getData(CHATS_FILE);

        const group = chats.find(c => c.id === groupId);
        if (!group) return res.status(404).json({ error: 'Group not found' });

        const groupMessages = messages.filter(m => m.chatId === groupId);
        const report = {};

        groupMessages.forEach(m => {
            const senderId = m.sender.id;
            if (m.analysis?.label === 'toxic' || m.analysis?.label === 'warning') {
                if (!report[senderId]) {
                    const user = users.find(u => u.id === senderId);
                    report[senderId] = {
                        userId: senderId,
                        name: m.sender.name || 'Unknown',
                        email: m.sender.email || 'Unknown',
                        status: user?.status || 'active',
                        isBannedFromGroup: group.bans?.includes(senderId) || false,
                        toxicCount: 0,
                        warningCount: 0,
                        incidents: []
                    };
                }

                if (m.analysis.label === 'toxic') report[senderId].toxicCount++;
                if (m.analysis.label === 'warning') report[senderId].warningCount++;

                report[senderId].incidents.push({
                    text: m.text,
                    timestamp: m.timestamp,
                    severity: m.analysis.label,
                    confidence: m.analysis.confidence
                });
            }
        });

        const offenders = Object.values(report).sort((a, b) => b.toxicCount - a.toxicCount);
        res.json({
            groupName: group.name,
            stats: {
                totalMessages: groupMessages.length,
                toxicCount: groupMessages.filter(m => m.analysis?.label === 'toxic').length
            },
            offenders: offenders.slice(0, 10) // Top 10 offenders
        });
    } catch (err) {
        res.status(500).json({ error: 'Failed to fetch group reports' });
    }
});

app.post('/api/admin/groups/:groupId/ban', requireAdmin, async (req, res) => {
    const { groupId } = req.params;
    const { userId, status } = req.body; // status: true to ban, false to unban

    try {
        const chats = await getData(CHATS_FILE);
        const chatIndex = chats.findIndex(c => c.id === groupId);

        if (chatIndex === -1) return res.status(404).json({ error: 'Group not found' });

        if (!chats[chatIndex].bans) chats[chatIndex].bans = [];

        if (status) {
            if (!chats[chatIndex].bans.includes(userId)) {
                chats[chatIndex].bans.push(userId);
            }
        } else {
            chats[chatIndex].bans = chats[chatIndex].bans.filter(id => id !== userId);
        }

        await saveData(CHATS_FILE, chats);
        io.to(groupId).emit('group_ban_updated', { userId, isBanned: status });
        res.json({ success: true });
    } catch (err) {
        res.status(500).json({ error: 'Failed to update group ban' });
    }
});

// --- Chat & Message Endpoints ---

app.get('/api/chats', async (req, res) => {
    const chats = await getData(CHATS_FILE);
    res.json(chats);
});

app.post('/api/chats', async (req, res) => {
    const { name, type, creatorId, participants } = req.body;
    try {
        const chats = await getData(CHATS_FILE);

        // Deduplication for direct chats
        if (type === 'direct' && participants && participants.length === 2) {
            const existingChat = chats.find(c =>
                c.type === 'direct' &&
                c.participants &&
                c.participants.length === 2 &&
                participants.every(p => c.participants.includes(p))
            );
            if (existingChat) return res.json(existingChat);
        }

        const newChat = {
            id: Date.now().toString(),
            name,
            type,
            online: true,
            admins: creatorId ? [creatorId] : [], // Set creator as admin
            participants: participants || [creatorId],
            bans: [],
            lastActivity: Date.now()
        };
        chats.push(newChat);
        await saveData(CHATS_FILE, chats);
        res.status(201).json(newChat);
    } catch (err) {
        console.error('Failed to create chat:', err);
        res.status(500).json({ error: 'Failed to create chat' });
    }
});

app.get('/api/messages/:chatId', async (req, res) => {
    const { chatId } = req.params;
    const messages = await getData(MESSAGES_FILE);
    res.json(messages.filter(m => m.chatId === chatId));
});

app.post('/api/messages', async (req, res) => {
    const { chatId, text, sender, analysis } = req.body;

    const chats = await getData(CHATS_FILE);
    const chat = chats.find(c => c.id === chatId);

    // Check if sender is blocked system-wide
    const users = await getData(USERS_FILE);
    const user = users.find(u => u.id === sender.id);
    if (user && user.status === 'blocked') {
        return res.status(403).json({ error: 'User is blocked' });
    }

    // Check if sender is banned from this specific group
    if (chat && chat.type === 'group' && chat.bans && chat.bans.includes(sender.id)) {
        return res.status(403).json({ error: 'You are banned from this group' });
    }

    const messages = await getData(MESSAGES_FILE);
    const newMessage = {
        id: Date.now().toString(),
        chatId,
        text,
        sender,
        analysis,
        timestamp: new Date().toISOString()
    };
    messages.push(newMessage);
    await saveData(MESSAGES_FILE, messages);
    res.status(201).json(newMessage);
});

server.listen(PORT, () => {
    console.log(`Tone Backend running on http://localhost:${PORT}`);
});
