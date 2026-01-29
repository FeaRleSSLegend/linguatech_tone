/**
 * API Service for Tone AI Analysis & Firebase Integration
 * Handles communication with Tone AI backend and Firebase Firestore
 */

import {
    auth,
    db,
    getCurrentUser,
    getUserProfile,
    updateOnlineStatus,
    createUserProfile,
    signInWithEmailAndPassword,
    createUserWithEmailAndPassword,
    signOut,
    updateProfile,
    collection,
    doc,
    setDoc,
    getDoc,
    getDocs,
    addDoc,
    updateDoc,
    query,
    where,
    orderBy,
    onSnapshot,
    serverTimestamp,
    Timestamp
} from '../lib/firebase';

const TONE_API_URL = import.meta.env.VITE_TONE_API_URL || 'https://mutekikazu-linguatech-tone.hf.space';

/**
 * Map sentiment label to emoji
 */
export function getSentimentEmoji(sentimentLabel) {
    const emojiMap = {
        'positive': '😊',
        'neutral': '😐',
        'negative': '☹️'
    };
    return emojiMap[sentimentLabel] || '😐';
}

/**
 * Map toxicity label to emoji (STRICT MAPPING)
 * Maps directly to toxicity score only, no sentiment mixing
 */
export function getToxicityEmoji(toxicityLabel) {
    const emojiMap = {
        'non-toxic': '😊',      // Clean/safe
        'mildly toxic': '🟡',   // Warning
        'toxic': '😡',          // Toxic
        'very toxic': '😡'      // Very toxic (same as toxic)
    };
    return emojiMap[toxicityLabel] || '😊';
}

/**
 * Get display emoji based ONLY on toxicity score
 * No sentiment fallback - pure toxicity mapping
 */
export function getDisplayEmoji(analysis) {
    if (!analysis) return '😊';

    // Only use toxicity label for emoji
    const toxicityLabel = analysis.toxicity?.label || 'non-toxic';
    return getToxicityEmoji(toxicityLabel);
}

/**
 * Analyze a message using Tone AI (DeBERTa + Groq LLM)
 *
 * @param {string} message - The message to analyze
 * @returns {Promise<Object>} Analysis result with sentiment, toxicity, and suggestions
 */
export async function analyzeMessage(message) {
    try {
        const response = await fetch(`${TONE_API_URL}/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: message }),
        });

        if (!response.ok) throw new Error(`Tone API error: ${response.status}`);

        const data = await response.json();

        // Add emoji based on analysis
        const emoji = getDisplayEmoji({
            sentiment: data.sentiment,
            toxicity: data.toxicity
        });

        // Return standardized schema
        return {
            text: data.text || message,
            sentiment: data.sentiment || { label: 'neutral', confidence: 0.5 },
            toxicity: data.toxicity || { label: 'non-toxic', confidence: 0 },
            feedback: data.feedback || 'Analysis complete.',
            should_warn: data.should_warn || false,
            rephrase: data.rephrase || null,
            emoji
        };
    } catch (error) {
        console.error('Tone AI Analysis error:', error);
        return getMockAnalysis(message);
    }
}

// ====== FIREBASE AUTH SERVICES ======

export async function loginUser(email, password) {
    try {
        const userCredential = await signInWithEmailAndPassword(auth, email, password);
        const user = userCredential.user;

        // Fetch profile from Firestore
        const profile = await getUserProfile(user.uid);

        // Update online status
        await updateOnlineStatus(true);

        return {
            user: {
                id: user.uid,
                email: user.email,
                ...profile
            }
        };
    } catch (error) {
        console.error('Login error:', error);
        throw new Error(error.message || 'Login failed');
    }
}

export async function registerUser(name, email, password, adminSecret) {
    try {
        const userCredential = await createUserWithEmailAndPassword(auth, email, password);
        const user = userCredential.user;

        // Update display name
        await updateProfile(user, { displayName: name });

        // Determine role
        const role = (adminSecret && adminSecret === import.meta.env.VITE_ADMIN_SECRET) ? 'admin' : 'user';

        // Create user profile in Firestore
        await createUserProfile(user.uid, { name, email, role });

        const profile = await getUserProfile(user.uid);

        return {
            user: {
                id: user.uid,
                email: user.email,
                ...profile
            }
        };
    } catch (error) {
        console.error('Registration error:', error);
        throw new Error(error.message || 'Registration failed');
    }
}

export async function logoutUser() {
    try {
        // Update online status before logout
        await updateOnlineStatus(false);
        await signOut(auth);
    } catch (error) {
        console.error('Logout error:', error);
        throw error;
    }
}

// ====== FIREBASE CHAT SERVICES ======

export async function getChats() {
    const user = getCurrentUser();
    if (!user) throw new Error('Not authenticated');

    try {
        // Query chats where current user is a participant
        const chatsQuery = query(
            collection(db, 'chats'),
            where('participants', 'array-contains', user.uid)
        );

        const snapshot = await getDocs(chatsQuery);
        const chats = [];

        for (const docSnap of snapshot.docs) {
            const chatData = { id: docSnap.id, ...docSnap.data() };

            // Get last message for each chat
            const messagesQuery = query(
                collection(db, 'messages'),
                where('chatId', '==', docSnap.id),
                orderBy('createdAt', 'desc'),
                limit(1)
            );

            const msgSnapshot = await getDocs(messagesQuery);
            if (!msgSnapshot.empty) {
                const lastMsg = msgSnapshot.docs[0].data();
                chatData.lastMessage = lastMsg.text;
                chatData.lastMessageAt = lastMsg.createdAt;
            }

            chats.push(chatData);
        }

        // Sort by last message time
        chats.sort((a, b) => {
            const timeA = a.lastMessageAt?.toMillis() || 0;
            const timeB = b.lastMessageAt?.toMillis() || 0;
            return timeB - timeA;
        });

        return chats;
    } catch (error) {
        console.error('Error fetching chats:', error);
        throw error;
    }
}

export async function createChat(name, type, creatorId, participants = []) {
    const user = getCurrentUser();
    if (!user) throw new Error('Not authenticated');

    try {
        // Ensure creator is in participants
        const allParticipants = [user.uid, ...participants].filter((v, i, a) => a.indexOf(v) === i);

        const chatData = {
            name,
            type,
            creatorId: creatorId || user.uid,
            participants: allParticipants,
            createdAt: serverTimestamp(),
            updatedAt: serverTimestamp()
        };

        const chatRef = await addDoc(collection(db, 'chats'), chatData);

        return {
            id: chatRef.id,
            ...chatData
        };
    } catch (error) {
        console.error('Error creating chat:', error);
        throw error;
    }
}

export async function createDirectChat(otherUserId, chatName = null) {
    const user = getCurrentUser();
    if (!user) throw new Error('Not authenticated');

    try {
        // Check if direct chat already exists
        const chatsQuery = query(
            collection(db, 'chats'),
            where('type', '==', 'direct'),
            where('participants', 'array-contains', user.uid)
        );

        const snapshot = await getDocs(chatsQuery);

        // Find existing chat with this user
        for (const docSnap of snapshot.docs) {
            const chatData = docSnap.data();
            if (chatData.participants.includes(otherUserId)) {
                return docSnap.id;
            }
        }

        // Create new direct chat
        const otherUserProfile = await getUserProfile(otherUserId);
        const name = chatName || otherUserProfile?.name || 'Direct Chat';

        const chatData = {
            name,
            type: 'direct',
            creatorId: user.uid,
            participants: [user.uid, otherUserId],
            createdAt: serverTimestamp(),
            updatedAt: serverTimestamp()
        };

        const chatRef = await addDoc(collection(db, 'chats'), chatData);
        return chatRef.id;
    } catch (error) {
        console.error('Error creating direct chat:', error);
        throw error;
    }
}

export async function getMessages(chatId) {
    const user = getCurrentUser();
    if (!user) throw new Error('Not authenticated');

    try {
        // Verify user has access to this chat
        const chatDoc = await getDoc(doc(db, 'chats', chatId));
        if (!chatDoc.exists()) throw new Error('Chat not found');

        const chatData = chatDoc.data();
        if (!chatData.participants?.includes(user.uid)) {
            throw new Error('Access denied');
        }

        // Fetch messages
        const messagesQuery = query(
            collection(db, 'messages'),
            where('chatId', '==', chatId),
            orderBy('createdAt', 'asc')
        );

        const snapshot = await getDocs(messagesQuery);
        const messages = [];

        for (const docSnap of snapshot.docs) {
            const msgData = docSnap.data();

            // Fetch sender info
            const senderProfile = await getUserProfile(msgData.senderId);

            messages.push({
                id: docSnap.id,
                chatId: msgData.chatId,
                text: msgData.text,
                sender: {
                    id: msgData.senderId,
                    name: senderProfile?.name || 'Unknown',
                    email: senderProfile?.email || ''
                },
                analysis: {
                    sentiment: {
                        label: msgData.sentimentLabel,
                        confidence: msgData.sentimentConfidence
                    },
                    toxicity: {
                        label: msgData.toxicityLabel,
                        confidence: msgData.toxicityConfidence
                    },
                    feedback: msgData.analysisFeedback,
                    emoji: msgData.emoji
                },
                timestamp: msgData.createdAt?.toDate()
            });
        }

        return messages;
    } catch (error) {
        console.error('Error fetching messages:', error);
        throw error;
    }
}

export async function sendMessage(chatId, text, sender, analysis) {
    const user = getCurrentUser();
    if (!user) throw new Error('Not authenticated');

    try {
        // Verify user has access to this chat
        const chatDoc = await getDoc(doc(db, 'chats', chatId));
        if (!chatDoc.exists()) throw new Error('Chat not found');

        const chatData = chatDoc.data();
        if (!chatData.participants?.includes(user.uid)) {
            throw new Error('Access denied');
        }

        const messageData = {
            chatId,
            senderId: user.uid,
            text,
            sentimentLabel: analysis?.sentiment?.label || 'neutral',
            sentimentConfidence: analysis?.sentiment?.confidence || 0,
            toxicityLabel: analysis?.toxicity?.label || analysis?.label || 'non-toxic',
            toxicityConfidence: analysis?.toxicity?.confidence || analysis?.confidence || 0,
            analysisFeedback: analysis?.feedback || 'No analysis available',
            emoji: analysis?.emoji || getDisplayEmoji(analysis) || '😐',
            createdAt: serverTimestamp()
        };

        const messageRef = await addDoc(collection(db, 'messages'), messageData);

        // Update chat's last message time
        await updateDoc(doc(db, 'chats', chatId), {
            updatedAt: serverTimestamp()
        });

        // Fetch sender profile
        const senderProfile = await getUserProfile(user.uid);

        return {
            id: messageRef.id,
            chatId: messageData.chatId,
            text: messageData.text,
            sender: {
                id: user.uid,
                name: senderProfile?.name || user.displayName || 'Unknown',
                email: senderProfile?.email || user.email
            },
            analysis: {
                sentiment: {
                    label: messageData.sentimentLabel,
                    confidence: messageData.sentimentConfidence
                },
                toxicity: {
                    label: messageData.toxicityLabel,
                    confidence: messageData.toxicityConfidence
                },
                feedback: messageData.analysisFeedback,
                emoji: messageData.emoji
            },
            timestamp: new Date()
        };
    } catch (error) {
        console.error('Error sending message:', error);
        throw error;
    }
}

// ====== ADMIN SERVICES ======

export async function getAdminReports() {
    const user = getCurrentUser();
    if (!user) throw new Error('Not authenticated');

    const profile = await getUserProfile(user.uid);
    if (profile?.role !== 'admin' && profile?.role !== 'super_admin') {
        throw new Error('Admin access required');
    }

    try {
        const reportsQuery = query(
            collection(db, 'reports'),
            orderBy('createdAt', 'desc')
        );

        const snapshot = await getDocs(reportsQuery);
        return snapshot.docs.map(doc => ({ id: doc.id, ...doc.data() }));
    } catch (error) {
        console.error('Error fetching reports:', error);
        throw error;
    }
}

export async function updateUserStatus(userId, status) {
    const user = getCurrentUser();
    if (!user) throw new Error('Not authenticated');

    try {
        await updateDoc(doc(db, 'users', userId), {
            status,
            updatedAt: serverTimestamp()
        });

        const updatedProfile = await getUserProfile(userId);
        return updatedProfile;
    } catch (error) {
        console.error('Error updating user status:', error);
        throw error;
    }
}

export async function getGroupReports(groupId) {
    const user = getCurrentUser();
    if (!user) throw new Error('Not authenticated');

    try {
        const reportsQuery = query(
            collection(db, 'reports'),
            where('chatId', '==', groupId),
            orderBy('createdAt', 'desc')
        );

        const snapshot = await getDocs(reportsQuery);
        return snapshot.docs.map(doc => ({ id: doc.id, ...doc.data() }));
    } catch (error) {
        console.error('Error fetching group reports:', error);
        throw error;
    }
}

export async function updateGroupBan(groupId, userId, isBanned) {
    const user = getCurrentUser();
    if (!user) throw new Error('Not authenticated');

    try {
        const chatRef = doc(db, 'chats', groupId);
        const chatDoc = await getDoc(chatRef);

        if (!chatDoc.exists()) throw new Error('Chat not found');

        const chatData = chatDoc.data();
        let bannedUsers = chatData.bannedUsers || [];

        if (isBanned && !bannedUsers.includes(userId)) {
            bannedUsers.push(userId);
        } else if (!isBanned) {
            bannedUsers = bannedUsers.filter(id => id !== userId);
        }

        await updateDoc(chatRef, {
            bannedUsers,
            updatedAt: serverTimestamp()
        });

        return { success: true };
    } catch (error) {
        console.error('Error updating group ban:', error);
        throw error;
    }
}

// ====== USER DISCOVERY ======

export async function searchUsers(searchQuery) {
    const user = getCurrentUser();
    if (!user) throw new Error('Not authenticated');

    try {
        const usersQuery = query(
            collection(db, 'users'),
            where('status', '==', 'active')
        );

        const snapshot = await getDocs(usersQuery);
        const users = snapshot.docs
            .map(doc => ({ id: doc.id, ...doc.data() }))
            .filter(u =>
                u.id !== user.uid &&
                (u.name?.toLowerCase().includes(searchQuery.toLowerCase()) ||
                    u.email?.toLowerCase().includes(searchQuery.toLowerCase()))
            )
            .slice(0, 10);

        return users;
    } catch (error) {
        console.error('Error searching users:', error);
        throw error;
    }
}

export async function getAllUsers() {
    const user = getCurrentUser();
    if (!user) throw new Error('Not authenticated');

    try {
        const usersQuery = query(
            collection(db, 'users'),
            where('status', '==', 'active'),
            orderBy('name')
        );

        const snapshot = await getDocs(usersQuery);
        return snapshot.docs
            .map(doc => ({ id: doc.id, ...doc.data() }))
            .filter(u => u.id !== user.uid);
    } catch (error) {
        console.error('Error fetching users:', error);
        throw error;
    }
}

// ====== FALLBACK MOCK ANALYSIS ======

/**
 * Generate mock analysis data when Tone AI is unavailable
 */
function getMockAnalysis(message) {
    const lowerMessage = message.toLowerCase().trim();

    if (!lowerMessage) {
        return {
            text: message,
            sentiment: { label: 'neutral', confidence: 0.5 },
            toxicity: { label: 'non-toxic', confidence: 0 },
            feedback: 'AI offline - using fallback detection',
            should_warn: false,
            rephrase: null,
            emoji: '😐'
        };
    }

    // Simple keyword detection
    const toxicWords = ['fuck', 'shit', 'bitch', 'asshole', 'kill', 'die', 'bastard'];
    const hostileWords = ['stupid', 'dumb', 'hate', 'ugly', 'shut up', 'useless'];
    const positiveWords = ['love', 'great', 'awesome', 'good', 'thanks', 'appreciate'];

    const hasToxic = toxicWords.some(word => lowerMessage.includes(word));
    const hasHostile = hostileWords.some(word => lowerMessage.includes(word));
    const hasPositive = positiveWords.some(word => lowerMessage.includes(word));

    let toxicityLabel = 'non-toxic';
    let toxicityConfidence = 0;
    let shouldWarn = false;
    let sentimentLabel = 'neutral';
    let sentimentConfidence = 0.5;
    let feedback = 'AI offline - basic analysis applied';
    let rephrase = null;
    let emoji = '😐';

    if (hasToxic) {
        toxicityLabel = 'very toxic';
        toxicityConfidence = 0.9;
        shouldWarn = true;
        sentimentLabel = 'negative';
        sentimentConfidence = 0.9;
        feedback = '🚫 Potentially toxic content detected (offline mode)';
        emoji = '🚫';
        rephrase = {
            suggestion: 'I have concerns about this. Can we discuss respectfully?',
            reason: 'Flagged by fallback detection'
        };
    } else if (hasHostile) {
        toxicityLabel = 'mildly toxic';
        toxicityConfidence = 0.6;
        shouldWarn = true;
        sentimentLabel = 'negative';
        sentimentConfidence = 0.7;
        feedback = '🟡 Message may seem unkind (offline mode)';
        emoji = '🟡';
        rephrase = {
            suggestion: 'I disagree with this approach. Let\'s find common ground.',
            reason: 'Flagged by fallback detection'
        };
    } else if (hasPositive) {
        sentimentLabel = 'positive';
        sentimentConfidence = 0.8;
        feedback = '😊 Positive tone detected (offline mode)';
        emoji = '😊';
    }

    return {
        text: message,
        sentiment: { label: sentimentLabel, confidence: sentimentConfidence },
        toxicity: { label: toxicityLabel, confidence: toxicityConfidence },
        feedback,
        should_warn: shouldWarn,
        rephrase,
        emoji
    };
}
