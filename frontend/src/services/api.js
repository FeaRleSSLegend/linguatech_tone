/**
 * API Service for Tone AI Analysis & Firebase Integration
 * Handles communication with Tone AI backend and Firebase Firestore
 * 
 * CRITICAL: This file includes robust fallback handling for sentiment analysis
 * to prevent addDoc crashes during live demos
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
    Timestamp,
    limit
} from '../lib/firebase';
import { containsPidginFlags } from '../utils/pidginDetection';


const TONE_API_URL = import.meta.env.VITE_TONE_API_URL || 'https://mutekikazu-linguatech-tone.hf.space';

/**
 * CRITICAL: Ensure all sentiment/toxicity fields have safe defaults
 * This prevents Firestore addDoc() crashes when API returns undefined
 */
function ensureSafeAnalysis(analysis) {
    return {
        text: analysis?.text || '',
        sentiment: {
            label: analysis?.sentiment?.label || 'neutral',
            confidence: typeof analysis?.sentiment?.confidence === 'number' ? analysis.sentiment.confidence : 0.5
        },
        toxicity: {
            label: analysis?.toxicity?.label || 'non-toxic',
            confidence: typeof analysis?.toxicity?.confidence === 'number' ? analysis.toxicity.confidence : 0
        },
        feedback: analysis?.feedback || 'Analysis complete.',
        should_warn: analysis?.should_warn || false,
        rephrase: analysis?.rephrase || null,
        emoji: analysis?.emoji || '😐'
    };
}

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
 */
export function getToxicityEmoji(toxicityLabel) {
    const emojiMap = {
        'non-toxic': '😊',
        'mildly toxic': '🟡',
        'toxic': '😡',
        'very toxic': '😡'
    };
    return emojiMap[toxicityLabel] || '😊';
}

/**
 * Get display emoji based ONLY on toxicity score
 */
export function getDisplayEmoji(analysis) {
    if (!analysis) return '😊';
    const toxicityLabel = analysis.toxicity?.label || 'non-toxic';
    return getToxicityEmoji(toxicityLabel);
}

/**
 * Analyze a message using Tone AI (DeBERTa + Groq LLM)
 * 
 * CRITICAL: Always returns a valid analysis object with defaults
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

        if (!response.ok) {
            console.warn(`Tone API error: ${response.status}, using fallback`);
            return getMockAnalysis(message);
        }

        const data = await response.json();

        // Ensure safe defaults for all fields
        const safeAnalysis = ensureSafeAnalysis({
            text: data.text || message,
            sentiment: data.sentiment,
            toxicity: data.toxicity,
            feedback: data.feedback,
            should_warn: data.should_warn,
            rephrase: data.rephrase
        });

        // Add emoji based on analysis
        safeAnalysis.emoji = getDisplayEmoji(safeAnalysis);

        return safeAnalysis;
    } catch (error) {
        console.error('Tone AI Analysis error:', error);
        console.log('Using fallback mock analysis');
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
        const chatsQuery = query(
            collection(db, 'chats'),
            where('type', '==', 'direct'),
            where('participants', 'array-contains', user.uid)
        );

        const snapshot = await getDocs(chatsQuery);

        for (const docSnap of snapshot.docs) {
            const chatData = docSnap.data();
            if (chatData.participants.includes(otherUserId)) {
                return docSnap.id;
            }
        }

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
        const chatDoc = await getDoc(doc(db, 'chats', chatId));
        if (!chatDoc.exists()) throw new Error('Chat not found');

        const chatData = chatDoc.data();
        if (!chatData.participants?.includes(user.uid)) {
            throw new Error('Access denied');
        }

        const messagesQuery = query(
            collection(db, 'messages'),
            where('chatId', '==', chatId),
            orderBy('createdAt', 'asc')
        );

        const snapshot = await getDocs(messagesQuery);
        const messages = [];

        for (const docSnap of snapshot.docs) {
            const msgData = docSnap.data();
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
                        label: msgData.sentimentLabel || 'neutral',
                        confidence: msgData.sentimentConfidence || 0.5
                    },
                    toxicity: {
                        label: msgData.toxicityLabel || 'non-toxic',
                        confidence: msgData.toxicityConfidence || 0
                    },
                    feedback: msgData.analysisFeedback || 'No analysis',
                    emoji: msgData.emoji || '😐'
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

/**
 * CRITICAL: Send message with robust fallback handling + AI Auto-Moderation
 * Ensures all fields have safe defaults to prevent Firestore crashes
 * Implements toxicity tracking and automatic temporary bans
 */
export async function sendMessage(chatId, text, sender, analysis) {
    const user = getCurrentUser();
    if (!user) throw new Error('Not authenticated');

    try {
        const chatDoc = await getDoc(doc(db, 'chats', chatId));
        if (!chatDoc.exists()) throw new Error('Chat not found');

        const chatData = chatDoc.data();
        if (!chatData.participants?.includes(user.uid)) {
            throw new Error('Access denied');
        }

        // CRITICAL: Ensure safe defaults for all analysis fields
        const safeAnalysis = ensureSafeAnalysis(analysis || {});

        // Check word blacklist for groups
        const wordBlacklist = chatData.wordBlacklist || [];
        const lowerText = text.toLowerCase();
        const containsBlacklistedWord = wordBlacklist.some(word => lowerText.includes(word.toLowerCase()));

        if (containsBlacklistedWord && chatData.type === 'group') {
            throw new Error('Your message contains a blacklisted word and cannot be sent.');
        }

        // Check if message is toxic
        const isToxic = safeAnalysis.toxicity.label === 'toxic' ||
                       safeAnalysis.toxicity.label === 'very toxic' ||
                       safeAnalysis.toxicity.label === 'mildly toxic';

        // Strict Mode: AI Auto-Mod for groups
        if (chatData.type === 'group' && chatData.strictMode && isToxic) {
            // Increment toxicity count
            const currentCount = chatData.moderationStats?.[user.uid] || 0;
            const newCount = currentCount + 1;

            // Update moderation stats
            await updateDoc(doc(db, 'chats', chatId), {
                [`moderationStats.${user.uid}`]: newCount,
                updatedAt: serverTimestamp()
            });

            // Auto temp-ban if reached 3 toxic messages
            if (newCount >= 3) {
                const banUntil = Date.now() + (2 * 60 * 60 * 1000); // 2 hours from now
                const tempBannedUsers = chatData.tempBannedUsers || [];

                // Remove old ban for this user if exists
                const updatedBans = tempBannedUsers.filter(b => b.userId !== user.uid);
                updatedBans.push({ userId: user.uid, bannedUntil: banUntil });

                await updateDoc(doc(db, 'chats', chatId), {
                    tempBannedUsers: updatedBans,
                    updatedAt: serverTimestamp()
                });

                throw new Error('You have been temporarily banned for 2 hours due to toxic behavior.');
            }

            throw new Error(`Warning: Toxic message detected (${newCount}/3). You will be auto-banned after 3 toxic messages.`);
        }

        const isFlagged = containsPidginFlags(text);

        const messageData = {
            chatId,
            senderId: user.uid,
            text,
            sentimentLabel: safeAnalysis.sentiment.label,
            sentimentConfidence: safeAnalysis.sentiment.confidence,
            toxicityLabel: safeAnalysis.toxicity.label,
            toxicityConfidence: safeAnalysis.toxicity.confidence,
            analysisFeedback: safeAnalysis.feedback,
            emoji: safeAnalysis.emoji,
            isFlagged: isFlagged,
            createdAt: serverTimestamp()
        };

        console.log('💾 Saving message with analysis:', messageData);

        const messageRef = await addDoc(collection(db, 'messages'), messageData);

        // Update chat's last message time
        await updateDoc(doc(db, 'chats', chatId), {
            lastMessage: text.length > 50 ? text.substring(0, 50) + '...' : text,
            lastMessageSenderId: user.uid,
            updatedAt: serverTimestamp()
        });

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

/**
 * Delete a message (Admin only - for "Delete for Everyone" feature)
 */
export async function deleteMessage(messageId, chatId) {
    const user = getCurrentUser();
    if (!user) throw new Error('Not authenticated');

    try {
        // Verify user is admin of the group
        const chatDoc = await getDoc(doc(db, 'chats', chatId));
        if (!chatDoc.exists()) throw new Error('Chat not found');

        const chatData = chatDoc.data();
        const isAdmin = chatData.creatorId === user.uid || chatData.admins?.includes(user.uid);

        if (!isAdmin) {
            throw new Error('Only admins can delete messages');
        }

        // Delete the message
        await deleteDoc(doc(db, 'messages', messageId));

        return { success: true };
    } catch (error) {
        console.error('Error deleting message:', error);
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
 * CRITICAL: Always returns valid defaults to prevent crashes
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