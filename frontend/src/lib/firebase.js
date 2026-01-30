/**
 * 🚨 EMERGENCY FIX: Firebase with EXACT Firestore schema matching
 * - Uses `updatedAt` field (NOT lastActivity)
 * - Updates `lastMessage` field on every message send
 * - Matches composite index: participants + updatedAt
 */

import { initializeApp } from 'firebase/app';
import {
    getAuth,
    signInWithEmailAndPassword,
    createUserWithEmailAndPassword,
    signOut,
    onAuthStateChanged,
    updateProfile
} from 'firebase/auth';
import {
    getFirestore,
    collection,
    doc,
    setDoc,
    getDoc,
    getDocs,
    addDoc,
    updateDoc,
    deleteDoc,
    query,
    where,
    orderBy,
    onSnapshot,
    serverTimestamp,
    Timestamp,
    limit,
    writeBatch
} from 'firebase/firestore';
import {
    getStorage,
    ref,
    uploadBytes,
    getDownloadURL
} from 'firebase/storage';

const firebaseConfig = {
    apiKey: import.meta.env.VITE_FIREBASE_API_KEY,
    authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN,
    projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID,
    storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET,
    messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID,
    appId: import.meta.env.VITE_FIREBASE_APP_ID
};

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const db = getFirestore(app);
const storage = getStorage(app);

export { auth, db, storage };

export {
    signInWithEmailAndPassword,
    createUserWithEmailAndPassword,
    signOut,
    onAuthStateChanged,
    updateProfile,
    collection,
    doc,
    setDoc,
    getDoc,
    getDocs,
    addDoc,
    updateDoc,
    deleteDoc,
    query,
    where,
    orderBy,
    onSnapshot,
    serverTimestamp,
    Timestamp,
    limit,
    writeBatch
};

export function getCurrentUser() {
    return auth.currentUser;
}

export async function getUserProfile(userId = null) {
    const uid = userId || auth.currentUser?.uid;
    if (!uid) return null;

    try {
        const userDoc = await getDoc(doc(db, 'users', uid));
        if (userDoc.exists()) {
            return { id: userDoc.id, ...userDoc.data() };
        }
        return null;
    } catch (error) {
        console.error('Error fetching user profile:', error);
        return null;
    }
}

export async function updateOnlineStatus(isOnline) {
    const user = auth.currentUser;
    if (!user) return;

    try {
        await updateDoc(doc(db, 'users', user.uid), {
            isOnline,
            lastSeen: serverTimestamp()
        });
    } catch (error) {
        console.error('Error updating online status:', error);
    }
}

export async function createUserProfile(userId, data) {
    try {
        await setDoc(doc(db, 'users', userId), {
            name: data.name,
            email: data.email,
            role: data.role || 'user',
            status: 'active',
            isOnline: true,
            profilePictureUrl: data.profilePictureUrl || '',
            lastSeen: serverTimestamp(),
            createdAt: serverTimestamp()
        });
    } catch (error) {
        console.error('Error creating user profile:', error);
        throw error;
    }
}

export async function updateUserProfile(userId, updates) {
    try {
        await updateDoc(doc(db, 'users', userId), {
            ...updates,
            updatedAt: serverTimestamp()
        });
    } catch (error) {
        console.error('Error updating user profile:', error);
        throw error;
    }
}

export function subscribeToUserPresence(userId, callback) {
    const userRef = doc(db, 'users', userId);
    return onSnapshot(userRef, (snapshot) => {
        if (snapshot.exists()) {
            callback({ id: snapshot.id, ...snapshot.data() });
        }
    });
}

export async function canAccessChat(chatId, userId) {
    try {
        const chatDoc = await getDoc(doc(db, 'chats', chatId));
        if (!chatDoc.exists()) return false;

        const chatData = chatDoc.data();
        return chatData.participants?.includes(userId) || false;
    } catch (error) {
        console.error('Error checking chat access:', error);
        return false;
    }
}

// ====== FRIEND REQUEST FUNCTIONS ======

export async function searchUserByEmail(email) {
    try {
        const usersQuery = query(
            collection(db, 'users'),
            where('email', '==', email.toLowerCase()),
            limit(1)
        );
        const snapshot = await getDocs(usersQuery);

        if (snapshot.empty) return null;

        const userDoc = snapshot.docs[0];
        return { id: userDoc.id, ...userDoc.data() };
    } catch (error) {
        console.error('Error searching user:', error);
        throw error;
    }
}

export async function sendFriendRequest(fromUserId, toUserId) {
    try {
        if (fromUserId === toUserId) {
            throw new Error('Cannot send friend request to yourself');
        }

        const existingQuery = query(
            collection(db, 'friend_requests'),
            where('fromUserId', '==', fromUserId),
            where('toUserId', '==', toUserId)
        );
        const existingSnapshot = await getDocs(existingQuery);

        if (!existingSnapshot.empty) {
            throw new Error('Friend request already sent');
        }

        const reverseQuery = query(
            collection(db, 'friend_requests'),
            where('fromUserId', '==', toUserId),
            where('toUserId', '==', fromUserId)
        );
        const reverseSnapshot = await getDocs(reverseQuery);

        if (!reverseSnapshot.empty) {
            throw new Error('This user already sent you a request');
        }

        const requestRef = await addDoc(collection(db, 'friend_requests'), {
            fromUserId,
            toUserId,
            status: 'pending',
            createdAt: serverTimestamp()
        });

        return { id: requestRef.id };
    } catch (error) {
        console.error('Error sending friend request:', error);
        throw error;
    }
}

export async function getFriendRequests(userId) {
    try {
        const requestsQuery = query(
            collection(db, 'friend_requests'),
            where('toUserId', '==', userId),
            where('status', '==', 'pending'),
            orderBy('createdAt', 'desc')
        );

        const snapshot = await getDocs(requestsQuery);
        const requests = [];

        for (const docSnap of snapshot.docs) {
            const requestData = docSnap.data();
            const fromUser = await getUserProfile(requestData.fromUserId);

            requests.push({
                id: docSnap.id,
                ...requestData,
                fromUser
            });
        }

        return requests;
    } catch (error) {
        console.error('Error fetching friend requests:', error);
        return [];
    }
}

/**
 * 🚨 EMERGENCY FIX: Accept friend request with EXACT schema
 * - Creates chat with `updatedAt` field (NOT lastActivity)
 * - Initializes `lastMessage` field for preview system
 */
export async function acceptFriendRequest(requestId, fromUserId, toUserId) {
    try {
        const fromUserProfile = await getUserProfile(fromUserId);
        const toUserProfile = await getUserProfile(toUserId);

        if (!fromUserProfile || !toUserProfile) {
            throw new Error('User profiles not found');
        }

        const batch = writeBatch(db);

        // 1. Update friend request status
        const requestRef = doc(db, 'friend_requests', requestId);
        batch.update(requestRef, {
            status: 'accepted',
            acceptedAt: serverTimestamp()
        });

        // 2. Create direct chat with EXACT schema matching your DB
        const chatRef = doc(collection(db, 'chats'));
        const chatData = {
            name: `${fromUserProfile.name} & ${toUserProfile.name}`,
            type: 'direct',  // 🔥 EXACT: lowercase 'direct'
            isGroup: false,
            creatorId: toUserId,
            participants: [fromUserId, toUserId],
            participantNames: {
                [fromUserId]: fromUserProfile.name,
                [toUserId]: toUserProfile.name
            },
            lastMessage: '',  // 🔥 NEW: For preview system
            lastMessageSenderId: '',
            createdAt: serverTimestamp(),
            updatedAt: serverTimestamp()  // 🔥 EXACT: updatedAt (NOT lastActivity)
        };
        batch.set(chatRef, chatData);

        await batch.commit();

        console.log('✅ [EMERGENCY] Friend request accepted, chat created');

        return {
            chatId: chatRef.id,
            ...chatData
        };
    } catch (error) {
        console.error('❌ Error accepting friend request:', error);
        throw error;
    }
}

export async function rejectFriendRequest(requestId) {
    try {
        await updateDoc(doc(db, 'friend_requests', requestId), {
            status: 'rejected',
            rejectedAt: serverTimestamp()
        });
    } catch (error) {
        console.error('Error rejecting friend request:', error);
        throw error;
    }
}

export function subscribeToFriendRequests(userId, callback) {
    const requestsQuery = query(
        collection(db, 'friend_requests'),
        where('toUserId', '==', userId),
        where('status', '==', 'pending'),
        orderBy('createdAt', 'desc')
    );

    return onSnapshot(requestsQuery, async (snapshot) => {
        const requests = [];

        for (const docSnap of snapshot.docs) {
            const requestData = docSnap.data();
            const fromUser = await getUserProfile(requestData.fromUserId);

            requests.push({
                id: docSnap.id,
                ...requestData,
                fromUser
            });
        }

        callback(requests);
    }, (error) => {
        console.error('❌ Friend requests listener error:', error);
    });
}

/**
 * 🚨 CRITICAL FIX: Subscribe to chats using EXACT Firestore schema
 * - Uses `updatedAt` field (NOT lastActivity)
 * - Matches composite index: participants (array-contains) + updatedAt (descending)
 */
export function subscribeToChats(userId, callback) {
    console.log('🔥 [EMERGENCY] subscribeToChats starting for user:', userId);

    const chatsQuery = query(
        collection(db, 'chats'),
        where('participants', 'array-contains', userId),
        orderBy('updatedAt', 'desc')  // 🔥 EXACT: updatedAt (NOT lastActivity)
    );

    return onSnapshot(chatsQuery, (snapshot) => {
        console.log('✅ [EMERGENCY] Firestore snapshot received:', {
            userId,
            documentCount: snapshot.docs.length,
            chatIds: snapshot.docs.map(d => d.id)
        });

        const chats = snapshot.docs.map(doc => {
            const data = doc.data();
            return {
                id: doc.id,
                ...data
            };
        });

        callback(chats);
    }, (error) => {
        console.error('❌ [EMERGENCY] Chats listener error:', error);
        console.error('Error details:', {
            code: error.code,
            message: error.message,
            stack: error.stack
        });
    });
}

/**
 * Subscribe to messages for a specific chat
 */
export function subscribeToMessages(chatId, userId, callback) {
    return onSnapshot(doc(db, 'chats', chatId), async (chatSnap) => {
        if (!chatSnap.exists()) {
            callback([]);
            return;
        }

        const chatData = chatSnap.data();
        if (!chatData.participants?.includes(userId)) {
            console.warn('Privacy Filter: User not authorized');
            callback([]);
            return;
        }

        const messagesQuery = query(
            collection(db, 'messages'),
            where('chatId', '==', chatId),
            orderBy('createdAt', 'asc')
        );

        onSnapshot(messagesQuery, async (snapshot) => {
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
                    isFlagged: msgData.isFlagged || false,  // 🔥 NEW: Pidgin detection
                    timestamp: msgData.createdAt?.toDate()
                });
            }

            callback(messages);
        });
    });
}

// ====== FIREBASE STORAGE FUNCTIONS ======

export async function uploadProfilePicture(userId, file) {
    try {
        if (!file.type.startsWith('image/')) {
            throw new Error('File must be an image');
        }

        const maxSize = 5 * 1024 * 1024;
        if (file.size > maxSize) {
            throw new Error('File size must be less than 5MB');
        }

        const fileExtension = file.name.split('.').pop();
        const fileName = `${userId}.${fileExtension}`;
        const storageRef = ref(storage, `avatars/${fileName}`);

        const snapshot = await uploadBytes(storageRef, file);
        const downloadURL = await getDownloadURL(snapshot.ref);

        return downloadURL;
    } catch (error) {
        console.error('Error uploading profile picture:', error);
        throw error;
    }
}

// ====== GROUP CHAT FUNCTIONS ======

function generateInviteCode() {
    const chars = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';
    let code = '';
    for (let i = 0; i < 6; i++) {
        code += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return code;
}

/**
 * 🚨 EMERGENCY FIX: Create group with EXACT schema
 */
export async function createGroupChat(creatorId, groupData) {
    try {
        let inviteCode = generateInviteCode();
        let isUnique = false;
        let attempts = 0;

        while (!isUnique && attempts < 10) {
            const existingQuery = query(
                collection(db, 'chats'),
                where('inviteCode', '==', inviteCode),
                limit(1)
            );
            const existingSnapshot = await getDocs(existingQuery);

            if (existingSnapshot.empty) {
                isUnique = true;
            } else {
                inviteCode = generateInviteCode();
                attempts++;
            }
        }

        if (!isUnique) {
            throw new Error('Failed to generate unique invite code');
        }

        const profilePictureUrl = groupData.profilePictureUrl ||
            `https://ui-avatars.com/api/?name=${encodeURIComponent(groupData.name)}&background=25a959&color=fff&size=200&bold=true`;

        const groupRef = await addDoc(collection(db, 'chats'), {
            name: groupData.name,
            description: groupData.description || '',
            profilePictureUrl,
            type: 'group',  // 🔥 EXACT: lowercase 'group'
            isGroup: true,
            creatorId,
            participants: [creatorId],
            members: [creatorId],
            admins: [creatorId],
            adminOnlyMessaging: groupData.adminOnlyMessaging || false,
            inviteCode,
            maxMembers: 500,
            lastMessage: '',  // 🔥 NEW: For preview system
            lastMessageSenderId: '',
            createdAt: serverTimestamp(),
            updatedAt: serverTimestamp()  // 🔥 EXACT: updatedAt (NOT lastActivity)
        });

        return {
            id: groupRef.id,
            inviteCode,
            profilePictureUrl,
            ...groupData
        };
    } catch (error) {
        console.error('❌ Group creation error:', error);
        throw error;
    }
}

export async function joinGroupByInviteCode(inviteCode, userId) {
    try {
        const groupQuery = query(
            collection(db, 'chats'),
            where('inviteCode', '==', inviteCode.toUpperCase()),
            where('isGroup', '==', true),
            limit(1)
        );

        const groupSnapshot = await getDocs(groupQuery);

        if (groupSnapshot.empty) {
            throw new Error('Invalid invite code');
        }

        const groupDoc = groupSnapshot.docs[0];
        const groupData = groupDoc.data();
        const groupId = groupDoc.id;

        if (groupData.participants?.includes(userId)) {
            return {
                id: groupId,
                ...groupData,
                alreadyMember: true
            };
        }

        if (groupData.participants?.length >= (groupData.maxMembers || 500)) {
            throw new Error('Group has reached maximum member limit');
        }

        const groupRef = doc(db, 'chats', groupId);
        await updateDoc(groupRef, {
            participants: [...(groupData.participants || []), userId],
            members: [...(groupData.members || []), userId],
            updatedAt: serverTimestamp()  // 🔥 EXACT: updatedAt
        });

        return {
            id: groupId,
            ...groupData,
            alreadyMember: false
        };
    } catch (error) {
        console.error('❌ Join group error:', error);
        throw error;
    }
}

export async function getGroupByInviteCode(inviteCode) {
    try {
        const groupQuery = query(
            collection(db, 'chats'),
            where('inviteCode', '==', inviteCode.toUpperCase()),
            where('isGroup', '==', true),
            limit(1)
        );

        const groupSnapshot = await getDocs(groupQuery);

        if (groupSnapshot.empty) {
            return null;
        }

        const groupDoc = groupSnapshot.docs[0];
        const groupData = groupDoc.data();

        return {
            id: groupDoc.id,
            name: groupData.name,
            description: groupData.description,
            profilePictureUrl: groupData.profilePictureUrl,
            memberCount: groupData.participants?.length || 0,
            maxMembers: groupData.maxMembers || 500
        };
    } catch (error) {
        console.error('Error fetching group:', error);
        throw error;
    }
}

// ====== MISSING FUNCTION FIX ======

/**
 * Uploads a group profile picture to Firebase Storage
 */
export async function uploadGroupPicture(chatId, file) {
    try {
        if (!file.type.startsWith('image/')) {
            throw new Error('File must be an image');
        }

        const fileExtension = file.name.split('.').pop();
        const fileName = `${chatId}_${Date.now()}.${fileExtension}`;
        const storageRef = ref(storage, `group_avatars/${fileName}`);

        const snapshot = await uploadBytes(storageRef, file);
        const downloadURL = await getDownloadURL(snapshot.ref);

        // Update the chat document with the new URL
        await updateDoc(doc(db, 'chats', chatId), {
            profilePictureUrl: downloadURL,
            updatedAt: serverTimestamp()
        });

        return downloadURL;
    } catch (error) {
        console.error('Error uploading group picture:', error);
        throw error;
    }
}