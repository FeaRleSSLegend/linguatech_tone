/**
 * Firebase Configuration & Initialization
 * Firebase v10+ Modular SDK
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

if (!firebaseConfig.apiKey || !firebaseConfig.projectId) {
    console.error(
        '❌ Missing Firebase credentials. Please set:\n' +
        '   VITE_FIREBASE_API_KEY\n' +
        '   VITE_FIREBASE_AUTH_DOMAIN\n' +
        '   VITE_FIREBASE_PROJECT_ID\n' +
        '   VITE_FIREBASE_STORAGE_BUCKET\n' +
        '   VITE_FIREBASE_MESSAGING_SENDER_ID\n' +
        '   VITE_FIREBASE_APP_ID\n' +
        '   in your .env file'
    );
}

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

/**
 * Search for users by email
 */
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

/**
 * Send a friend request
 */
export async function sendFriendRequest(fromUserId, toUserId) {
    try {
        // Prevent self-friend requests
        if (fromUserId === toUserId) {
            throw new Error('Cannot send friend request to yourself');
        }

        // Check if request already exists
        const existingQuery = query(
            collection(db, 'friend_requests'),
            where('fromUserId', '==', fromUserId),
            where('toUserId', '==', toUserId)
        );
        const existingSnapshot = await getDocs(existingQuery);

        if (!existingSnapshot.empty) {
            throw new Error('Friend request already sent');
        }

        // Check reverse direction
        const reverseQuery = query(
            collection(db, 'friend_requests'),
            where('fromUserId', '==', toUserId),
            where('toUserId', '==', fromUserId)
        );
        const reverseSnapshot = await getDocs(reverseQuery);

        if (!reverseSnapshot.empty) {
            throw new Error('This user already sent you a request');
        }

        // Create friend request
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

/**
 * Get pending friend requests for a user
 */
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
 * Accept a friend request and create a direct chat (ATOMIC using writeBatch)
 */
export async function acceptFriendRequest(requestId, fromUserId, toUserId) {
    try {
        // Fetch user profiles first (read operations must be done before batch)
        const fromUserProfile = await getUserProfile(fromUserId);
        const toUserProfile = await getUserProfile(toUserId);

        if (!fromUserProfile || !toUserProfile) {
            throw new Error('User profiles not found');
        }

        // Create a batch for atomic operations
        const batch = writeBatch(db);

        // 1. Update friend request status
        const requestRef = doc(db, 'friend_requests', requestId);
        batch.update(requestRef, {
            status: 'accepted',
            acceptedAt: serverTimestamp()
        });

        // 2. Create direct chat
        const chatRef = doc(collection(db, 'chats'));
        const chatData = {
            name: `${fromUserProfile.name} & ${toUserProfile.name}`,
            type: 'direct',
            isGroup: false,
            creatorId: toUserId,
            participants: [fromUserId, toUserId],
            participantNames: {
                [fromUserId]: fromUserProfile.name,
                [toUserId]: toUserProfile.name
            },
            createdAt: serverTimestamp(),
            updatedAt: serverTimestamp(),
            lastActivity: serverTimestamp()
        };
        batch.set(chatRef, chatData);

        // Commit the batch atomically
        await batch.commit();

        console.log('✅ Friend request accepted and chat created atomically');

        return {
            chatId: chatRef.id,
            ...chatData
        };
    } catch (error) {
        console.error('❌ Error accepting friend request:', error);
        throw error;
    }
}

/**
 * Reject a friend request
 */
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

/**
 * Check if users are already friends (have an accepted request)
 */
export async function areUsersFriends(userId1, userId2) {
    try {
        const query1 = query(
            collection(db, 'friend_requests'),
            where('fromUserId', '==', userId1),
            where('toUserId', '==', userId2),
            where('status', '==', 'accepted')
        );

        const query2 = query(
            collection(db, 'friend_requests'),
            where('fromUserId', '==', userId2),
            where('toUserId', '==', userId1),
            where('status', '==', 'accepted')
        );

        const [snapshot1, snapshot2] = await Promise.all([
            getDocs(query1),
            getDocs(query2)
        ]);

        return !snapshot1.empty || !snapshot2.empty;
    } catch (error) {
        console.error('Error checking friendship:', error);
        return false;
    }
}

/**
 * Subscribe to friend requests in real-time
 */
export function subscribeToFriendRequests(userId, callback) {
    console.log('🔍 [Friend Requests] Subscribing for userId:', userId);

    const requestsQuery = query(
        collection(db, 'friend_requests'),
        where('toUserId', '==', userId),
        where('status', '==', 'pending'),
        orderBy('createdAt', 'desc')
    );

    return onSnapshot(requestsQuery, async (snapshot) => {
        console.log('📬 [Friend Requests] Snapshot received:', {
            userId,
            documentCount: snapshot.docs.length,
            requestIds: snapshot.docs.map(d => d.id)
        });

        const requests = [];

        for (const docSnap of snapshot.docs) {
            const requestData = docSnap.data();
            console.log('📧 [Friend Request] Processing:', {
                id: docSnap.id,
                fromUserId: requestData.fromUserId,
                toUserId: requestData.toUserId,
                status: requestData.status
            });

            const fromUser = await getUserProfile(requestData.fromUserId);

            if (!fromUser) {
                console.warn('⚠️ [Friend Request] Sender profile not found:', requestData.fromUserId);
            }

            requests.push({
                id: docSnap.id,
                ...requestData,
                fromUser
            });
        }

        console.log('✅ [Friend Requests] Final list:', requests.length, 'requests');
        callback(requests);
    }, (error) => {
        console.error('❌ [Friend Requests] Listener error:', error);
    });
}

/**
 * Subscribe to chats in real-time (with privacy filter)
 */
export function subscribeToChats(userId, callback) {
    console.log('💬 [Chats] Subscribing to chats for userId:', userId);

    const chatsQuery = query(
        collection(db, 'chats'),
        where('participants', 'array-contains', userId),
        orderBy('updatedAt', 'desc')
    );

    return onSnapshot(chatsQuery, (snapshot) => {
        console.log('💬 [Chats] Snapshot received:', {
            userId,
            documentCount: snapshot.docs.length,
            chatIds: snapshot.docs.map(d => d.id)
        });

        const chats = snapshot.docs.map(doc => {
            const data = doc.data();
            console.log('💬 [Chat]', {
                id: doc.id,
                name: data.name,
                type: data.type,
                isGroup: data.isGroup,
                participants: data.participants
            });
            return {
                id: doc.id,
                ...data
            };
        });

        callback(chats);
    }, (error) => {
        console.error('❌ [Chats] Listener error:', error);
    });
}

/**
 * Subscribe to messages for a specific chat (with privacy filter)
 */
export function subscribeToMessages(chatId, userId, callback) {
    return onSnapshot(doc(db, 'chats', chatId), async (chatSnap) => {
        if (!chatSnap.exists()) {
            callback([]);
            return;
        }

        const chatData = chatSnap.data();
        if (!chatData.participants?.includes(userId)) {
            console.warn('Privacy Filter: User not authorized to view messages');
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
                    timestamp: msgData.createdAt?.toDate()
                });
            }

            callback(messages);
        });
    });
}

// ====== FIREBASE STORAGE FUNCTIONS ======

/**
 * Upload profile picture to Firebase Storage
 * @param {string} userId - User ID
 * @param {File} file - Image file to upload
 * @returns {Promise<string>} - Download URL of uploaded image
 */
export async function uploadProfilePicture(userId, file) {
    try {
        // Validate file type
        if (!file.type.startsWith('image/')) {
            throw new Error('File must be an image');
        }

        // Validate file size (5MB max)
        const maxSize = 5 * 1024 * 1024;
        if (file.size > maxSize) {
            throw new Error('File size must be less than 5MB');
        }

        // Create storage reference
        const fileExtension = file.name.split('.').pop();
        const fileName = `${userId}.${fileExtension}`;
        const storageRef = ref(storage, `avatars/${fileName}`);

        // Upload file
        const snapshot = await uploadBytes(storageRef, file);
        console.log('Upload successful:', snapshot);

        // Get download URL
        const downloadURL = await getDownloadURL(snapshot.ref);
        console.log('Download URL:', downloadURL);

        return downloadURL;
    } catch (error) {
        console.error('Error uploading profile picture:', error);
        throw error;
    }
}

// ====== GROUP CHAT FUNCTIONS ======

/**
 * Generate a unique 6-character invite code
 */
function generateInviteCode() {
    const chars = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'; // Removed ambiguous chars
    let code = '';
    for (let i = 0; i < 6; i++) {
        code += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return code;
}

/**
 * Create a new group chat
 * @param {string} creatorId - User ID of the creator
 * @param {Object} groupData - Group information
 * @returns {Promise<Object>} - Created group with ID and invite code
 */
export async function createGroupChat(creatorId, groupData) {
    try {
        console.log('🏗️ [Group Chat] Creating group:', groupData.name);

        // Generate unique invite code
        let inviteCode = generateInviteCode();
        let isUnique = false;
        let attempts = 0;

        // Ensure invite code is unique
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

        // Use placeholder avatar if no profilePictureUrl provided or if it's empty
        const profilePictureUrl = groupData.profilePictureUrl ||
            `https://ui-avatars.com/api/?name=${encodeURIComponent(groupData.name)}&background=25a959&color=fff&size=200&bold=true`;

        // Create group document
        const groupRef = await addDoc(collection(db, 'chats'), {
            name: groupData.name,
            description: groupData.description || '',
            profilePictureUrl,
            type: 'group',
            isGroup: true,
            creatorId,
            participants: [creatorId], // Start with creator
            members: [creatorId], // For explicit member tracking
            admins: [creatorId], // Creator is admin
            adminOnlyMessaging: groupData.adminOnlyMessaging || false,
            inviteCode,
            maxMembers: 500,
            createdAt: serverTimestamp(),
            updatedAt: serverTimestamp(),
            lastActivity: serverTimestamp()
        });

        console.log('✅ [Group Chat] Group created:', {
            id: groupRef.id,
            inviteCode,
            name: groupData.name,
            profilePictureUrl
        });

        return {
            id: groupRef.id,
            inviteCode,
            profilePictureUrl,
            ...groupData
        };
    } catch (error) {
        console.error('❌ [Group Chat] Creation error:', error);
        throw error;
    }
}

/**
 * Join a group using an invite code
 * @param {string} inviteCode - 6-character invite code
 * @param {string} userId - User ID joining the group
 * @returns {Promise<Object>} - Group information
 */
export async function joinGroupByInviteCode(inviteCode, userId) {
    try {
        console.log('🚪 [Group Chat] Joining with code:', inviteCode.toUpperCase());

        // Find group by invite code
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

        // Check if already a member
        if (groupData.participants?.includes(userId)) {
            console.log('ℹ️ [Group Chat] User already a member');
            return {
                id: groupId,
                ...groupData,
                alreadyMember: true
            };
        }

        // Check member limit
        if (groupData.participants?.length >= (groupData.maxMembers || 500)) {
            throw new Error('Group has reached maximum member limit (500)');
        }

        // Add user to group
        const groupRef = doc(db, 'chats', groupId);
        await updateDoc(groupRef, {
            participants: [...(groupData.participants || []), userId],
            members: [...(groupData.members || []), userId],
            updatedAt: serverTimestamp()
        });

        console.log('✅ [Group Chat] User joined successfully:', {
            groupId,
            groupName: groupData.name,
            userId
        });

        return {
            id: groupId,
            ...groupData,
            alreadyMember: false
        };
    } catch (error) {
        console.error('❌ [Group Chat] Join error:', error);
        throw error;
    }
}

/**
 * Get group information by invite code
 * @param {string} inviteCode - 6-character invite code
 * @returns {Promise<Object>} - Group information (public data only)
 */
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
        console.error('Error fetching group by invite code:', error);
        throw error;
    }
}

/**
 * Upload group profile picture
 * @param {string} groupId - Group ID
 * @param {File} file - Image file
 * @returns {Promise<string>} - Download URL
 */
export async function uploadGroupPicture(groupId, file) {
    try {
        if (!file.type.startsWith('image/')) {
            throw new Error('File must be an image');
        }

        const maxSize = 5 * 1024 * 1024;
        if (file.size > maxSize) {
            throw new Error('File size must be less than 5MB');
        }

        const fileExtension = file.name.split('.').pop();
        const fileName = `${groupId}.${fileExtension}`;
        const storageRef = ref(storage, `groups/${fileName}`);

        const snapshot = await uploadBytes(storageRef, file);
        const downloadURL = await getDownloadURL(snapshot.ref);

        console.log('✅ [Group Chat] Profile picture uploaded:', downloadURL);

        return downloadURL;
    } catch (error) {
        console.error('❌ [Group Chat] Upload error:', error);
        throw error;
    }
}
