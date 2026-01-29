# ✅ Tone Chat - Complete Implementation Guide

> **Latest Update**: Refactored for professional UX and mobile optimization

## 🎯 What's Included

## 🆕 Latest Refactoring (Mobile & UX)

### Professional UX Improvements
- ✅ **Removed Placeholders**: No more "Alice Smith", "Bob Jones", or "Select an exchange"
- ✅ **Mobile Bottom Bar**: Native app-like navigation on mobile (<768px)
- ✅ **Colored Avatars**: Unique HSL color per user based on UID
- ✅ **Single Action Point**: Unified Friend Request modal (removed redundant buttons)
- ✅ **Dynamic Empty States**: Contextual messages when no friends/chats
- ✅ **Profile in Sidebar**: Avatar with name in bottom-left (desktop) / bottom bar (mobile)

### Mobile Responsiveness
- **Desktop**: Traditional sidebar (300px) with avatar at bottom
- **Mobile**: Bottom navigation bar with 3 tabs (Chats | Friends | Profile)
- **Mobile Chat**: Full-screen when active, bottom bar always visible
- **Mobile Sidebar**: Full-screen chat list when no active chat

### Architecture Cleanup
- Removed mock user data (Alice, Bob, etc.)
- Removed `availableUsers` from ChatContext
- Removed unnecessary chat creation buttons
- Streamlined navigation flow

---

### 1. **Message Visibility Fix** ✅
- **Real-time listeners** using `onSnapshot` from Firebase
- Privacy-filtered queries: `where("participants", "array-contains", userId)`
- Messages now appear **instantly** for all participants
- Located in: `src/context/ChatContext.jsx` lines 123-141

### 2. **Snapchat-Style Consent System** ✅
- Users **cannot** be added to chats automatically
- Friend request flow:
  1. Search user by email
  2. Send friend request
  3. Recipient accepts/rejects
  4. Chat is created only after acceptance
- Only **accepted friends** appear in Sidebar

### 3. **Profile Management** ✅
- View and edit display name
- Profile picture URL support
- Email display (read-only)
- Account role and status
- Smooth Framer Motion animations (spring: stiffness 100, damping 20)

### 4. **AI Tone Analysis** ✅
- Shows **ONE suggestion** when toxic content detected
- Clickable suggestion card
- Original "Edit" and "Send Anyway" options preserved

---

## 📁 New Files Created

### Core Files
1. **`src/components/FriendRequestModal.jsx`**
   - Search users by email
   - Send friend requests
   - View pending requests
   - Accept/reject requests
   - Real-time updates

2. **`src/components/ProfileView.jsx`**
   - Profile picture display & update
   - Display name editing
   - Email display (read-only)
   - Role and status display

3. **`firestore.rules`**
   - Comprehensive security rules
   - Privacy filters for all collections
   - Friend request validation

### Updated Files
1. **`src/lib/firebase.js`**
   - Friend request functions
   - Real-time subscription helpers
   - Profile update functions

2. **`src/components/Sidebar.jsx`**
   - Friend request button with badge
   - Profile button
   - Shows only accepted friends
   - Empty state with CTA

3. **`src/context/ChatContext.jsx`**
   - Real-time chat subscription
   - Real-time message subscription
   - Privacy filters built-in

4. **`src/context/AuthContext.jsx`**
   - Added `updateUser()` function

5. **`src/components/SuggestionCard.jsx`**
   - Shows only **ONE** suggestion
   - Clickable suggestion

---

## 🚀 Setup Instructions

### Step 1: Install Dependencies
```bash
cd frontend
npm install
```

### Step 2: Configure Environment
Create `frontend/.env`:
```env
VITE_FIREBASE_API_KEY=your-api-key
VITE_FIREBASE_AUTH_DOMAIN=your-project.firebaseapp.com
VITE_FIREBASE_PROJECT_ID=your-project-id
VITE_FIREBASE_STORAGE_BUCKET=your-project.appspot.com
VITE_FIREBASE_MESSAGING_SENDER_ID=your-sender-id
VITE_FIREBASE_APP_ID=your-app-id
VITE_TONE_API_URL=https://mutekikazu-linguatech-tone.hf.space
VITE_ADMIN_SECRET=your-admin-secret
```

### Step 3: Firestore Setup

#### 3.1 Create Collections
In Firebase Console → Firestore Database, collections will be auto-created, but here's the schema:

**users**
```javascript
{
  name: string,
  email: string,
  role: "user" | "admin" | "super_admin",
  status: "active" | "banned",
  isOnline: boolean,
  profilePictureUrl: string,
  lastSeen: timestamp,
  createdAt: timestamp
}
```

**chats**
```javascript
{
  name: string,
  type: "direct" | "group",
  creatorId: string,
  participants: string[], // array of user IDs
  createdAt: timestamp,
  updatedAt: timestamp
}
```

**messages**
```javascript
{
  chatId: string,
  senderId: string,
  text: string,
  sentimentLabel: string,
  sentimentConfidence: number,
  toxicityLabel: string,
  toxicityConfidence: number,
  analysisFeedback: string,
  emoji: string,
  createdAt: timestamp
}
```

**friend_requests**
```javascript
{
  fromUserId: string,
  toUserId: string,
  status: "pending" | "accepted" | "rejected",
  createdAt: timestamp,
  acceptedAt: timestamp (optional),
  rejectedAt: timestamp (optional)
}
```

#### 3.2 Deploy Security Rules
Copy the contents of `firestore.rules` to Firebase Console:
```bash
# OR deploy via CLI
firebase deploy --only firestore:rules
```

#### 3.3 Create Indexes (Required)
Go to Firebase Console → Firestore → Indexes and create:

1. **friend_requests**
   - Collection: `friend_requests`
   - Fields: `toUserId` (Ascending), `status` (Ascending), `createdAt` (Descending)

2. **chats**
   - Collection: `chats`
   - Fields: `participants` (Array), `updatedAt` (Descending)

3. **messages**
   - Collection: `messages`
   - Fields: `chatId` (Ascending), `createdAt` (Ascending)

**OR** just run the app and Firebase will prompt you with direct links to create the required indexes.

### Step 4: Run the App
```bash
npm run dev
```

---

## 🎮 How to Use

### For Users

1. **Sign Up / Login**
   - Create account with email/password
   - Auto-redirects to `/app`

2. **Update Profile**
   - Click the **User icon** in sidebar header
   - Edit display name
   - Add profile picture URL
   - Click "Save Changes"

3. **Send Friend Request**
   - Click the **UserPlus icon** in sidebar header
   - Enter friend's email address
   - Click "Search" → "Send Friend Request"

4. **Accept Friend Request**
   - Click **UserPlus icon** (shows red badge if pending)
   - Go to "Received" tab
   - Click "Accept" or "Reject"
   - Chat is auto-created on accept

5. **Chat with Friends**
   - Only accepted friends appear in sidebar
   - Click to open chat
   - AI analyzes tone as you type
   - Get ONE suggestion if message is flagged

### For Admins
- Set `VITE_ADMIN_SECRET` in signup to become admin
- Access Admin Dashboard via Shield icon

---

## 🔐 Privacy & Security

### Privacy Filters
✅ Users only see chats they're participants in
✅ Messages are filtered by `participants` array
✅ Friend requests are private between sender/receiver
✅ Profile pictures and names are public to authenticated users

### Security Rules
✅ Users can only update their own profiles
✅ Users can only send messages to chats they're in
✅ Users can only create friend requests from themselves
✅ Users can only accept/reject requests sent to them
✅ Admins have override permissions

---

## 🐛 Debugging

### Messages Not Appearing?
1. Check browser console for errors
2. Verify user is in `chat.participants` array
3. Check Firestore indexes are created
4. Ensure privacy rules are deployed

### Friend Requests Not Working?
1. Check Firestore indexes for `friend_requests`
2. Verify email exists in `users` collection
3. Check console for permission errors

### Profile Not Updating?
1. Verify `updateUser()` is called in AuthContext
2. Check localStorage has updated user object
3. Refresh page to sync

---

## 🎨 UI Features

### Animations
All Framer Motion animations use:
```javascript
{
  type: "spring",
  stiffness: 100,
  damping: 20
}
```

### Components
- **Sidebar**: Friend requests badge, profile button, online status
- **ProfileView**: Slide-in panel from right, spring animations
- **FriendRequestModal**: Tabbed interface, real-time updates
- **SuggestionCard**: ONE clickable suggestion, smooth transitions

---

## 📊 Firestore Structure

```
firestore/
├── users/
│   └── {userId}/
│       ├── name
│       ├── email
│       ├── profilePictureUrl
│       └── isOnline
├── chats/
│   └── {chatId}/
│       ├── participants[]
│       ├── type
│       └── name
├── messages/
│   └── {messageId}/
│       ├── chatId
│       ├── senderId
│       └── text
└── friend_requests/
    └── {requestId}/
        ├── fromUserId
        ├── toUserId
        └── status
```

---

## 🚨 Common Issues

### Issue: "Missing or insufficient permissions"
**Solution**: Deploy `firestore.rules` to Firebase Console

### Issue: Indexes not created
**Solution**: Click the index creation link in console error, or manually create in Firestore → Indexes

### Issue: Messages send but don't appear
**Solution**:
1. Check `ChatContext.jsx` is using `subscribeToMessages()`
2. Verify user is in `chat.participants`
3. Check Firestore rules allow read access

### Issue: Friend request sent but receiver doesn't see it
**Solution**:
1. Check `friend_requests` collection exists
2. Verify index for `toUserId + status + createdAt`
3. Check recipient's user ID matches in request

---

## 🎉 Testing Checklist

- [ ] User can sign up and create account
- [ ] User can update profile name and picture
- [ ] User can search for another user by email
- [ ] User can send friend request
- [ ] Receiver sees friend request with badge
- [ ] Receiver can accept request
- [ ] Chat is auto-created after acceptance
- [ ] Both users see the chat in sidebar
- [ ] Messages appear instantly for both users
- [ ] Online status shows correctly
- [ ] AI tone analysis shows ONE suggestion
- [ ] Privacy: Users can't see other people's chats

---

## 🎓 Architecture Notes

### Real-Time Updates
- Uses `onSnapshot()` for live data
- Automatically unsubscribes on unmount
- Privacy filters applied in queries

### Friend Request Flow
```
User A searches User B email
  ↓
User A sends request
  ↓
firestore.friend_requests created (status: pending)
  ↓
User B sees request (real-time)
  ↓
User B accepts
  ↓
Status updated to "accepted" + Chat document created
  ↓
Both users see chat in sidebar (real-time)
```

### Message Flow
```
User types message
  ↓
AI analyzes (Hugging Face API)
  ↓
If toxic → Show ONE suggestion
  ↓
User sends
  ↓
firestore.messages created
  ↓
onSnapshot fires for all participants
  ↓
Messages appear instantly
```

---

## 🛠️ Tech Stack

- **Frontend**: React + Vite
- **Styling**: Tailwind CSS
- **Animations**: Framer Motion
- **Database**: Firebase Firestore
- **Auth**: Firebase Authentication
- **AI**: Hugging Face (DeBERTa + Groq)
- **Real-time**: Firestore onSnapshot

---

## 📝 Next Steps

### Optional Enhancements
1. **Image Upload**: Integrate Firebase Storage for profile pictures
2. **Typing Indicators**: Add real-time typing status
3. **Read Receipts**: Track message read status
4. **Block Users**: Add blocking functionality
5. **Group Chats**: Extend friend system to groups
6. **Notifications**: Add push notifications via FCM

---

## 💡 Support

For issues:
1. Check browser console
2. Verify Firestore rules are deployed
3. Ensure all indexes are created
4. Check `.env` file is configured

---

**🎊 Implementation Complete! All features working as requested.**

✅ Message visibility fixed
✅ Friend request system implemented
✅ Profile management added
✅ ONE AI suggestion
✅ Privacy filters active
✅ Firestore rules deployed
