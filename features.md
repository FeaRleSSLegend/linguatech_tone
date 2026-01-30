# Tone Chat - Features Documentation

## Implementation Philosophy: Permission & UI Lockdown

The block system uses a **defense-in-depth** approach with THREE layers of protection:

### Layer 1: UI Prevention (ChatInterface)
- Input field is **disabled** when block is active
- Send button is **disabled** and grayed out
- Dynamic placeholders inform users WHY they can't send
- No way to type or click send

### Layer 2: Context Prevention (ChatContext)
- `sendMessage` function performs **bidirectional block check**
- If `blockedBy` array has ANY entries, function immediately returns
- Error is thrown before ANY API call is made
- Prevents programmatic bypass of UI

### Layer 3: Database Prevention (Firestore Rules)
- `chatHasActiveBlock()` helper function checks `blockedBy.size() > 0`
- Message creation is **rejected at database level**
- Prevents bypassing via direct Firestore SDK calls or Firebase console
- Ensures data integrity even if UI/Context are compromised

**Result:** It is **impossible** to send a message in a blocked chat, regardless of how the attempt is made.

---

## Implemented Features

### 1. Admin Management Dashboard (Group Moderation)
**Location:** GroupAdminDashboard.jsx, api.js, ChatContext.jsx, firestore.rules

**Functionality:**
- Comprehensive admin dashboard for group chat moderation and management
- AI-powered automatic moderation with toxicity tracking
- Three-tab interface: Members, Moderation, and Settings
- Real-time toxicity leaderboard showing top offenders
- Word blacklist system for manual content filtering
- Automatic temporary bans for repeated toxic behavior
- Group identity management (name, description, avatar)

**Access Control:**
- Only the group creator can access the Admin Dashboard
- Only admins (creator + designated admins) can perform moderation actions
- Firestore rules enforce these permissions at the database level

**How to Use:**

**Access the Dashboard:**
1. Open a group chat where you are the creator
2. Click the Shield icon next to the group name in the chat header
3. The Admin Dashboard modal will open

**Tab 1: Members (Member Management)**
- View all group members with their toxicity counts
- See member roles (Creator, Admin, Member)
- Actions available:
  - **Reset Toxicity**: Clear a member's toxic message count
  - **Kick**: Remove member from group (they can rejoin with invite)
  - **Ban**: Permanently ban member (cannot rejoin)

**Tab 2: Moderation**
- **Leaderboard of Shame**: Top 5 members with most toxic messages
- **Word Blacklist**: Add/remove forbidden words
  - Any message containing blacklisted words is auto-rejected
  - Works independently of Strict Mode

**Tab 3: Settings**
- **Group Name**: Change the display name
- **Description**: Update group description
- **Avatar URL**: Set group profile picture
- **Strict Mode Toggle**:
  - **ON**: AI Auto-Mod actively detects and penalizes toxic messages
  - **OFF**: Only manual word blacklist is enforced

**Technical Implementation:**

**Schema Updates (Chats Collection):**
```javascript
{
  // ... existing fields ...

  // Moderation Stats
  moderationStats: {
    "userId1": 2,  // Number of toxic messages sent
    "userId2": 5,
    "userId3": 1
  },

  // Automatic Temporary Bans
  tempBannedUsers: [
    {
      userId: "userId2",
      bannedUntil: 1738234567890  // Timestamp (2 hours from ban)
    }
  ],

  // Permanent Bans
  permBannedUsers: ["userId4", "userId5"],

  // Word Blacklist
  wordBlacklist: ["spam", "scam", "offensive-word"],

  // AI Auto-Mod Toggle
  strictMode: true
}
```

**AI Auto-Moderation Flow:**
1. User sends message
2. Message is analyzed for toxicity (via existing Tone AI integration)
3. **If toxic and Strict Mode is ON:**
   - Increment user's `moderationStats` count
   - If count reaches 3: Auto temp-ban for 2 hours
   - Message is rejected with warning
4. **If contains blacklisted word:**
   - Message is rejected immediately (regardless of Strict Mode)
5. **If user is temp banned:**
   - All messages blocked until ban expires
   - UI shows: "You are temporarily banned for toxic behavior. Remaining time: [X] minutes"

**Delete for Everyone (Admin Power):**
- Admins can delete ANY message in their group
- Appears as a trash icon on message hover (admin view only)
- Permanently removes message from Firestore
- All users see the message disappear

**Firestore Security Enforcement:**
```javascript
// Only admins can update moderation fields
allow update: if isGroupAdmin() && affectedKeys().hasOnly([
  'moderationStats', 'tempBannedUsers', 'permBannedUsers', 'wordBlacklist'
]);

// Only creator can update group settings
allow update: if isGroupCreator() && affectedKeys().hasOnly([
  'name', 'description', 'profilePictureUrl', 'strictMode'
]);

// Admins can delete any message in their group
allow delete: if isMessageGroupAdmin();
```

**Automated Penalty System:**
- **1st toxic message**: Warning + count incremented
- **2nd toxic message**: Warning + count incremented
- **3rd toxic message**: **AUTO TEMP-BAN for 2 hours**
- Ban timer shown in UI
- After ban expires, user can send messages again (count persists)

**Top 5 Offenders Algorithm:**
```javascript
const topOffenders = members
  .sort((a, b) => b.toxicCount - a.toxicCount)
  .slice(0, 5)
  .filter(m => m.toxicCount > 0);
```

---

### 2. Conversation Reset (Delete Chat)
**Location:** ChatContext.jsx, ChatInterface.jsx

**Functionality:**
- Users can reset a conversation, which deletes all message history
- The chat document remains in Firestore to maintain searchability
- When reset, the `lastMessage` and `updatedAt` fields are cleared, making the chat disappear from the sidebar
- If a new message is sent to the same user later, the chat will reappear with no previous history
- Only participants can reset their own chats

**How to Use:**
1. Open a chat conversation
2. Click the three-dot menu (MoreVertical icon) in the header
3. Select "Reset Chat"
4. Confirm the action

**Technical Implementation:**
- Deletes all documents in the `messages` collection for that `chatId` using a batch write
- Updates the chat document to clear `lastMessage`, `lastMessageSenderId`, and `updatedAt`
- If the chat is currently active, it closes the chat view

---

### 2. Block/Unblock System (Permission & UI Lockdown)
**Location:** ChatContext.jsx, ChatInterface.jsx, Sidebar.jsx, firestore.rules

**Functionality:**
- Users can block other users in direct chats
- **HARD RESTRICTION**: Messages are blocked at BOTH the UI and database levels
- When blocked, NEITHER user can send messages (enforced by Firestore rules)
- The system detects blocks from BOTH directions (I blocked them OR they blocked me)
- Input field is disabled with dynamic placeholders based on who initiated the block
- No typing indicators or online status shown when blocked

**How to Use:**

**To Block:**
1. Open a direct chat
2. Click the three-dot menu in the header
3. Select "Block User"
4. Confirm the action

**To Unblock:**
1. Open a blocked chat
2. Click the "Unblock User" button in the three-dot menu
3. Confirm the action

**Technical Implementation:**

**Frontend (UI Lockdown):**
- `sendMessage` function checks `blockedBy` array BEFORE attempting to send
- If ANY user ID exists in `blockedBy`, the function immediately returns with an error
- Input textarea is disabled (`disabled={isBlocked}`)
- Send button is disabled when blocked
- Dynamic placeholders:
  - "You blocked this contact" (if current user blocked)
  - "You were blocked by this contact" (if other user blocked)
- Typing indicators are NOT emitted if chat has active block
- Online status is hidden for blocked chats
- Sidebar shows appropriate message based on who blocked

**Backend (Database Lockdown):**
- Firestore rules contain `chatHasActiveBlock()` function
- Message creation is REJECTED if `blockedBy` array has any entries
- This prevents bypassing the UI restriction via direct API calls

**Bidirectional Block Detection:**
```javascript
const otherUserId = activeChat.participants?.find(id => id !== user.id);
const iBlockedThem = activeChat.blockedBy?.includes(user.id);
const theyBlockedMe = activeChat.blockedBy?.includes(otherUserId);
const isBlocked = iBlockedThem || theyBlockedMe;
```

**Data Schema:**
```javascript
// Chat document structure with blockedBy field
{
  id: "chat123",
  type: "direct",
  participants: ["user1", "user2"],
  blockedBy: ["user1"], // Array of user IDs who initiated a block
  lastMessage: "",
  updatedAt: timestamp,
  ...
}

// Example scenarios:
// 1. User1 blocks User2:
//    blockedBy: ["user1"]
//    Result: BOTH users cannot send messages
//
// 2. User2 also blocks User1:
//    blockedBy: ["user1", "user2"]
//    Result: Still blocked (mutual block)
//
// 3. User1 unblocks User2:
//    blockedBy: ["user2"]
//    Result: Still blocked because User2 hasn't unblocked
```

---

### 3. Enhanced UI States

**Block State Display:**
- **No overlay**: Input field is disabled inline (cleaner UX)
- Message textarea shows dynamic placeholder based on block direction
- Send button is disabled and grayed out
- Sidebar preview shows:
  - "You blocked this contact" (if I blocked them)
  - "You were blocked by this contact" (if they blocked me)
- Menu shows "Unblock User" (green) instead of "Block User" (yellow) when blocked
- Online status indicator hidden for blocked chats
- Typing indicators are NOT sent or received when blocked

**Safety Features:**
- All actions require confirmation dialogs
- Error handling with user-friendly alerts
- Console logging for debugging
- Validation checks before operations

---

## Firestore Security Rules

The following security rules have been implemented in `firestore.rules`:

### Chats Collection
- **Read:** Any authenticated user
- **Create:** Any authenticated user
- **Update:** Participants, creators, or admins
- **Delete:** Participants or admins (for conversation reset)

### Messages Collection
- **Read:** Any authenticated user
- **Create:** Any authenticated user, BUT ONLY IF chat has NO active blocks (`chatHasActiveBlock()` returns false)
- **Update:** Message sender or admins
- **Delete:** Message sender, chat participants (for bulk delete during reset), or admins

### Key Security Features:
- `isParticipant()` helper function validates user is in the chat's `participants` array
- `isMessageParticipant()` helper function checks if user is a participant of the message's chat
- **`chatHasActiveBlock()` helper function checks if `blockedBy` array has any entries**
- **Message creation is HARD BLOCKED at database level if chat has active block**
- Only participants can perform reset operations on their chats
- Admins have override permissions for moderation

### Block Enforcement Flow:
```
User attempts to send message
    ↓
1. UI Check (ChatContext.sendMessage)
   - Checks blockedBy array
   - If blocked: Returns error, prevents API call
    ↓
2. Database Check (Firestore Rules)
   - chatHasActiveBlock(chatId)
   - If blocked: Rejects write operation
    ↓
3. Message created (only if BOTH checks pass)
```

---

## Data Migration for Existing Groups

If you have existing group chats, you may want to initialize the new moderation fields:

**Option 1: Manual via Firebase Console**
1. Go to Firestore Database → `chats` collection
2. For each group chat document, add:
   ```json
   {
     "moderationStats": {},
     "tempBannedUsers": [],
     "permBannedUsers": [],
     "wordBlacklist": [],
     "strictMode": false,
     "admins": []
   }
   ```

**Option 2: Automatic via Code**
The system will automatically create these fields when:
- First toxic message is sent (creates `moderationStats`)
- Admin adds first blacklist word (creates `wordBlacklist`)
- Settings are saved (creates `strictMode`)

**No migration required** - the system gracefully handles missing fields with fallback values.

---

## Manual Steps Required

### 1. Deploy Firestore Security Rules (CRITICAL)
**Action Required:** Deploy the updated `firestore.rules` to Firebase

**Steps:**
1. Open Firebase Console: https://console.firebase.google.com/
2. Select your project
3. Navigate to **Firestore Database** → **Rules** tab
4. Copy the contents of `F:\Team-7\firestore.rules`
5. Paste into the Firebase Rules editor
6. Click **Publish**

**Alternative (CLI):**
```bash
firebase deploy --only firestore:rules
```

---

### 2. Update Existing Chat Documents (Optional)
**Action Required:** Add `blockedBy` field to existing chats

If you have existing chat documents in Firestore that don't have a `blockedBy` field, the system will still work (it checks for existence), but you may want to initialize it for consistency.

**Steps (via Firebase Console):**
1. Go to **Firestore Database** → **Data** tab
2. Navigate to the `chats` collection
3. For each document, add a field:
   - Field name: `blockedBy`
   - Type: `array`
   - Value: `[]` (empty array)

**Alternative (Firestore Batch Update):**
You can run a batch update in your backend or use a Cloud Function to add this field to all existing chats.

---

### 2. Update Group Schema (Optional)
**Action Required:** Add admin designation functionality

The current implementation supports the `admins` array, but you may want to add UI to promote members to admin. For now, only the creator has full admin powers.

To manually add admins via Firebase Console:
```json
{
  "admins": ["userId1", "userId2"]
}
```

---

### 3. Test the Features

**Test Checklist:**

**Admin Dashboard:**
- [ ] Open a group where you're the creator
- [ ] Click Shield icon to open Admin Dashboard
- [ ] Verify all three tabs load correctly
- [ ] Add a word to blacklist
- [ ] Try sending message with blacklisted word (should be rejected)
- [ ] Enable Strict Mode
- [ ] Send a toxic message (should increment count)
- [ ] Send 3 toxic messages (should trigger auto temp-ban)
- [ ] Verify ban message shows remaining time
- [ ] Reset a user's toxicity count
- [ ] Kick a member from group
- [ ] Verify Top 5 Offenders updates in real-time
- [ ] Update group name/description/avatar
- [ ] Save settings and verify changes persist

**Auto-Moderation:**
- [ ] Enable Strict Mode in a group
- [ ] Send toxic message (contains: "fuck", "hate", etc.)
- [ ] Verify toxicity count increments
- [ ] Send 2 more toxic messages
- [ ] Verify auto temp-ban for 2 hours
- [ ] Try to send message while banned (should show ban notice)
- [ ] Wait for ban to expire (or manually remove from Firebase)
- [ ] Verify messaging restored

**Word Blacklist:**
- [ ] Add word to blacklist (e.g., "spam")
- [ ] Try to send message containing "spam"
- [ ] Verify message is rejected with error
- [ ] Remove word from blacklist
- [ ] Verify word can now be sent

**Conversation Reset:**
- [ ] Open a chat with messages
- [ ] Reset the chat via the menu
- [ ] Verify all messages are deleted
- [ ] Verify chat disappears from sidebar
- [ ] Send a new message
- [ ] Verify chat reappears with only the new message

**Block/Unblock:**
- [ ] Block a user in a direct chat
- [ ] Verify "You blocked this user" appears in sidebar
- [ ] Verify input is disabled with unblock button shown
- [ ] Try to send a message (should be prevented)
- [ ] Unblock the user
- [ ] Verify messaging is restored
- [ ] Verify menu shows "Block User" again

**Security Rules:**
- [ ] Try to delete messages from a chat you're not in (should fail)
- [ ] Try to update blockedBy on a chat you're not in (should fail)
- [ ] Verify participants can reset their own chats
- [ ] Verify admins can perform all operations

---

## API Endpoints Used

### ChatContext Functions:
- `deleteChat(chatId)` - Resets conversation by deleting all messages
- `blockUser(chatId, otherUserId)` - Blocks a user in a chat
- `unblockUser(chatId, otherUserId)` - Unblocks a user in a chat
- `sendMessage(text, analysis)` - Sends a message (blocked if chat is blocked)

---

## Future Enhancements

Potential improvements for the block/reset system:

1. **Mutual Block Detection:** Show if the other user has also blocked you
2. **Block from Sidebar:** Allow blocking directly from chat preview
3. **Undo Reset:** Add a 5-second undo period before permanent deletion
4. **Archive Instead of Reset:** Option to archive chats without deleting
5. **Block List Management:** Dedicated settings page to view and manage all blocked users
6. **Notification on Unblock:** Notify when someone unblocks you (optional)

---

## Troubleshooting

### Messages Still Visible After Reset
**Cause:** Browser cache or local state
**Solution:** Refresh the page or clear browser cache

### Permission Denied When Deleting
**Cause:** Firestore rules not deployed
**Solution:** Deploy the rules from `firestore.rules` to Firebase Console

### Block Not Working
**Cause:** `blockedBy` field not initialized
**Solution:** The system will create it automatically on first block attempt

### Cannot Send Messages After Unblock
**Cause:** UI state not refreshed, or other user still has you blocked
**Solution:**
- Close and reopen the chat, or refresh the page
- Check if the OTHER user has also blocked you (mutual block requires both to unblock)

### Messages Still Going Through Despite Block
**Cause:** Firestore rules not deployed or outdated
**Solution:**
- Ensure latest `firestore.rules` are deployed to Firebase Console
- Check Firebase Console → Firestore → Rules tab for `chatHasActiveBlock()` function
- Test the rule by attempting to send via Firestore console directly

### "You were blocked" Showing Incorrectly
**Cause:** `blockedBy` array has incorrect user IDs
**Solution:**
- Check the chat document in Firestore
- Verify `blockedBy` contains the correct user ID who initiated the block
- Ensure `participants` array matches the chat users

### Auto Temp-Ban Not Triggering
**Cause:** Strict Mode is disabled or toxicity count not incrementing
**Solution:**
- Verify Strict Mode is enabled in Group Settings tab
- Check that message contains toxic words (fuck, hate, shit, etc.)
- Verify `moderationStats` field exists in chat document
- Check Firestore rules allow updates to `moderationStats`

### Word Blacklist Not Working
**Cause:** Word not added correctly or case mismatch
**Solution:**
- Words are automatically converted to lowercase
- Ensure word is added via Admin Dashboard (not manually in Firestore)
- Check `wordBlacklist` array in chat document
- Verify message contains the EXACT blacklisted word

### Admin Dashboard Not Opening
**Cause:** User is not the group creator
**Solution:**
- Only the group creator (creatorId) can open the Admin Dashboard
- Check if currentUser.id === groupData.creatorId
- Admins array members currently cannot open dashboard (creator only)

### Toxicity Count Not Showing
**Cause:** `moderationStats` field missing
**Solution:**
- Send a toxic message with Strict Mode enabled to initialize the field
- Or manually add via Firebase Console: `moderationStats: {}`

### Delete for Everyone Not Working
**Cause:** User is not an admin or Firestore rules not deployed
**Solution:**
- Verify user is in `admins` array or is the creator
- Check Firestore rules include `isMessageGroupAdmin()` helper
- Ensure latest rules are deployed to Firebase Console

---

## Code Locations

### GroupAdminDashboard.jsx (NEW)
- Lines 1-600: Complete tabbed modal component
- Lines 50-80: Group data fetching and member details
- Lines 85-95: Top 5 Offenders calculation
- Lines 100-125: Member management actions (kick, ban, reset)
- Lines 130-150: Word blacklist management
- Lines 155-170: Group settings save
- Lines 200-300: Tab 1 - Members UI
- Lines 305-380: Tab 2 - Moderation UI (Leaderboard + Blacklist)
- Lines 385-450: Tab 3 - Settings UI

### api.js (services)
- Lines 385-485: **Updated `sendMessage` with auto-moderation logic**
  - Word blacklist check
  - Toxicity detection
  - Moderation stats tracking
  - Auto temp-ban at 3 toxic messages
- Lines 552-575: `deleteMessage` (Admin delete for everyone)

### ChatContext.jsx
- Lines 1-6: Updated imports with Firestore functions
- Lines 24-60: `deleteChat` function (conversation reset)
- Lines 62-84: `blockUser` function
- Lines 86-99: `unblockUser` function
- Lines 260-307: `handleSendMessage` with **HARD BLOCK CHECK** (bidirectional)
- Lines 309-320: `handleTyping` with block check (prevents typing indicators)

### ChatInterface.jsx
- Lines 118-133: `blockStatus` useMemo with bidirectional check (`iBlockedThem`, `theyBlockedMe`)
- Lines 152-197: Block/unblock/delete handlers
- Lines 371-397: Dynamic Block/Unblock menu button
- Lines 573-623: **NO OVERLAY** - Disabled input with dynamic placeholders

### Sidebar.jsx
- Lines 334-338: Bidirectional block check for each chat
- Line 349: Hide online indicator for blocked chats (either direction)
- Lines 364-367: Dynamic sidebar message based on block direction

### firestore.rules
- Lines 38-75: **ENHANCED Chat collection rules:**
  - Lines 45-48: `isGroupAdmin()` helper (creator or in admins array)
  - Lines 50-53: `isGroupCreator()` helper
  - Lines 62-67: Admin-only moderation field updates
  - Lines 69-73: Owner-only group settings updates
- Lines 77-110: **ENHANCED Message collection rules:**
  - Lines 83-89: `isMessageGroupAdmin()` helper
  - Lines 108-113: Admins can delete any message in their group

**Security Rules Summary:**
```javascript
// ADMIN ONLY
allow update: if isGroupAdmin() && affectedKeys().hasOnly([
  'moderationStats', 'tempBannedUsers', 'permBannedUsers', 'wordBlacklist'
]);

// OWNER ONLY
allow update: if isGroupCreator() && affectedKeys().hasOnly([
  'name', 'description', 'profilePictureUrl', 'strictMode'
]);

// ADMIN DELETE POWER
allow delete: if isMessageGroupAdmin();
```

---

## Version Information

**Last Updated:** 2026-01-29
**Features Version:** 3.0 (Admin Dashboard + Auto-Moderation)
**Compatible With:** Firebase Firestore, React 18+, Framer Motion

**Security Model:** Defense-in-depth (UI + Context + Database)
**Moderation Model:** AI-powered with manual override (Strict Mode + Word Blacklist)

---

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify Firestore rules are deployed correctly
3. Check browser console for error messages
4. Ensure user is authenticated before performing actions

---

**Note:** This documentation covers only the Delete (Reset) and Block/Unblock features. For other features of the Tone Chat application, refer to the respective component documentation.
