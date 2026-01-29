# Tone: AI-Powered Educational Safety Platform

Tone is a premium, mature communication platform designed for modern educational environments. It combines high-fidelity **Real-time AI Resonance Tracking** with a sophisticated user interface to foster respectful and constructive dialogue between students.

---

##  Design Philosophy: Emerald & Sand
Tone utilizes a curated **Emerald Green & Sand** design system. This "Hybrid" aesthetic is crafted to feel both technologically advanced and emotionally grounded—moving away from generic "app" colors to a more balanced, professional atmosphere.

## Core Features

###  For Students
- **Real-time Resonance Shield**: Instant AI feedback on tone and sentiment as you type. A minimal status bar provides a visual "resonance check" (Green, Sand, or Red) before you even hit send.
- **WhatsApp-style Synchronization**: Built on Socket.io for zero-latency messaging. Chat lists re-order dynamically, showing live previews and "Just now" timestamps.
- **Direct & Collaborative Channels**: Secure 1-on-1 exchanges and student-managed group study rooms.
- **Responsive History Search**: Lightning-fast keyword filtering across all message archives.

### For Administrators (Lecturers & Authorities)
- **Authority Dashboard**: A specialized interface for monitoring behavioral patterns and managing community safety.
- **Aggression Tracking**: Automated alerts and reports for "toxic" or "warning" category messages to identify bullying or conflicts early.
- **Room Moderation**: Full control over group participants, including banning and unblocking capabilities based on platform-wide history.

## Support & Tech Stack

### Frontend 
- **React 19** + **Vite**: Ultra-fast HMR and modern rendering.
- **Tailwind CSS 4**: Next-gen styling with a custom Emerald & Sand tokens.
- **Framer Motion**: Fluid, organic UI transitions for a premium feel.

### Backend 
- **Node.js & Express**: Robust API handling.
- **Socket.io**: Full-duplex communication for the WhatsApp-style sync.
- **Heuristic AI Engine**: Custom pattern-recognition system for tone analysis.

---

## Implementation & Launch

### 🔧 Prerequisites

1. **Node.js 18+** installed on your system
2. **Supabase Account** (Free tier works perfectly)
   - Sign up at [https://supabase.com](https://supabase.com)
   - Create a new project
3. **Groq API Key** (Optional, for AI rephrasing in backend)
   - Get free API key at [https://console.groq.com](https://console.groq.com)

---

### 📦 Step 1: Install Dependencies

```bash
# Frontend
cd frontend
npm install @supabase/supabase-js
npm install

# Backend (Python - for Tone AI analysis)
cd ../backend
pip install -r requirements.txt
```

---

### 🗄️ Step 2: Set Up Supabase Database

1. **Go to your Supabase project dashboard**
2. **Navigate to SQL Editor** (left sidebar)
3. **Open the file `SQL_codes.md` in the project root**
4. **Copy and paste each SQL migration** from the file (in order) and run them
5. **Verify tables were created** by going to Table Editor

This will create:
- `profiles` table (users)
- `chats` table (conversations)
- `chat_participants` table (who's in what chat)
- `messages` table (with AI analysis columns)
- `reports` table (for admin moderation)
- **Row Level Security (RLS) policies** for privacy

---

### 🔑 Step 3: Configure Environment Variables

#### Frontend (`frontend/.env`)

Create a `.env` file in the `frontend` directory:

```bash
# Copy the example file
cp .env.example .env
```

Then edit `.env` and add your credentials:

```env
# Get these from: Supabase Dashboard → Settings → API
VITE_SUPABASE_URL=https://your-project-id.supabase.co
VITE_SUPABASE_ANON_KEY=your-anon-key-here

# Tone AI API endpoint
VITE_TONE_API_URL=https://mutekikazu-linguatech-tone.hf.space

# Optional: Admin secret for creating admin users
VITE_ADMIN_SECRET=your-secret-phrase-here
```

#### Backend (`backend/.env`)

Create a `.env` file in the `backend` directory:

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Required: For AI-powered message rephrasing
GROQ_API_KEY=your-groq-api-key-here

# Optional: Server port (defaults to 7860)
PORT=7860
```

---

### 🚀 Step 4: Launch the Application

Run these in **separate terminals**:

#### Terminal 1: Start Backend (Tone AI API)

```bash
cd backend
python app.py
```

You should see:
```
✅ Model loaded successfully!
✅ Groq LLM initialized
🚀 Starting Digital Empathy Assistant API v2.0
🔗 Server running on port 7860
```

#### Terminal 2: Start Frontend

```bash
cd frontend
npm run dev
```

You should see:
```
VITE ready in X ms
➜  Local:   http://localhost:5173/
```

---

### 🔐 Step 5: Create Your First Account

1. Navigate to `http://localhost:5173/`
2. Click **Sign Up**
3. Fill in your details:
   - Name, Email, Password
   - (Optional) Enter admin secret to become an admin
4. Check your email for confirmation link (if email confirmations are enabled in Supabase)
5. Log in and start chatting!

---

## 🛠️ Troubleshooting

### Issue: "Missing Supabase credentials"

**Solution**: Make sure you've created the `.env` file in the `frontend` directory and added your Supabase URL and anon key.

```bash
# Check if .env exists
ls -la frontend/.env

# If not, copy the example
cp frontend/.env.example frontend/.env
```

### Issue: "Model not loaded" in backend

**Solution**: The DeBERTa model is being downloaded from Hugging Face. This can take a few minutes on first run. Wait for the download to complete.

### Issue: "Failed to fetch chats" or "Not authenticated"

**Solution**:
1. Make sure you've run all SQL migrations in Supabase
2. Verify RLS policies are enabled (check Table Editor → Policies tab)
3. Clear localStorage and log in again

```javascript
// In browser console:
localStorage.clear();
location.reload();
```

### Issue: Online status shows "Always Online"

**Solution**: The Supabase Realtime Presence system updates online status. Make sure:
1. You've run the SQL migrations (includes `update_online_status` function)
2. The frontend is calling `updateOnlineStatus()` on login/logout

### Issue: Can't see messages from other users

**Solution**: This is expected! Privacy filtering is working correctly. Messages are only visible if:
- You sent the message, OR
- You're a participant in the chat

To test with multiple users:
1. Open an incognito window
2. Create a second account
3. Start a direct chat between the two users

---

## 🏗️ Architecture Overview

### Privacy & Security

- **Row Level Security (RLS)**: Database-level privacy enforcement
- **User Isolation**: Messages filtered by chat participation
- **Admin Controls**: Role-based access for moderation features
- **Realtime Sync**: Socket.io for live typing indicators and messages

### AI Analysis Pipeline

1. User types a message → debounced after 500ms
2. `useAnalyze` hook calls `/analyze` endpoint
3. **DeBERTa model** analyzes sentiment & toxicity
4. **Groq LLM** generates rephrase suggestions (if needed)
5. Results displayed in real-time above input field
6. Analysis stored with message in database

### Data Flow

```
User Input → useAnalyze Hook → Tone API (DeBERTa + Groq)
                                      ↓
                                  Analysis Result
                                      ↓
ChatInterface → sendMessage() → Supabase (with RLS)
                                      ↓
                            Realtime Broadcast
                                      ↓
                            All Chat Participants
```

---

## 📊 Supabase Dashboard Tips

### View Your Data

- **Table Editor**: See all tables and their contents
- **SQL Editor**: Run custom queries
- **Authentication**: Manage users
- **Logs**: Debug RLS policies and queries

### Useful SQL Queries

**Check your chats:**
```sql
SELECT * FROM user_chats_view;
```

**See all messages in a chat:**
```sql
SELECT
    m.text,
    p.name as sender,
    m.toxicity_label,
    m.created_at
FROM messages m
JOIN profiles p ON p.id = m.sender_id
WHERE m.chat_id = 'your-chat-id-here'
ORDER BY m.created_at;
```

**Make yourself an admin:**
```sql
UPDATE profiles
SET role = 'admin'
WHERE email = 'your-email@example.com';
```

---

## 🎨 Animation System

Tone uses **Framer Motion** with spring physics for buttery-smooth UI transitions:

- **Spring config**: `{ type: "spring", stiffness: 100, damping: 20 }`
- **Message animations**: Scale + fade on appear
- **Analysis panel**: Height animation with spring
- **Search bar**: Slide in/out with motion

---

## 📝 Development Notes

### Local vs Production

- **Local Backend**: `http://localhost:7860` (for testing AI changes)
- **Production Backend**: Deployed Hugging Face Space (always available)

Update `VITE_TONE_API_URL` in `.env` to switch between them.

### Testing Privacy

Create multiple test accounts to verify privacy:
1. User A sends message in Chat 1
2. User B (not in Chat 1) should NOT see the message
3. Use browser DevTools → Network to verify API calls respect RLS

---

© 2026 Tone. *Encouraging respectful communication through AI-driven insight.*
