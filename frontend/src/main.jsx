import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { AuthProvider } from './context/AuthContext'
import { ChatProvider } from './context/ChatContext'
import { NotificationProvider } from './context/NotificationContext'
import { SocketContextProvider } from './context/SocketContext' // 1. Import it
import './index.css'
import App from './App.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <BrowserRouter>
      <AuthProvider>
        {/* 2. Wrap everything inside SocketContextProvider */}
        <SocketContextProvider> 
          <NotificationProvider>
            <ChatProvider>
              <App />
            </ChatProvider>
          </NotificationProvider>
        </SocketContextProvider>
      </AuthProvider>
    </BrowserRouter>
  </StrictMode>,
)