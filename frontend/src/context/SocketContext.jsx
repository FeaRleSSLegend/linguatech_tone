import { createContext, useState, useEffect, useContext } from "react";
import { useAuth } from "./AuthContext";
import io from "socket.io-client";

const SocketContext = createContext();

export const useSocket = () => useContext(SocketContext);

export const SocketContextProvider = ({ children }) => {
    const [socket, setSocket] = useState(null);
    const [onlineUsers, setOnlineUsers] = useState([]);
    const { user } = useAuth(); // Assuming your AuthContext provides the logged-in user

    useEffect(() => {
        if (user) {
            // Replace with your backend URL
            const newSocket = io("http://localhost:3001", {
                query: { userId: user.id },
            });

            setSocket(newSocket);

            // Listen for the list of online user IDs from the server
            newSocket.on("getOnlineUsers", (users) => {
                setOnlineUsers(users);
            });

            return () => newSocket.close();
        } else {
            if (socket) {
                socket.close();
                setSocket(null);
            }
        }
    }, [user]);

    return (
        <SocketContext.Provider value={{ socket, onlineUsers }}>
            {children}
        </SocketContext.Provider>
    );
};