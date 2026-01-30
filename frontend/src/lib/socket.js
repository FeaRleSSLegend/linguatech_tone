import { Server } from "socket.io";
import http from "http";
import express from "express";

const app = express();
const server = http.createServer(app);

const io = new Server(server, {
    cors: {
        origin: ["http://localhost:5173"], // Your Frontend URL
        methods: ["GET", "POST"],
    },
});

// Helper map: { userId: socketId }
const userSocketMap = {}; 

export const getReceiverSocketId = (receiverId) => {
    return userSocketMap[receiverId];
};

io.on("connection", (socket) => {
    console.log("A user connected", socket.id);

    const userId = socket.handshake.query.userId;
    if (userId !== "undefined") userSocketMap[userId] = socket.id;

    // Tell everyone who is online right now
    io.emit("getOnlineUsers", Object.keys(userSocketMap));

    socket.on("disconnect", () => {
        console.log("User disconnected", socket.id);
        delete userSocketMap[userId];
        // Update everyone that someone left
        io.emit("getOnlineUsers", Object.keys(userSocketMap));
    });
});

export { app, io, server };