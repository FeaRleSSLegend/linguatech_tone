import { motion, AnimatePresence } from 'framer-motion';
import { useNotification } from '../context/NotificationContext';
import { Bell, AlertTriangle, CheckCircle, Info, X } from 'lucide-react';

export default function NotificationContainer() {
    const { notifications, removeNotification } = useNotification();

    return (
        <div className="fixed top-4 right-4 z-[9999] flex flex-col gap-3 pointer-events-none">
            <AnimatePresence>
                {notifications.map((notification) => (
                    <NotificationToast
                        key={notification.id}
                        notification={notification}
                        onClose={() => removeNotification(notification.id)}
                    />
                ))}
            </AnimatePresence>
        </div>
    );
}

function NotificationToast({ notification, onClose }) {
    const icons = {
        info: <Info className="w-5 h-5 text-blue-400" />,
        warning: <AlertTriangle className="w-5 h-5 text-yellow-400" />,
        toxic: <AlertTriangle className="w-5 h-5 text-red-500" />,
        success: <CheckCircle className="w-5 h-5 text-emerald-400" />,
        message: <Bell className="w-5 h-5 text-[var(--color-primary)]" />
    };

    const bgColor = {
        info: 'bg-blue-500/10 border-blue-500/20',
        warning: 'bg-yellow-500/10 border-yellow-500/20',
        toxic: 'bg-red-500/10 border-red-500/20',
        success: 'bg-emerald-500/10 border-emerald-500/20',
        message: 'bg-[#1a1a1a] border-[#2f3335]'
    };

    return (
        <motion.div
            initial={{ opacity: 0, x: 50, scale: 0.9 }}
            animate={{ opacity: 1, x: 0, scale: 1 }}
            exit={{ opacity: 0, x: 20, scale: 0.95 }}
            className={`
                pointer-events-auto
                min-w-[300px] max-w-[400px] p-4 rounded-xl border backdrop-blur-md shadow-2xl
                flex items-start gap-3 relative
                ${bgColor[notification.type] || bgColor.info}
            `}
        >
            <div className="shrink-0 mt-0.5">
                {icons[notification.type] || icons.info}
            </div>

            <div className="flex-1 pr-6">
                <h4 className="text-sm font-semibold text-[#e0ddd9] mb-1">
                    {notification.title}
                </h4>
                <p className="text-xs text-[#958d73] leading-relaxed">
                    {notification.message}
                </p>
            </div>

            <button
                onClick={onClose}
                className="absolute top-3 right-3 text-[#958d73] hover:text-[#e0ddd9] transition-colors"
            >
                <X className="w-4 h-4" />
            </button>
        </motion.div>
    );
}
