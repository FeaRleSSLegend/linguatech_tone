import { Bell } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { forwardRef } from 'react';

const NotificationBell = forwardRef(({ count = 0, onClick, isActive = false }, ref) => {
    return (
        <button
            ref={ref}
            onClick={onClick}
            className={`relative text-[#958d73] hover:text-[var(--color-primary)] transition-colors p-2 hover:bg-[var(--color-primary)]/10 rounded-lg ${
                isActive ? 'bg-[var(--color-primary)]/10 text-[var(--color-primary)]' : ''
            }`}
            title={`${count} pending friend request${count !== 1 ? 's' : ''}`}
        >
            <Bell className="w-5 h-5" />

            <AnimatePresence>
                {count > 0 && (
                    <motion.span
                        initial={{ scale: 0, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        exit={{ scale: 0, opacity: 0 }}
                        transition={{ type: "spring", stiffness: 500, damping: 25 }}
                        className="absolute -top-1 -right-1 min-w-[18px] h-[18px] bg-red-500 text-white text-[10px] font-bold rounded-full flex items-center justify-center px-1 shadow-[0_0_12px_rgba(239,68,68,0.6)]"
                    >
                        {count > 9 ? '9+' : count}
                    </motion.span>
                )}
            </AnimatePresence>
        </button>
    );
});

NotificationBell.displayName = 'NotificationBell';

export default NotificationBell;
