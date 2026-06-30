import { useEffect, useRef, useState } from 'react';
import { Bell, Check, Trash2, Info, CheckCircle, AlertTriangle, XCircle, ExternalLink } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { useNotificationsStore } from '@/store/useNotificationsStore';
import { useAuthStore } from '@/store/useAuthStore';
import type { Notification } from '@/services/notificationsService';

function timeAgo(dateStr: string): string {
  const diff = Date.now() - new Date(dateStr).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  return `${days}d ago`;
}

function typeIcon(type: Notification['type']) {
  switch (type) {
    case 'success':
      return <CheckCircle size={14} className="text-green-500" />;
    case 'warning':
      return <AlertTriangle size={14} className="text-yellow-500" />;
    case 'error':
      return <XCircle size={14} className="text-red-500" />;
    default:
      return <Info size={14} className="text-blue-500" />;
  }
}

function typeBg(type: Notification['type']) {
  switch (type) {
    case 'success': return 'bg-green-100 dark:bg-green-500/20';
    case 'warning': return 'bg-yellow-100 dark:bg-yellow-500/20';
    case 'error':   return 'bg-red-100 dark:bg-red-500/20';
    default:        return 'bg-blue-100 dark:bg-blue-500/20';
  }
}

export function NotificationDropdown() {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();
  const { user, accessToken } = useAuthStore();
  const {
    notifications,
    unreadCount,
    loading,
    fetchNotifications,
    markRead,
    markAllRead,
    remove,
    connectWS,
    disconnectWS,
  } = useNotificationsStore();

  // Connect WS and fetch on mount
  useEffect(() => {
    if (!user?.id || !accessToken) return;
    fetchNotifications();
    connectWS(user.id, accessToken);
    return () => disconnectWS();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user?.id, accessToken]);

  // Close on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const handleNotificationClick = async (n: Notification) => {
    if (!n.is_read) await markRead(n.id);
    if (n.link) {
      setOpen(false);
      navigate(n.link);
    }
  };

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen((v) => !v)}
        className="relative rounded-full p-2 text-gray-500 transition-colors hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-white/10 focus:outline-none"
      >
        <Bell size={20} />
        {unreadCount > 0 && (
          <span className="absolute top-1 right-1 flex h-4 w-4 items-center justify-center rounded-full bg-primary text-[10px] font-bold text-white ring-2 ring-white dark:ring-dark-bg">
            {unreadCount > 9 ? '9+' : unreadCount}
          </span>
        )}
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: 10, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 10, scale: 0.95 }}
            transition={{ duration: 0.15, ease: 'easeOut' }}
            className="absolute right-0 top-12 w-80 rounded-xl border border-gray-200 dark:border-white/10 bg-white dark:bg-dark-card shadow-xl z-50 overflow-hidden"
          >
            {/* Header */}
            <div className="flex items-center justify-between px-4 py-3 border-b border-gray-100 dark:border-white/5">
              <h3 className="text-sm font-semibold text-gray-900 dark:text-white">
                Notifications
                {unreadCount > 0 && (
                  <span className="ml-2 inline-flex items-center rounded-full bg-primary/10 px-2 py-0.5 text-xs font-medium text-primary">
                    {unreadCount} new
                  </span>
                )}
              </h3>
              {unreadCount > 0 && (
                <button
                  onClick={markAllRead}
                  className="flex items-center gap-1 text-xs text-primary hover:underline"
                >
                  <Check size={12} />
                  Mark all read
                </button>
              )}
            </div>

            {/* List */}
            <div className="max-h-[320px] overflow-y-auto custom-scrollbar">
              {loading && notifications.length === 0 ? (
                <div className="flex items-center justify-center py-8 text-sm text-gray-400">
                  Loading…
                </div>
              ) : notifications.length === 0 ? (
                <div className="flex flex-col items-center justify-center gap-2 py-8 text-sm text-gray-400">
                  <Bell size={24} className="opacity-30" />
                  <span>No notifications yet</span>
                </div>
              ) : (
                <ul className="divide-y divide-gray-50 dark:divide-white/5">
                  {notifications.map((n) => (
                    <li
                      key={n.id}
                      className={`flex items-start gap-3 px-4 py-3 transition-colors group ${
                        n.is_read
                          ? 'hover:bg-gray-50 dark:hover:bg-white/5'
                          : 'bg-primary/5 hover:bg-primary/10 dark:bg-primary/10 dark:hover:bg-primary/15'
                      } ${n.link ? 'cursor-pointer' : ''}`}
                      onClick={() => handleNotificationClick(n)}
                    >
                      {/* Icon */}
                      <div className={`mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-full ${typeBg(n.type)}`}>
                        {typeIcon(n.type)}
                      </div>

                      {/* Content */}
                      <div className="min-w-0 flex-1">
                        <div className="flex items-start justify-between gap-1">
                          <p className={`text-sm leading-snug ${n.is_read ? 'text-gray-700 dark:text-gray-300' : 'font-medium text-gray-900 dark:text-white'}`}>
                            {n.title}
                          </p>
                          {n.link && (
                            <ExternalLink size={12} className="mt-0.5 shrink-0 text-gray-400 opacity-0 group-hover:opacity-100" />
                          )}
                        </div>
                        <p className="mt-0.5 text-xs text-gray-500 dark:text-gray-400 line-clamp-2">
                          {n.message}
                        </p>
                        <span className="mt-1 text-[11px] text-gray-400">
                          {timeAgo(n.created_at)}
                        </span>
                      </div>

                      {/* Delete */}
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          remove(n.id);
                        }}
                        className="mt-0.5 shrink-0 rounded p-1 text-gray-400 opacity-0 group-hover:opacity-100 hover:bg-red-50 hover:text-red-500 dark:hover:bg-red-500/10 transition-colors"
                        title="Dismiss"
                      >
                        <Trash2 size={12} />
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
