import { create } from 'zustand';
import {
  notificationsService,
  type Notification,
} from '@/services/notificationsService';

const WS_BASE =
  (import.meta.env.VITE_API_URL || '/api')
    .replace(/^http/, 'ws')
    .replace(/\/api$/, '') + '/api';

interface NotificationsState {
  notifications: Notification[];
  unreadCount: number;
  loading: boolean;
  wsConnected: boolean;

  // actions
  fetchNotifications: () => Promise<void>;
  markRead: (id: number) => Promise<void>;
  markAllRead: () => Promise<void>;
  remove: (id: number) => Promise<void>;
  connectWS: (userId: number, token: string) => void;
  disconnectWS: () => void;
  _pushNotification: (n: Notification) => void;
}

let _ws: WebSocket | null = null;

export const useNotificationsStore = create<NotificationsState>((set, get) => ({
  notifications: [],
  unreadCount: 0,
  loading: false,
  wsConnected: false,

  fetchNotifications: async () => {
    set({ loading: true });
    try {
      const notifications = await notificationsService.list();
      const unreadCount = notifications.filter((n) => !n.is_read).length;
      set({ notifications, unreadCount, loading: false });
    } catch {
      set({ loading: false });
    }
  },

  markRead: async (id: number) => {
    await notificationsService.markRead(id);
    set((state) => {
      const notifications = state.notifications.map((n) =>
        n.id === id ? { ...n, is_read: true } : n
      );
      return {
        notifications,
        unreadCount: notifications.filter((n) => !n.is_read).length,
      };
    });
  },

  markAllRead: async () => {
    await notificationsService.markAllRead();
    set((state) => ({
      notifications: state.notifications.map((n) => ({ ...n, is_read: true })),
      unreadCount: 0,
    }));
  },

  remove: async (id: number) => {
    await notificationsService.remove(id);
    set((state) => {
      const notifications = state.notifications.filter((n) => n.id !== id);
      return {
        notifications,
        unreadCount: notifications.filter((n) => !n.is_read).length,
      };
    });
  },

  _pushNotification: (n: Notification) => {
    set((state) => ({
      notifications: [n, ...state.notifications].slice(0, 50),
      unreadCount: state.unreadCount + (n.is_read ? 0 : 1),
    }));
  },

  connectWS: (userId: number, token: string) => {
    if (_ws && _ws.readyState < 2) return; // already open or connecting

    const url = `${WS_BASE}/notifications/ws/${userId}?token=${token}`;
    _ws = new WebSocket(url);

    _ws.onopen = () => set({ wsConnected: true });

    _ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'notification') {
          get()._pushNotification(data.payload as Notification);
        }
      } catch {
        // ignore unparseable frames
      }
    };

    _ws.onclose = () => {
      set({ wsConnected: false });
      _ws = null;
    };

    _ws.onerror = () => {
      _ws?.close();
    };
  },

  disconnectWS: () => {
    _ws?.close();
    _ws = null;
    set({ wsConnected: false });
  },
}));
