import { api } from './api';

export interface Notification {
  id: number;
  user_id: number;
  title: string;
  message: string;
  type: 'info' | 'success' | 'warning' | 'error';
  is_read: boolean;
  link: string | null;
  created_at: string;
}

export const notificationsService = {
  list(): Promise<Notification[]> {
    return api.get<Notification[]>('/notifications').then((r) => r.data);
  },

  unreadCount(): Promise<number> {
    return api
      .get<{ count: number }>('/notifications/unread-count')
      .then((r) => r.data.count);
  },

  markRead(id: number): Promise<void> {
    return api.put(`/notifications/${id}/read`).then(() => undefined);
  },

  markAllRead(): Promise<void> {
    return api.put('/notifications/read-all').then(() => undefined);
  },

  remove(id: number): Promise<void> {
    return api.delete(`/notifications/${id}`).then(() => undefined);
  },
};
