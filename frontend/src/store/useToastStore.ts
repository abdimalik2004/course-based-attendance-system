import { create } from "zustand";

export type ToastType = "error" | "warning" | "success" | "info";

export interface Toast {
  id: string;
  message: string;
  type: ToastType;
}

interface ToastState {
  toasts: Toast[];
  push: (message: string, type?: ToastType) => void;
  dismiss: (id: string) => void;
}

export const useToastStore = create<ToastState>((set) => ({
  toasts: [],
  push: (message, type = "error") => {
    const id = Math.random().toString(36).slice(2);
    set((s) => ({ toasts: [...s.toasts, { id, message, type }] }));
    // Auto-dismiss after 4 s
    setTimeout(() => {
      set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) }));
    }, 4000);
  },
  dismiss: (id) =>
    set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })),
}));

/** Imperative helper — use outside React components (e.g. axios interceptor). */
export const toast = {
  error: (msg: string) => useToastStore.getState().push(msg, "error"),
  warning: (msg: string) => useToastStore.getState().push(msg, "warning"),
  success: (msg: string) => useToastStore.getState().push(msg, "success"),
  info: (msg: string) => useToastStore.getState().push(msg, "info"),
};
