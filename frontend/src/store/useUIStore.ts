import { create } from 'zustand';

interface UIState {
  isEditProfileOpen: boolean;
  openEditProfile: () => void;
  closeEditProfile: () => void;

  isChangePasswordOpen: boolean;
  openChangePassword: () => void;
  closeChangePassword: () => void;
}

export const useUIStore = create<UIState>((set) => ({
  isEditProfileOpen: false,
  openEditProfile: () => set({ isEditProfileOpen: true }),
  closeEditProfile: () => set({ isEditProfileOpen: false }),

  isChangePasswordOpen: false,
  openChangePassword: () => set({ isChangePasswordOpen: true }),
  closeChangePassword: () => set({ isChangePasswordOpen: false }),
}));
