/**
 * Centralised teacher state store.
 *
 * Solves two issues:
 *  - #22: every other role already has a store; teacher role had none.
 *  - #24: teacherId derivation was copy-pasted across 4 pages.
 *
 * Usage
 * ─────
 *   const { teacherId, isUnlinked } = useTeacherId();
 *
 * Profile loading
 * ───────────────
 *   Call `useTeacherStore.getState().fetchProfile()` once from TeacherLayout
 *   so all teacher pages share a single fetch.
 */

import { create } from 'zustand';
import type { TeacherProfile } from '@/services/teacherService';
import teacherService from '@/services/teacherService';
import { useAuthStore } from './useAuthStore';

interface TeacherState {
  /** Full teacher profile, or null if not yet loaded / not linked */
  profile: TeacherProfile | null;
  profileLoading: boolean;
  profileError: string | null;

  /**
   * Fetch the authenticated teacher's profile from GET /teachers/me.
   * No-ops while a fetch is already in flight.
   * Safe to call on every TeacherLayout mount — idempotent if the profile
   * is already loaded.
   */
  fetchProfile: () => Promise<void>;

  /** Clear state on logout */
  clearProfile: () => void;
}

export const useTeacherStore = create<TeacherState>((set, get) => ({
  profile: null,
  profileLoading: false,
  profileError: null,

  fetchProfile: async () => {
    // Avoid concurrent fetches
    if (get().profileLoading) return;
    // Skip if already loaded
    if (get().profile) return;

    set({ profileLoading: true, profileError: null });
    try {
      const profile = await teacherService.getMyProfile();
      set({ profile, profileLoading: false });
    } catch (err: any) {
      const msg =
        err?.response?.data?.detail ??
        err?.response?.data?.message ??
        err?.message ??
        'Failed to load teacher profile';
      set({ profileError: msg, profileLoading: false });
    }
  },

  clearProfile: () =>
    set({ profile: null, profileError: null, profileLoading: false }),
}));

/**
 * Convenience hook used in all 4 teacher pages (#24).
 *
 * Derives teacherId from the JWT first (already in useAuthStore),
 * falling back to the fetched profile ID if the JWT predates linking.
 */
export function useTeacherId() {
  const user = useAuthStore(s => s.user);
  const profile = useTeacherStore(s => s.profile);

  // JWT carries teacher_id as teacherId when set during login after linking.
  // If the account was linked after the current session started, the fetched
  // profile.id is the reliable fallback.
  const teacherId = Number(user?.teacherId ?? profile?.id ?? 0);
  const isUnlinked = teacherId === 0;

  return { teacherId, isUnlinked };
}

export default useTeacherStore;
