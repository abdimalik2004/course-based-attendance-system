import { create } from "zustand";
import { persist } from "zustand/middleware";

type Role =
  | "SUPER_ADMIN"
  | "ACADEMIA"
  | "TEACHER"
  | "HR"
  | "ADMISSIONS"
  | "FACULTY"
  | "STUDENT"
  | null;

interface User {
  id?: string | number;
  username: string;
  email?: string;
  role?: Role;
  facultyId?: string | number | null;
  teacherId?: string | number | null;
  studentId?: number | null;
  studentNumber?: string | null;
  profile_image_url?: string | null;
  full_name?: string | null;
}

interface AuthState {
  user: User | null;
  isAuthenticated: boolean;
  accessToken: string | null;
  hasHydrated: boolean;
  setTokens: (access: string) => void;
  clearTokens: () => void;
  login: (user: User, access?: string) => void;
  logout: () => void;
  updateProfileImage: (url: string | null) => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      user: null,
      isAuthenticated: false,
      accessToken: null,
      hasHydrated: false,
      setTokens: (access: string) =>
        set({
          accessToken: access,
          isAuthenticated: true,
        }),
      clearTokens: () => set({ accessToken: null }),
      login: (user: any, access?: string) => {
        const primaryRole =
          Array.isArray(user?.role_names) && user.role_names.length > 0
            ? user.role_names[0]
            : user?.role || null;
        const mappedUser: User = {
          id: user?.id,
          username: user?.username,
          email: user?.email,
          facultyId: user?.faculty_id ?? user?.facultyId ?? null,
          teacherId: user?.teacher_id ?? user?.teacherId ?? null,
          studentId: user?.student_id ?? user?.studentId ?? null,
          studentNumber: user?.student_number ?? user?.studentNumber ?? null,
          profile_image_url: user?.profile_image_url ?? null,
          full_name: user?.full_name ?? null,
          role: primaryRole
            ? (primaryRole.toString().toUpperCase() as Role)
            : null,
        };
        return set({
          user: mappedUser,
          isAuthenticated: true,
          accessToken: access ?? null,
        });
      },
      logout: () =>
        set({
          user: null,
          isAuthenticated: false,
          accessToken: null,
        }),
      updateProfileImage: (url: string | null) =>
        set((state) => ({
          user: state.user ? { ...state.user, profile_image_url: url } : state.user,
        })),
    }),
    {
      name: "heegan-auth",
      partialize: (state) => ({
        user: state.user,
        isAuthenticated: state.isAuthenticated,
      }),
      onRehydrateStorage: () => () => {
        useAuthStore.setState({ hasHydrated: true });
      },
    },
  ),
);
