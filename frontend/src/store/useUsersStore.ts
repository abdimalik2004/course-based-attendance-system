import { create } from "zustand";
import type {
  User,
  Role,
  Faculty,
  CreateUserPayload,
} from "@/types/users.types";
import { usersService } from "@/services/usersService";

interface UsersState {
  users: User[];
  roles: Role[];
  faculties: Faculty[];
  isLoading: boolean;
  isModalOpen: boolean;
  error: string | null;

  fetchUsers: () => Promise<void>;
  fetchRolesAndFaculties: () => Promise<void>;
  addUser: (data: CreateUserPayload) => Promise<void>;
  editUser: (id: string, data: Partial<User>) => Promise<void>;
  deleteUser: (id: string) => Promise<void>;
  setModalOpen: (open: boolean) => void;

  addRole: (name: string) => Promise<void>;
  editRole: (id: string, name: string) => Promise<void>;
  removeRole: (id: string) => Promise<void>;
}

export const useUsersStore = create<UsersState>((set) => ({
  users: [],
  roles: [],
  faculties: [],
  isLoading: false,
  isModalOpen: false,
  error: null,

  fetchUsers: async () => {
    set({ isLoading: true, error: null });
    try {
      const res = await usersService.getUsers();
      const users = res?.items || [];
      set({ users, isLoading: false });
    } catch {
      set({ error: "Failed to fetch users", isLoading: false });
    }
  },

  fetchRolesAndFaculties: async () => {
    try {
      const [roles, faculties] = await Promise.all([
        usersService.getRoles(),
        usersService.getFaculties(),
      ]);
      set({ roles, faculties });
    } catch (error) {
      console.error("Failed to fetch roles or faculties", error);
    }
  },

  addUser: async (data) => {
    try {
      const newUser = await usersService.createUser(data);
      set((state) => ({
        users: [...state.users, newUser],
        isModalOpen: false,
      }));
    } catch {
      throw new Error("Failed to create user");
    }
  },

  editUser: async (id: string, data: Partial<User>) => {
    try {
      const updatedUser = await usersService.updateUser(id, data);
      set((state) => ({
        users: state.users.map((u) => (u.id === id ? updatedUser : u)),
      }));
    } catch {
      throw new Error("Failed to update user");
    }
  },

  deleteUser: async (id) => {
    // Attempt server delete then update local state
    try {
      await usersService.deleteUser(id);
      set((state) => ({ users: state.users.filter((u) => u.id !== id) }));
    } catch (err) {
      // If delete failed, still remove locally to keep UI responsive; caller may refresh later
      console.error("Failed to delete user on server", err);
      set((state) => ({ users: state.users.filter((u) => u.id !== id) }));
    }
  },

  setModalOpen: (open) => set({ isModalOpen: open }),

  addRole: async (name: string) => {
    try {
      const newRole = await usersService.createRole(name);
      set((state) => ({ roles: [...state.roles, newRole] }));
    } catch {
      throw new Error("Failed to create role");
    }
  },

  editRole: async (id: string, name: string) => {
    try {
      const updatedRole = await usersService.updateRole(id, name);
      set((state) => ({
        roles: state.roles.map((r) => (r.id === id ? updatedRole : r)),
      }));
    } catch {
      throw new Error("Failed to update role");
    }
  },

  removeRole: async (id: string) => {
    try {
      await usersService.deleteRole(id);
      set((state) => ({ roles: state.roles.filter((r) => r.id !== id) }));
    } catch {
      throw new Error("Failed to delete role");
    }
  },
}));
