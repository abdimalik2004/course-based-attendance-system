import { useEffect, useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { Plus, Edit2, Trash2, Eye, X, Save } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/Table";
import { Modal } from "@/components/ui/Modal";
import { useUsersStore } from "@/store/useUsersStore";

// Roles that are internal/backend-only and should not be shown in the UI list
const HIDDEN_ROLES = new Set<string>();

const roleLabelMap: Record<string, string> = {
  SUPER_ADMIN: "Admin",
  ACADEMIA: "Academia",
  ADMISSIONS: "Admission",
  HR: "HR",
  FACULTY: "Faculty",
  TEACHER: "Teacher",
  STUDENT: "Student",
};

const roleSchema = z.object({
  name: z.string().min(2, "Role name must be at least 2 characters"),
});

type RoleFormData = z.infer<typeof roleSchema>;

export default function RolesManagement() {
  const { roles, fetchRolesAndFaculties, addRole, editRole, removeRole } =
    useUsersStore();

  // Modals state
  const [viewModalOpen, setViewModalOpen] = useState(false);
  const [createModalOpen, setCreateModalOpen] = useState(false);
  const [editModalOpen, setEditModalOpen] = useState(false);
  const [deleteModalOpen, setDeleteModalOpen] = useState(false);

  const [selectedRole, setSelectedRole] = useState<{
    id: string;
    name: string;
  } | null>(null);
  const [roleToDelete, setRoleToDelete] = useState<{
    id: string;
    name: string;
  } | null>(null);

  // Edit state
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editName, setEditName] = useState("");
  const [editError, setEditError] = useState("");
  const [deleteError, setDeleteError] = useState("");
  const [editSaving, setEditSaving] = useState(false);
  const [deleteDeleting, setDeleteDeleting] = useState(false);

  const {
    register,
    handleSubmit,
    reset,
    formState: { errors, isSubmitting },
  } = useForm<RoleFormData>({
    resolver: zodResolver(roleSchema),
  });

  useEffect(() => {
    fetchRolesAndFaculties();
  }, [fetchRolesAndFaculties]);

  const onCreateSubmit = async (data: RoleFormData) => {
    try {
      await addRole(data.name);
      reset();
      setCreateModalOpen(false);
    } catch (err) {
      console.error(err);
    }
  };

  const handleEditClick = (role: { id: string; name: string }) => {
    setEditingId(role.id);
    setEditName(role.name);
    setEditError("");
    setEditModalOpen(true);
  };

  const handleSaveEdit = async () => {
    if (!editName.trim() || !editingId) return;
    setEditSaving(true);
    setEditError("");
    try {
      await editRole(editingId, editName.trim());
      setEditModalOpen(false);
      setEditingId(null);
    } catch (err: any) {
      setEditError(err?.message ?? "Failed to update role");
    } finally {
      setEditSaving(false);
    }
  };

  const handleCancelEdit = () => {
    setEditModalOpen(false);
    setEditingId(null);
    setEditName("");
    setEditError("");
  };

  const handleDeleteClick = (role: { id: string; name: string }) => {
    setRoleToDelete(role);
    setDeleteError("");
    setDeleteModalOpen(true);
  };

  const confirmDelete = async () => {
    if (!roleToDelete) return;
    setDeleteDeleting(true);
    setDeleteError("");
    try {
      await removeRole(roleToDelete.id);
      setDeleteModalOpen(false);
      setRoleToDelete(null);
    } catch (err: any) {
      setDeleteError(err?.message ?? "Failed to delete role");
    } finally {
      setDeleteDeleting(false);
    }
  };

  const handleView = (role: { id: string; name: string }) => {
    setSelectedRole(role);
    setViewModalOpen(true);
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-gray-900 dark:text-gray-100">
            Roles Management
          </h1>
          <p className="text-gray-500 dark:text-gray-400 mt-1">
            Create and manage user roles in the system
          </p>
        </div>
        <Button onClick={() => setCreateModalOpen(true)} className="gap-2">
          <Plus size={18} />
          Add Role
        </Button>
      </div>

      <Card className="glass-card shadow-2xl shadow-primary/5">
        <CardContent className="p-0">
          <div className="overflow-x-auto custom-scrollbar">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-24">ID</TableHead>
                  <TableHead>Name</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {roles.length === 0 ? (
                  <TableRow>
                    <TableCell
                      colSpan={3}
                      className="h-24 text-center text-gray-500"
                    >
                      No roles found. Create one to get started.
                    </TableCell>
                  </TableRow>
                ) : (
                  roles
                    .filter((role) => !HIDDEN_ROLES.has(role.name))
                    .map((role) => (
                      <TableRow key={role.id}>
                        <TableCell className="font-medium text-gray-500 dark:text-gray-400">
                          #{role.id}
                        </TableCell>
                        <TableCell className="font-medium text-gray-900 dark:text-white">
                          {roleLabelMap[role.name] ?? role.name}
                        </TableCell>
                        <TableCell className="text-right">
                          <div className="flex items-center justify-end gap-2">
                            <button
                              onClick={() => handleView(role)}
                              className="p-1.5 rounded-lg text-blue-500 hover:text-blue-600 hover:bg-blue-50 dark:hover:bg-blue-500/10 transition-colors"
                              title="View"
                            >
                              <Eye size={18} />
                            </button>
                            <button
                              onClick={() => handleEditClick(role)}
                              className="p-1.5 rounded-lg text-amber-500 hover:text-amber-600 hover:bg-amber-50 dark:hover:bg-amber-500/10 transition-colors"
                              title="Edit"
                            >
                              <Edit2 size={18} />
                            </button>
                            <button
                              onClick={() => handleDeleteClick(role)}
                              className="p-1.5 rounded-lg text-red-500 hover:text-red-600 hover:bg-red-50 dark:hover:bg-red-500/10 transition-colors"
                              title="Delete"
                            >
                              <Trash2 size={18} />
                            </button>
                          </div>
                        </TableCell>
                      </TableRow>
                    ))
                )}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>

      {/* View Modal */}
      <Modal
        isOpen={viewModalOpen}
        onClose={() => setViewModalOpen(false)}
        title="Role Details"
      >
        <div className="space-y-4 pt-2">
          <div className="grid grid-cols-3 gap-4 border-b border-gray-100 dark:border-white/5 pb-4">
            <div className="text-sm font-medium text-gray-500">Role ID</div>
            <div className="col-span-2 text-sm font-medium text-gray-900 dark:text-white">
              #{selectedRole?.id}
            </div>
          </div>
          <div className="grid grid-cols-3 gap-4 border-b border-gray-100 dark:border-white/5 pb-4">
            <div className="text-sm font-medium text-gray-500">Role Name</div>
            <div className="col-span-2 text-sm font-medium text-gray-900 dark:text-white">
              {selectedRole?.name}
            </div>
          </div>
          <div className="flex justify-end pt-4">
            <Button onClick={() => setViewModalOpen(false)}>Close</Button>
          </div>
        </div>
      </Modal>

      {/* Create Role Modal */}
      <Modal
        isOpen={createModalOpen}
        onClose={() => setCreateModalOpen(false)}
        title="Create New Role"
        className="md:max-w-md"
      >
        <form
          onSubmit={handleSubmit(onCreateSubmit)}
          className="space-y-4 pt-2"
        >
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">
              Role Name <span className="text-red-500">*</span>
            </label>
            <Input
              placeholder="e.g. Guest"
              {...register("name")}
              error={errors.name?.message}
              className="glass-input"
            />
          </div>

          <div className="flex items-center justify-end gap-3 pt-6 border-t border-gray-100 dark:border-white/5 mt-6">
            <Button
              type="button"
              variant="ghost"
              onClick={() => {
                setCreateModalOpen(false);
                reset();
              }}
            >
              Cancel
            </Button>
            <Button type="submit" isLoading={isSubmitting} className="gap-2">
              <Plus size={18} />
              Create role
            </Button>
          </div>
        </form>
      </Modal>

      {/* Edit Role Modal */}
      <Modal
        isOpen={editModalOpen}
        onClose={handleCancelEdit}
        title="Edit Role"
        className="md:max-w-md"
      >
        <div className="space-y-4 pt-2">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">
              Role Name <span className="text-red-500">*</span>
            </label>
            <Input
              value={editName}
              onChange={(e) => setEditName(e.target.value)}
              placeholder="e.g. Guest"
              className="glass-input"
              autoFocus
            />
          </div>

          {editError && (
            <p className="text-sm text-red-400">{editError}</p>
          )}

          <div className="flex items-center justify-end gap-3 pt-6 border-t border-gray-100 dark:border-white/5 mt-6">
            <Button type="button" variant="ghost" onClick={handleCancelEdit}>
              Cancel
            </Button>
            <Button onClick={handleSaveEdit} isLoading={editSaving} className="gap-2">
              <Save size={18} />
              Save Changes
            </Button>
          </div>
        </div>
      </Modal>

      {/* Delete Confirmation Modal */}
      <Modal
        isOpen={deleteModalOpen}
        onClose={() => setDeleteModalOpen(false)}
        title="Confirm Deletion"
        className="md:max-w-sm"
      >
        <div className="space-y-4 pt-2">
          <p className="text-gray-600 dark:text-gray-300">
            Are you sure you want to delete the role{" "}
            <span className="font-semibold text-gray-900 dark:text-white">
              {roleToDelete?.name}
            </span>
            ? This action cannot be undone.
          </p>

          {deleteError && (
            <p className="text-sm text-red-400">{deleteError}</p>
          )}

          <div className="flex items-center justify-end gap-3 pt-6 border-t border-gray-100 dark:border-white/5 mt-6">
            <Button
              type="button"
              variant="ghost"
              onClick={() => {
                setDeleteModalOpen(false);
                setRoleToDelete(null);
                setDeleteError("");
              }}
            >
              Cancel
            </Button>
            <Button
              onClick={confirmDelete}
              isLoading={deleteDeleting}
              className="gap-2 bg-red-500 hover:bg-red-600 text-white"
            >
              <Trash2 size={18} />
              Delete Role
            </Button>
          </div>
        </div>
      </Modal>
    </div>
  );
}