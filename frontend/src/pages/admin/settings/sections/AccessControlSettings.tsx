import { useState, useEffect } from 'react';
import { Plus, CheckCircle, AlertCircle } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { Switch } from '@/components/ui/Switch';
import { Modal } from '@/components/ui/Modal';
import { Input } from '@/components/ui/Input';
import { useUsersStore } from '@/store/useUsersStore';
import { fetchSettings, saveSettings } from '@/services/settingsService';

interface RolePerms {
  manageUsers: boolean;
  manageRoles: boolean;
  viewReports: boolean;
  takeAttendance: boolean;
  editSettings: boolean;
  manageAcademic: boolean;
  manageAdmissions: boolean;
}

// Roles that are internal/backend-only and should not appear in the Access Control UI
const HIDDEN_ROLES = new Set<string>();

// Default permission templates keyed by uppercase role name
const defaultPermsFor = (name: string): RolePerms => {
  switch (name.toUpperCase()) {
    case 'SUPER_ADMIN':
      return { manageUsers: true, manageRoles: true, viewReports: true, takeAttendance: true, editSettings: true, manageAcademic: true, manageAdmissions: true };
    case 'ACADEMIA':
      return { manageUsers: false, manageRoles: false, viewReports: true, takeAttendance: false, editSettings: false, manageAcademic: true, manageAdmissions: false };
    case 'ADMISSIONS':
      return { manageUsers: false, manageRoles: false, viewReports: true, takeAttendance: false, editSettings: false, manageAcademic: false, manageAdmissions: true };
    case 'HR':
      return { manageUsers: true, manageRoles: false, viewReports: true, takeAttendance: false, editSettings: false, manageAcademic: false, manageAdmissions: false };
    case 'FACULTY':
      return { manageUsers: false, manageRoles: false, viewReports: true, takeAttendance: false, editSettings: false, manageAcademic: true, manageAdmissions: false };
    case 'TEACHER':
      return { manageUsers: false, manageRoles: false, viewReports: false, takeAttendance: true, editSettings: false, manageAcademic: false, manageAdmissions: false };
    case 'STUDENT':
      return { manageUsers: false, manageRoles: false, viewReports: false, takeAttendance: false, editSettings: false, manageAcademic: false, manageAdmissions: false };
    default:
      return { manageUsers: false, manageRoles: false, viewReports: false, takeAttendance: false, editSettings: false, manageAcademic: false, manageAdmissions: false };
  }
};

const LABEL_MAP: Record<string, string> = {
  SUPER_ADMIN: 'Administrator',
  ACADEMIA: 'Academic Office',
  ADMISSIONS: 'Admissions Office',
  HR: 'Human Resources',
  FACULTY: 'Faculty Admin',
  TEACHER: 'Teacher',
  STUDENT: 'Student',
};

const DESC_MAP: Record<string, string> = {
  SUPER_ADMIN: 'Full System Access',
  ACADEMIA: 'Curriculum & Structures',
  ADMISSIONS: 'Student Onboarding',
  HR: 'Staffing & Reports',
  FACULTY: 'Department Analytics',
  TEACHER: 'Course Management',
  STUDENT: 'Learning Portal',
};

export function AccessControlSettings() {
  const { roles, fetchRolesAndFaculties, addRole } = useUsersStore();

  const [selectedRole, setSelectedRole] = useState<string>('');
  const [permissions, setPermissions] = useState<Record<string, RolePerms>>({});
  const [hasChanges, setHasChanges] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [saveError, setSaveError] = useState('');

  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [newRoleName, setNewRoleName] = useState('');
  const [isAdding, setIsAdding] = useState(false);
  const [addError, setAddError] = useState('');

  // Load roles from the DB on mount, and restore any saved permissions from system_settings
  useEffect(() => {
    fetchRolesAndFaculties();
    fetchSettings()
      .then((data) => {
        if (data['access_control.permissions']) {
          try {
            const saved = JSON.parse(data['access_control.permissions']) as Record<string, RolePerms>;
            setPermissions((prev) => ({ ...prev, ...saved }));
          } catch {
            // ignore malformed JSON
          }
        }
      })
      .catch(() => {});
  }, [fetchRolesAndFaculties]);

  // Initialise permission map whenever roles list changes
  useEffect(() => {
    if (roles.length === 0) return;
    setPermissions(prev => {
      const next = { ...prev };
      roles.forEach((r) => {
        if (!next[r.name]) {
          next[r.name] = defaultPermsFor(r.name);
        }
      });
      return next;
    });
    // Auto-select first role if nothing selected yet
    setSelectedRole(prev => prev || roles[0]?.name || '');
  }, [roles]);

  const handleToggle = (key: keyof RolePerms) => {
    if (selectedRole === 'SUPER_ADMIN') return;
    setPermissions(prev => ({
      ...prev,
      [selectedRole]: {
        ...prev[selectedRole],
        [key]: !prev[selectedRole]?.[key],
      },
    }));
    setHasChanges(true);
  };

  const handleSavePermissions = async () => {
    setIsSaving(true);
    setSaveError('');
    setSaveSuccess(false);
    try {
      await saveSettings({ 'access_control.permissions': JSON.stringify(permissions) });
      setHasChanges(false);
      setSaveSuccess(true);
      setTimeout(() => setSaveSuccess(false), 3000);
    } catch {
      setSaveError('Failed to save permissions. Please try again.');
    } finally {
      setIsSaving(false);
    }
  };

  const handleAddRole = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newRoleName.trim()) return;
    setAddError('');
    setIsAdding(true);
    try {
      await addRole(newRoleName.trim());
      setNewRoleName('');
      setIsAddModalOpen(false);
    } catch (err: any) {
      const detail = err?.response?.data?.detail;
      setAddError(typeof detail === 'string' ? detail : 'Failed to create role.');
    } finally {
      setIsAdding(false);
    }
  };

  const currentPerms = permissions[selectedRole];
  const isSuperAdmin = selectedRole === 'SUPER_ADMIN';

  const permRows: { key: keyof RolePerms; label: string }[] = [
    { key: 'manageUsers', label: 'Manage Users & Accounts' },
    { key: 'manageRoles', label: 'Manage System Roles' },
    { key: 'viewReports', label: 'View & Export Reports' },
    { key: 'takeAttendance', label: 'Execute Live Attendance Scanning' },
    { key: 'editSettings', label: 'Modify Global Settings' },
    { key: 'manageAcademic', label: 'Manage Academic Structures' },
    { key: 'manageAdmissions', label: 'Approve / Reject Admissions' },
  ];

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 border-b border-white/5 pb-4">
        <div>
          <h2 className="text-xl font-bold text-white mb-1">Access Control</h2>
          <p className="text-sm text-gray-400">Manage permissions per organisational role.</p>
        </div>
        <Button size="sm" className="gap-2" onClick={() => { setAddError(''); setIsAddModalOpen(true); }}>
          <Plus size={15} />
          Add Role
        </Button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">

        {/* Roles list — loaded from DB */}
        <div className="flex flex-col gap-2 max-h-[500px] overflow-y-auto pr-1 custom-scrollbar">
          {roles.length === 0 ? (
            <p className="text-sm text-gray-500 px-2">Loading roles…</p>
          ) : (
            roles.filter((role) => !HIDDEN_ROLES.has(role.name)).map((role) => (
              <button
                key={role.id}
                onClick={() => setSelectedRole(role.name)}
                className={`p-4 rounded-xl text-left transition-all border ${
                  selectedRole === role.name
                    ? 'bg-primary/20 border-primary text-primary-accent'
                    : 'bg-white/5 border-white/5 text-gray-400 hover:bg-white/10 hover:text-white'
                }`}
              >
                <div className="font-semibold mb-1 text-sm">
                  {LABEL_MAP[role.name] ?? role.name}
                </div>
                <div className="text-xs opacity-70">
                  {DESC_MAP[role.name] ?? 'Custom role'}
                </div>
              </button>
            ))
          )}
        </div>

        {/* Permissions panel */}
        <Card className="glass-card border-white/5 lg:col-span-3">
          <CardContent className="p-6">
            {!selectedRole || !currentPerms ? (
              <p className="text-gray-500 text-sm">Select a role to view permissions.</p>
            ) : (
              <>
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-white font-medium text-lg">
                    {LABEL_MAP[selectedRole] ?? selectedRole} Permissions
                  </h3>
                  {isSuperAdmin && <Badge variant="success">Immutable</Badge>}
                </div>

                <div className="space-y-1">
                  {permRows.map(({ key, label }) => (
                    <div key={key} className="flex items-center justify-between p-3 border-b border-white/5">
                      <span className="text-gray-300 text-sm">{label}</span>
                      <Switch
                        checked={currentPerms[key]}
                        disabled={isSuperAdmin}
                        onChange={() => handleToggle(key)}
                      />
                    </div>
                  ))}
                </div>

                <div className="pt-6 space-y-3">
                  {saveSuccess && (
                    <div className="flex items-center gap-2 text-sm text-emerald-400">
                      <CheckCircle size={14} />
                      <span>Permissions saved.</span>
                    </div>
                  )}
                  {saveError && (
                    <div className="flex items-center gap-2 text-sm text-red-400">
                      <AlertCircle size={14} />
                      <span>{saveError}</span>
                    </div>
                  )}
                  <Button
                    disabled={!hasChanges || isSuperAdmin || isSaving}
                    isLoading={isSaving}
                    onClick={handleSavePermissions}
                  >
                    Save Permissions
                  </Button>
                  {isSuperAdmin && (
                    <p className="text-xs text-gray-500 mt-2">
                      Administrator permissions cannot be modified.
                    </p>
                  )}
                </div>
              </>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Add Role Modal */}
      <Modal
        isOpen={isAddModalOpen}
        onClose={() => setIsAddModalOpen(false)}
        title="Add New Role"
        className="md:max-w-md"
      >
        <form onSubmit={handleAddRole} className="space-y-4 pt-2">
          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Role Name <span className="text-red-500">*</span>
            </label>
            <Input
              value={newRoleName}
              onChange={(e) => { setNewRoleName(e.target.value); setAddError(''); }}
              placeholder="e.g. GUEST_INSTRUCTOR"
              autoFocus
            />
            <p className="text-xs text-gray-500 ml-1">
              Role names are stored in uppercase (e.g. HR, TEACHER, GUEST_ADMIN).
            </p>
          </div>

          {addError && (
            <p className="text-sm text-red-400">{addError}</p>
          )}

          <div className="flex justify-end gap-3 pt-4 border-t border-gray-100 dark:border-white/5">
            <Button type="button" variant="ghost" onClick={() => setIsAddModalOpen(false)}>
              Cancel
            </Button>
            <Button type="submit" isLoading={isAdding} disabled={!newRoleName.trim() || isAdding} className="gap-2">
              <Plus size={15} />
              Create Role
            </Button>
          </div>
        </form>
      </Modal>
    </div>
  );
}
