import { useState } from 'react';
import { Card, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { Switch } from '@/components/ui/Switch';
import { Modal } from '@/components/ui/Modal';
import { Input } from '@/components/ui/Input';

interface RolePerms {
  manageUsers: boolean;
  manageRoles: boolean;
  viewReports: boolean;
  takeAttendance: boolean;
  editSettings: boolean;
  manageAcademic: boolean;
  manageAdmissions: boolean;
}

type SystemRole = string;

const initialRoles = [
  { id: 'admin', name: 'Administrator', desc: 'Full System Access' },
  { id: 'academia', name: 'Academic Office', desc: 'Curriculum & Structures' },
  { id: 'admission', name: 'Admissions Office', desc: 'Student Onboarding' },
  { id: 'hr', name: 'Human Resources', desc: 'Staffing & Reports' },
  { id: 'faculty', name: 'Faculty Admin', desc: 'Department Analytics' },
  { id: 'teacher', name: 'Teacher', desc: 'Course Management' },
  { id: 'student', name: 'Student', desc: 'Learning Portal' }
];

export function AccessControlSettings() {
  const [hasChanges, setHasChanges] = useState(false);
  const [rolesList, setRolesList] = useState(initialRoles);
  const [selectedRole, setSelectedRole] = useState<SystemRole>('faculty');
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [newRoleName, setNewRoleName] = useState('');
  const [newRoleDesc, setNewRoleDesc] = useState('');

  const [permissions, setPermissions] = useState<Record<SystemRole, RolePerms>>({
    admin: { manageUsers: true, manageRoles: true, viewReports: true, takeAttendance: true, editSettings: true, manageAcademic: true, manageAdmissions: true },
    academia: { manageUsers: false, manageRoles: false, viewReports: true, takeAttendance: false, editSettings: false, manageAcademic: true, manageAdmissions: false },
    admission: { manageUsers: false, manageRoles: false, viewReports: true, takeAttendance: false, editSettings: false, manageAcademic: false, manageAdmissions: true },
    hr: { manageUsers: true, manageRoles: false, viewReports: true, takeAttendance: false, editSettings: false, manageAcademic: false, manageAdmissions: false },
    faculty: { manageUsers: false, manageRoles: false, viewReports: true, takeAttendance: false, editSettings: false, manageAcademic: false, manageAdmissions: false },
    teacher: { manageUsers: false, manageRoles: false, viewReports: false, takeAttendance: true, editSettings: false, manageAcademic: false, manageAdmissions: false },
    student: { manageUsers: false, manageRoles: false, viewReports: false, takeAttendance: false, editSettings: false, manageAcademic: false, manageAdmissions: false }
  });

  const handleToggle = (key: keyof RolePerms) => {
    if (selectedRole === 'admin') return; // Admins have static god-mode
    setPermissions(prev => ({
      ...prev,
      [selectedRole]: {
        ...prev[selectedRole],
        [key]: !prev[selectedRole][key]
      }
    }));
    setHasChanges(true);
  };

  const handleAddRole = (e: React.FormEvent) => {
    e.preventDefault();
    if (!newRoleName.trim()) return;

    const newId = newRoleName.toLowerCase().replace(/\s+/g, '_');
    
    setRolesList(prev => [...prev, { id: newId, name: newRoleName, desc: newRoleDesc }]);
    setPermissions(prev => ({
      ...prev,
      [newId]: {
        manageUsers: false,
        manageRoles: false,
        viewReports: false,
        takeAttendance: false,
        editSettings: false,
        manageAcademic: false,
        manageAdmissions: false,
      }
    }));
    
    setNewRoleName('');
    setNewRoleDesc('');
    setIsAddModalOpen(false);
    setSelectedRole(newId);
  };

  const currentPerms = permissions[selectedRole];

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 border-b border-white/5 pb-4">
        <div>
          <h2 className="text-xl font-bold text-white mb-1">Access Control</h2>
          <p className="text-sm text-gray-400">Manage fine-grained permissions per organizational role.</p>
        </div>
        <Button size="sm" variant="secondary" onClick={() => setIsAddModalOpen(true)}>+ Add Role</Button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        
        {/* Roles List */}
        <div className="flex flex-col gap-2 h-[500px] overflow-y-auto pr-2 custom-scrollbar">
           {rolesList.map((role) => (
             <button 
               key={role.id}
               onClick={() => setSelectedRole(role.id as SystemRole)}
               className={`p-4 rounded-xl text-left transition-all border ${selectedRole === role.id ? 'bg-primary/20 border-primary text-primary-accent' : 'bg-white/5 border-white/5 text-gray-400 hover:bg-white/10 hover:text-white'}`}
             >
               <div className="font-semibold mb-1">{role.name}</div>
               <div className="text-xs opacity-70">{role.desc}</div>
             </button>
           ))}
        </div>

        {/* Permissions Grid */}
        <Card className="glass-card border-white/5 lg:col-span-3">
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-6">
               <h3 className="text-white font-medium text-lg capitalize">{selectedRole} Permissions</h3>
               {selectedRole === 'admin' && <Badge variant="success">Immutable</Badge>}
            </div>

            <div className="space-y-4">
              <div className="flex items-center justify-between p-3 border-b border-white/5">
                <span className="text-gray-300 text-sm">Manage Users & Accounts</span>
                <Switch checked={currentPerms.manageUsers} disabled={selectedRole === 'admin'} onChange={() => handleToggle('manageUsers')} />
              </div>
              <div className="flex items-center justify-between p-3 border-b border-white/5">
                <span className="text-gray-300 text-sm">Manage System Roles</span>
                <Switch checked={currentPerms.manageRoles} disabled={selectedRole === 'admin'} onChange={() => handleToggle('manageRoles')} />
              </div>
              <div className="flex items-center justify-between p-3 border-b border-white/5">
                <span className="text-gray-300 text-sm">View & Export Target Reports</span>
                <Switch checked={currentPerms.viewReports} disabled={selectedRole === 'admin'} onChange={() => handleToggle('viewReports')} />
              </div>
              <div className="flex items-center justify-between p-3 border-b border-white/5">
                <span className="text-gray-300 text-sm">Execute Live Attendance Scanning</span>
                <Switch checked={currentPerms.takeAttendance} disabled={selectedRole === 'admin'} onChange={() => handleToggle('takeAttendance')} />
              </div>
              <div className="flex items-center justify-between p-3 border-b border-white/5">
                <span className="text-gray-300 text-sm">Modify Global Settings</span>
                <Switch checked={currentPerms.editSettings} disabled={selectedRole === 'admin'} onChange={() => handleToggle('editSettings')} />
              </div>
              <div className="flex items-center justify-between p-3 border-b border-white/5">
                <span className="text-gray-300 text-sm">Manage Academic Structures</span>
                <Switch checked={currentPerms.manageAcademic} disabled={selectedRole === 'admin'} onChange={() => handleToggle('manageAcademic')} />
              </div>
              <div className="flex items-center justify-between p-3 border-b border-white/5">
                <span className="text-gray-300 text-sm">Approve/Reject Admissions</span>
                <Switch checked={currentPerms.manageAdmissions} disabled={selectedRole === 'admin'} onChange={() => handleToggle('manageAdmissions')} />
              </div>
            </div>

            <div className="pt-8">
              <Button disabled={!hasChanges || selectedRole === 'admin'}>Save Permissions</Button>
            </div>
          </CardContent>
        </Card>
      </div>

      <Modal
        isOpen={isAddModalOpen}
        onClose={() => setIsAddModalOpen(false)}
        title="Add New Role"
      >
        <form onSubmit={handleAddRole} className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">Role Name</label>
            <Input 
              value={newRoleName}
              onChange={(e) => setNewRoleName(e.target.value)}
              placeholder="e.g. Guest Instructor"
              autoFocus
            />
          </div>
          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">Description</label>
            <Input 
              value={newRoleDesc}
              onChange={(e) => setNewRoleDesc(e.target.value)}
              placeholder="Brief description of responsibilities"
            />
          </div>
          <div className="flex justify-end gap-3 pt-4">
            <Button type="button" variant="ghost" onClick={() => setIsAddModalOpen(false)}>
              Cancel
            </Button>
            <Button type="submit" disabled={!newRoleName.trim()}>
              Create Role
            </Button>
          </div>
        </form>
      </Modal>
    </div>
  );
}
