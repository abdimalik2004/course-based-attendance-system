import { useState, useEffect } from 'react';
import { Plus, Search, Edit2, Trash2, Link2, Link2Off } from 'lucide-react';
import { useSearchParams } from 'react-router-dom';
import { useHrStore } from '@/store/useHrStore';
import type { Teacher } from '@/services/hrService';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { Card, CardContent } from '@/components/ui/Card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/Table';
import { TeacherModal } from '@/components/hr/TeacherModal';
import { ViewButton } from '@/components/ui/ViewButton';
import { ViewModal } from '@/components/ui/ViewModal';
import { ConfirmDeleteModal } from '@/components/academia/ConfirmDeleteModal';

export default function Teachers() {
  const { teachers, faculties, departments, fetchAll, deleteTeacher, isLoading } = useHrStore();
  const [modalState, setModalState] = useState<{ isOpen: boolean; mode: 'create' | 'edit'; record: Teacher | null }>({
    isOpen: false,
    mode: 'create',
    record: null
  });
  const [viewModalState, setViewModalState] = useState<{ isOpen: boolean; record: Teacher | null }>({
    isOpen: false,
    record: null
  });
  const [searchParams] = useSearchParams();
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>(searchParams.get('status') ?? 'All');
  const [deleteTarget, setDeleteTarget] = useState<Teacher | null>(null);

  // Sync status filter if URL param changes (e.g. back-navigate from dashboard)
  useEffect(() => {
    setStatusFilter(searchParams.get('status') ?? 'All');
  }, [searchParams]);

  useEffect(() => {
    fetchAll();
  }, [fetchAll]);

  const filteredTeachers = teachers.filter(t => {
    const matchesSearch =
      t.fullName.toLowerCase().includes(searchQuery.toLowerCase()) ||
      t.teacherNumber.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesStatus = statusFilter === 'All' || t.status === statusFilter;
    return matchesSearch && matchesStatus;
  });

  const getFacultyName = (id: string) => faculties.find(f => f.id === id)?.name || id;
  const getDepartmentName = (id: string) => departments.find(d => d.id === id)?.name || id;

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Teachers Management</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">Manage all teaching staff across faculties.</p>
        </div>
        <Button onClick={() => setModalState({ isOpen: true, mode: 'create', record: null })} className="w-full sm:w-auto">
          <Plus size={18} className="mr-2" />
          Add New Teacher
        </Button>
      </div>

      <Card className="glass-card shadow-2xl shadow-primary/5">
        {/* Table Toolbar */}
        <div className="p-4 border-b border-gray-200 dark:border-white/10 flex flex-col sm:flex-row gap-4 justify-between items-center bg-gray-50/50 dark:bg-white/5">
          <div className="flex flex-col sm:flex-row gap-3 w-full sm:w-auto">
            <div className="relative w-full sm:w-72">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-gray-500">
                <Search size={16} />
              </div>
              <Input
                type="text"
                placeholder="Search by name or T-NO..."
                className="pl-9 bg-white dark:bg-dark-card"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="w-full sm:w-44 rounded-xl border border-gray-300 dark:border-white/10 bg-white dark:bg-dark-card px-3 py-2 text-sm text-gray-900 dark:text-white focus:border-primary focus:ring-2 focus:ring-primary/20 outline-none"
            >
              <option value="All">All Statuses</option>
              <option value="Active">Active</option>
              <option value="On Leave">On Leave</option>
              <option value="Inactive">Inactive</option>
            </select>
          </div>
          <span className="text-xs text-gray-400 shrink-0">{filteredTeachers.length} teacher{filteredTeachers.length !== 1 ? 's' : ''}</span>
        </div>

        {/* Table Wrapper */}
        <CardContent className="p-0">
          <div className="overflow-x-auto custom-scrollbar w-full">
            <Table className="w-full whitespace-nowrap min-w-max">
              <TableHeader>
                <TableRow>
                  <TableHead>T-NO</TableHead>
                  <TableHead>Name</TableHead>
                  <TableHead>Faculty</TableHead>
                  <TableHead>Department</TableHead>
                  <TableHead>Role</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {isLoading ? (
                  Array.from({ length: 5 }).map((_, i) => (
                    <TableRow key={`skeleton-${i}`}>
                      <TableCell><div className="h-4 w-24 bg-gray-200 dark:bg-white/10 rounded animate-pulse" /></TableCell>
                      <TableCell><div className="h-4 w-32 bg-gray-200 dark:bg-white/10 rounded animate-pulse" /></TableCell>
                      <TableCell><div className="h-4 w-24 bg-gray-200 dark:bg-white/10 rounded animate-pulse" /></TableCell>
                      <TableCell><div className="h-4 w-32 bg-gray-200 dark:bg-white/10 rounded animate-pulse" /></TableCell>
                      <TableCell><div className="h-4 w-24 bg-gray-200 dark:bg-white/10 rounded animate-pulse" /></TableCell>
                      <TableCell><div className="h-6 w-20 bg-gray-200 dark:bg-white/10 rounded-full animate-pulse" /></TableCell>
                      <TableCell><div className="h-8 w-24 bg-gray-200 dark:bg-white/10 rounded animate-pulse ml-auto" /></TableCell>
                    </TableRow>
                  ))
                ) : filteredTeachers.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={7} className="h-24 text-center text-gray-500">
                      No teachers found.
                    </TableCell>
                  </TableRow>
                ) : (
                  filteredTeachers.map((teacher) => (
                    <TableRow key={teacher.id}>
                      <TableCell className="font-medium text-gray-900 dark:text-white font-mono">
                        {teacher.teacherNumber || teacher.id}
                      </TableCell>
                      <TableCell>
                        {teacher.fullName}
                      </TableCell>
                      <TableCell>
                        <div className="text-gray-900 dark:text-gray-300">
                          {getFacultyName(teacher.facultyId)}
                        </div>
                      </TableCell>
                      <TableCell>
                        <div className="text-gray-900 dark:text-gray-300">
                          {getDepartmentName(teacher.departmentId)}
                        </div>
                      </TableCell>
                      <TableCell>
                        <div className="text-sm text-gray-700 dark:text-gray-300">
                          {teacher.role}
                        </div>
                      </TableCell>
                      <TableCell>
                        <span className={`px-2.5 py-1 rounded-full text-xs font-medium border ${
                          teacher.status === 'Active' 
                            ? 'bg-emerald-50 text-emerald-700 border-emerald-200 dark:bg-emerald-500/10 dark:text-emerald-400 dark:border-emerald-500/20' 
                            : teacher.status === 'On Leave'
                            ? 'bg-amber-50 text-amber-700 border-amber-200 dark:bg-amber-500/10 dark:text-amber-400 dark:border-amber-500/20'
                            : 'bg-rose-50 text-rose-700 border-rose-200 dark:bg-rose-500/10 dark:text-rose-400 dark:border-rose-500/20'
                        }`}>
                          {teacher.status}
                        </span>
                      </TableCell>
                      <TableCell className="text-right">
                        <div className="flex items-center justify-end gap-2">
                          <ViewButton onClick={() => setViewModalState({ isOpen: true, record: teacher})} tooltip="View" />
                          <button 
                            onClick={() => setModalState({ isOpen: true, mode: 'edit', record: teacher })}
                            className="p-1.5 rounded-lg text-amber-500 hover:text-amber-600 hover:bg-amber-50 dark:hover:bg-amber-500/10 transition-colors" 
                            title="Edit"
                          >
                            <Edit2 size={16} />
                          </button>
                          <button
                            onClick={() => setDeleteTarget(teacher)}
                            className="p-1.5 rounded-lg text-red-500 hover:text-red-600 hover:bg-red-50 dark:hover:bg-red-500/10 transition-colors"
                            title="Delete"
                          >
                            <Trash2 size={16} />
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

      <TeacherModal
        isOpen={modalState.isOpen}
        mode={modalState.mode}
        record={modalState.record}
        onClose={() => setModalState(prev => ({ ...prev, isOpen: false }))}
        onSave={(updated) => setModalState(prev => ({ ...prev, record: updated }))}
      />

      <ViewModal
        isOpen={viewModalState.isOpen}
        onClose={() => setViewModalState(prev => ({ ...prev, isOpen: false }))}
        title="Teacher Details"
        data={viewModalState.record ? [
          { label: 'T-NO', value: viewModalState.record.teacherNumber || viewModalState.record.id },
          { label: 'Name', value: viewModalState.record.fullName },
          { label: 'Faculty', value: getFacultyName(viewModalState.record.facultyId) },
          { label: 'Department', value: getDepartmentName(viewModalState.record.departmentId) },
          { label: 'Role', value: viewModalState.record.role },
          ...(viewModalState.record.phone ? [{ label: 'Phone', value: viewModalState.record.phone }] : []),
          ...(viewModalState.record.email ? [{ label: 'Email', value: viewModalState.record.email }] : []),
          ...(viewModalState.record.hireDate ? [{ label: 'Hire Date', value: new Date(viewModalState.record.hireDate).toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' }) }] : []),
          { label: 'Login Account', value: viewModalState.record.linkedUsername ? (
            <span className="flex items-center gap-1.5 text-sm font-medium text-primary">
              <Link2 size={13} />
              @{viewModalState.record.linkedUsername}
            </span>
          ) : (
            <span className="flex items-center gap-1.5 text-sm text-gray-400 dark:text-gray-500">
              <Link2Off size={13} />
              Not linked
            </span>
          )},
          { label: 'Status', value: (
            <span className={`px-2.5 py-1 rounded-full text-xs font-medium border ${
              viewModalState.record.status === 'Active'
                ? 'bg-emerald-50 text-emerald-700 border-emerald-200 dark:bg-emerald-500/10 dark:text-emerald-400 dark:border-emerald-500/20'
                : viewModalState.record.status === 'On Leave'
                ? 'bg-amber-50 text-amber-700 border-amber-200 dark:bg-amber-500/10 dark:text-amber-400 dark:border-amber-500/20'
                : 'bg-rose-50 text-rose-700 border-rose-200 dark:bg-rose-500/10 dark:text-rose-400 dark:border-rose-500/20'
            }`}>
              {viewModalState.record.status}
            </span>
          )}
        ] : null}
      />

      <ConfirmDeleteModal
        isOpen={!!deleteTarget}
        onClose={() => setDeleteTarget(null)}
        onConfirm={async () => {
          if (deleteTarget) await deleteTeacher(deleteTarget.id);
        }}
        title="Delete Teacher"
        message={deleteTarget
          ? `Are you sure you want to delete "${deleteTarget.fullName}" (${deleteTarget.teacherNumber || deleteTarget.id})? This action cannot be undone.`
          : undefined}
      />
    </div>
  );
}
