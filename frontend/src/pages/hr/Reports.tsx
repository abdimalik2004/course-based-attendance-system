import { useState, useEffect } from 'react';
import { Filter, FileText } from 'lucide-react';
import { useHrStore } from '@/store/useHrStore';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { Card, CardContent } from '@/components/ui/Card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/Table';
import { Badge } from '@/components/ui/Badge';

export default function Reports() {
  const { teachers, faculties, departments, fetchTeachers, fetchFaculties, fetchDepartments, isLoading } = useHrStore();
  
  const [filterFaculty, setFilterFaculty] = useState('All');
  const [filterDepartment, setFilterDepartment] = useState('All');
  const [filterRole, setFilterRole] = useState('All');

  useEffect(() => {
    fetchTeachers();
    fetchFaculties();
    fetchDepartments();
  }, [fetchTeachers, fetchFaculties, fetchDepartments]);

  const getFacultyName = (id: string) => faculties.find(f => f.id === id)?.name || id;
  const getDepartmentName = (id: string) => departments.find(d => d.id === id)?.name || id;

  const filteredTeachers = teachers.filter(t => {
    if (filterFaculty !== 'All' && t.facultyId !== filterFaculty) return false;
    if (filterDepartment !== 'All' && t.departmentId !== filterDepartment) return false;
    if (filterRole !== 'All' && t.role !== filterRole) return false;
    return true;
  });

  const availableDepartments = filterFaculty === 'All' 
    ? departments 
    : departments.filter(d => d.facultyId === filterFaculty);

  const roles = Array.from(new Set(teachers.map((teacher) => teacher.role)));

  const handleExport = () => {
    alert('Exporting live report as CSV...');
  };

  const getPerformance = (id: string) => {
    const teacher = teachers.find((item) => item.id === id);
    if (!teacher) return '0/0';
    if (teacher.status === 'Active') return '10/10';
    if (teacher.status === 'On Leave') return '7/10';
    return '0/10';
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">HR Reports</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">Generate and export staff reports.</p>
        </div>
      </div>

      <Card className="glass-card shadow-2xl shadow-primary/5">
        {/* Filters Section */}
        <div className="p-4 border-b border-gray-200 dark:border-white/10 bg-gray-50/50 dark:bg-white/5 space-y-4">
          <div className="flex items-center gap-2 mb-2 text-sm font-medium text-gray-700 dark:text-gray-300">
            <Filter size={16} /> Filters
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 w-full">
            {/* Faculty Filter */}
            <div>
              <label className="block text-xs text-gray-500 mb-1">Faculty</label>
              <select 
                value={filterFaculty}
                onChange={(e) => {
                  setFilterFaculty(e.target.value);
                  setFilterDepartment('All'); // Reset department on faculty change
                }}
                className="w-full rounded-xl border border-gray-300 dark:border-white/10 bg-white dark:bg-dark-card px-3 py-2 text-sm text-gray-900 dark:text-white focus:border-primary focus:ring-2 focus:ring-primary/20 outline-none"
              >
                <option value="All">All Faculties</option>
                {faculties.map(f => (
                  <option key={f.id} value={f.id}>{f.name}</option>
                ))}
              </select>
            </div>

            {/* Department Filter */}
            <div>
              <label className="block text-xs text-gray-500 mb-1">Department</label>
              <select 
                value={filterDepartment}
                onChange={(e) => setFilterDepartment(e.target.value)}
                className="w-full rounded-xl border border-gray-300 dark:border-white/10 bg-white dark:bg-dark-card px-3 py-2 text-sm text-gray-900 dark:text-white focus:border-primary focus:ring-2 focus:ring-primary/20 outline-none"
                disabled={filterFaculty !== 'All' && availableDepartments.length === 0}
              >
                <option value="All">All Departments</option>
                {availableDepartments.map(d => (
                  <option key={d.id} value={d.id}>{d.name}</option>
                ))}
              </select>
            </div>

            {/* Role Filter */}
            <div>
              <label className="block text-xs text-gray-500 mb-1">Role</label>
              <select 
                value={filterRole}
                onChange={(e) => setFilterRole(e.target.value)}
                className="w-full rounded-xl border border-gray-300 dark:border-white/10 bg-white dark:bg-dark-card px-3 py-2 text-sm text-gray-900 dark:text-white focus:border-primary focus:ring-2 focus:ring-primary/20 outline-none"
              >
                <option value="All">All Roles</option>
                {roles.map(r => (
                  <option key={r} value={r}>{r}</option>
                ))}
              </select>
            </div>
          </div>
          
          <div className="flex flex-col sm:flex-row items-end gap-4 w-full pt-4 border-t border-gray-100 dark:border-white/5">
            <div className="flex-1 w-full sm:w-auto">
              <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1.5 ml-1">Start Date</label>
              <Input type="date" className="text-gray-900 dark:text-white dark:[color-scheme:dark]" />
            </div>
            <div className="flex-1 w-full sm:w-auto">
              <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1.5 ml-1">End Date</label>
              <Input type="date" className="text-gray-900 dark:text-white dark:[color-scheme:dark]" />
            </div>
            <div className="flex items-center justify-end gap-3 w-full sm:w-auto">
              <Button variant="ghost" className="w-full sm:w-auto">Reset</Button>
              <Button className="w-full sm:w-auto whitespace-nowrap" onClick={handleExport}>
                <FileText size={16} className="mr-2" /> Generate Report
              </Button>
            </div>
          </div>
        </div>

        {/* Table */}
        <CardContent className="p-0">
          <div className="overflow-x-auto custom-scrollbar">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Name</TableHead>
                  <TableHead>Role</TableHead>
                  <TableHead>Faculty</TableHead>
                  <TableHead>Department</TableHead>
                  <TableHead>Performance</TableHead>
                  <TableHead>Status</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {isLoading ? (
                  Array.from({ length: 5 }).map((_, i) => (
                    <TableRow key={`skeleton-${i}`}>
                      <TableCell><div className="h-4 w-32 bg-gray-200 dark:bg-white/10 rounded animate-pulse" /></TableCell>
                      <TableCell><div className="h-4 w-24 bg-gray-200 dark:bg-white/10 rounded animate-pulse" /></TableCell>
                      <TableCell><div className="h-4 w-24 bg-gray-200 dark:bg-white/10 rounded animate-pulse" /></TableCell>
                      <TableCell><div className="h-4 w-32 bg-gray-200 dark:bg-white/10 rounded animate-pulse" /></TableCell>
                      <TableCell><div className="h-4 w-12 bg-gray-200 dark:bg-white/10 rounded animate-pulse" /></TableCell>
                      <TableCell><div className="h-6 w-20 bg-gray-200 dark:bg-white/10 rounded-md animate-pulse" /></TableCell>
                    </TableRow>
                  ))
                ) : filteredTeachers.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={6} className="h-24 text-center text-gray-500">
                      <FileText size={32} className="mx-auto mb-3 opacity-20" />
                      No data matching your filters.
                    </TableCell>
                  </TableRow>
                ) : (
                  filteredTeachers.map((teacher) => (
                    <TableRow key={teacher.id}>
                      <TableCell className="font-medium text-gray-900 dark:text-white">
                        {teacher.fullName}
                      </TableCell>
                      <TableCell className="text-gray-600 dark:text-gray-300">
                        {teacher.role}
                      </TableCell>
                      <TableCell className="text-gray-600 dark:text-gray-300">
                        {getFacultyName(teacher.facultyId)}
                      </TableCell>
                      <TableCell className="text-gray-600 dark:text-gray-300">
                        {getDepartmentName(teacher.departmentId)}
                      </TableCell>
                      <TableCell className="font-medium text-gray-900 dark:text-white">
                        {getPerformance(teacher.id)}
                      </TableCell>
                      <TableCell>
                        <Badge 
                          variant={
                            teacher.status === 'Active' ? 'success' : 
                            teacher.status === 'On Leave' ? 'warning' : 'danger'
                          }
                        >
                          {teacher.status}
                        </Badge>
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
