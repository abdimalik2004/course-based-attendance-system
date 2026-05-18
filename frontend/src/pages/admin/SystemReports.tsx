import { useEffect, useState, useRef } from 'react';
import { useVirtualizer } from '@tanstack/react-virtual';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { Users, GraduationCap, Building2, Activity, Search, FileText, Download } from 'lucide-react';
import jsPDF from 'jspdf';
import autoTable from 'jspdf-autotable';

import { Card, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Select } from '@/components/ui/Select';
import { Input } from '@/components/ui/Input';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/Table';
import { Badge } from '@/components/ui/Badge';
import { StatCard } from '@/components/ui/StatCard';
import { ExportButtons } from '@/components/ui/ExportButtons';
import { useReportSummary, useAbsenceRanking, useAttendanceChartData, useDistributionSummary } from '@/hooks/queries/useReports';

export default function SystemReports() {
  const { data: summary, isLoading: isLoadingSummary } = useReportSummary();
  const [page, setPage] = useState(1);
  const [limit] = useState(10);
  const [reportType, setReportType] = useState('');
  const [faculty, setFaculty] = useState('all');
  const [department, setDepartment] = useState('all');
  const [course, setCourse] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  
  const [appliedFilters, setAppliedFilters] = useState({
    search: '', type: '', faculty: 'all', department: 'all', course: 'all'
  });

  useEffect(() => {
    const handler = setTimeout(() => {
      setAppliedFilters(prev => {
        if (prev.search !== searchQuery) {
          setPage(1);
          return { ...prev, search: searchQuery };
        }
        return prev;
      });
    }, 300);
    return () => clearTimeout(handler);
  }, [searchQuery]);

  const { data: absenceData, isLoading: isLoadingAbsence } = useAbsenceRanking({
    page,
    limit,
    ...appliedFilters
  });
  const displayedRecords = absenceData?.data || [];
  const totalRecords = absenceData?.total || 0;

  const { data: chartData = [], isLoading: isLoadingChart } = useAttendanceChartData();
  const { data: distribution, isLoading: isLoadingDistribution } = useDistributionSummary();
  const isLoading = isLoadingSummary || isLoadingAbsence || isLoadingChart || isLoadingDistribution;
  const tableContainerRef = useRef<HTMLDivElement>(null);


  const rowVirtualizer = useVirtualizer({
    count: displayedRecords.length,
    getScrollElement: () => tableContainerRef.current,
    estimateSize: () => 53,
    overscan: 5,
  });

  const handleGenerateReport = () => {
    setPage(1);
    setAppliedFilters(prev => ({
      ...prev,
      type: reportType,
      faculty,
      department,
      course
    }));
  };

  const handleReset = () => {
    setReportType('');
    setFaculty('all');
    setDepartment('all');
    setCourse('all');
    setSearchQuery('');
    setPage(1);
    setAppliedFilters({
      search: '', type: '', faculty: 'all', department: 'all', course: 'all'
    });
  };

  const handleExportPDF = () => {
    // Server-side PDF export
    const queryParams = new URLSearchParams(appliedFilters as Record<string, string>).toString();
    window.open(`/api/reports/export?format=pdf&${queryParams}`, '_blank');
  };

  const handleExportCSV = () => {
    // Server-side CSV export
    const queryParams = new URLSearchParams(appliedFilters as Record<string, string>).toString();
    window.open(`/api/reports/export?format=csv&${queryParams}`, '_blank');
  };

  const handlePrint = () => {
    window.print();
  };

  const handleDownloadSingle = (record: any) => {
    const doc = new jsPDF();
    doc.text(`Report - ${record.studentName}`, 14, 15);
    
    autoTable(doc, {
      head: [['Field', 'Value']],
      body: [
        ['Name', record.studentName],
        ['Type', record.type],
        ['Department / Course', record.facultyOrDepartment],
        ['Total Absences', `${record.totalAbsences} days`],
        ['Attendance', `${record.attendancePercentage}%`],
        ['Status', record.status]
      ],
      startY: 20,
    });

    doc.save(`${record.studentName.replace(/\s+/g, '_')}_report.pdf`);
  };

  // Badge logic based on status
  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'Active': return <Badge variant="success">Active</Badge>;
      case 'Low': return <Badge variant="success">Low</Badge>;
      case 'Medium': return <Badge variant="warning">Medium</Badge>;
      case 'High': return <Badge variant="danger">High</Badge>;
      case 'Inactive': return <Badge variant="default">Inactive</Badge>;
      default: return <Badge variant="default">{status}</Badge>;
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col gap-1">
        <h1 className="text-3xl font-bold tracking-tight text-gray-900 dark:text-white">
          System Reports
        </h1>
        <p className="text-gray-500 dark:text-gray-400">
          Generate and analyze reports across the system
        </p>
      </div>

      {/* Filter Bar */}
      <Card className="glass-card">
        <CardContent className="p-4 sm:p-6 flex flex-col gap-4">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 w-full">
            <div>
              <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1.5 ml-1">Report Type</label>
              <Select 
                options={[
                  { value: '', label: 'Please Choose Report Type' },
                  { value: 'student_attendance', label: 'Student Attendance Report' },
                  { value: 'teacher_attendance', label: 'Teacher Attendance Report' },
                ]}
                value={reportType}
                onChange={(e) => setReportType(e.target.value)}
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1.5 ml-1">Faculty</label>
              <Select 
                options={[
                  { value: 'all', label: 'All Faculties' },
                  { value: 'cs', label: 'Computer Science' },
                ]}
                value={faculty}
                onChange={(e) => setFaculty(e.target.value)}
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1.5 ml-1">Department</label>
              <Select 
                options={[
                  { value: 'all', label: 'All Departments' },
                  { value: 'se', label: 'Software Engineering' },
                ]}
                value={department}
                onChange={(e) => setDepartment(e.target.value)}
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1.5 ml-1">Course</label>
              <Select 
                options={[
                  { value: 'all', label: 'All Courses' },
                  { value: 'math', label: 'Mathematics' },
                  { value: 'physics', label: 'Physics' },
                ]}
                value={course}
                onChange={(e) => setCourse(e.target.value)}
              />
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
              <Button variant="ghost" className="w-full sm:w-auto" onClick={handleReset}>Reset</Button>
              <Button className="w-full sm:w-auto whitespace-nowrap" onClick={handleGenerateReport}>Generate Report</Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Summary KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard 
          title="Total Students" 
          value={isLoading ? '-' : `+${summary?.totalStudents?.toLocaleString()}`}
          icon={Users} 
          iconColor="primary" 
        />
        <StatCard 
          title="Total Teachers" 
          value={isLoading ? '-' : summary?.totalTeachers || 0} 
          icon={GraduationCap} 
          iconColor="success" 
        />
        <StatCard 
          title="Total Faculties" 
          value={isLoading ? '-' : summary?.totalFaculties || 0} 
          icon={Building2} 
          iconColor="warning" 
        />
        <StatCard 
          title="Attendance Rate" 
          value={isLoading ? '-' : `${summary?.attendanceRate}%`} 
          icon={Activity} 
          iconColor="primary" 
        />
      </div>

      {/* Analytics Section */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card className="glass-card lg:col-span-2">
          <CardContent className="p-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">Attendance Overview</h3>
            <div className="h-[300px] w-full mt-4">
              {isLoading ? (
                <div className="w-full h-full bg-gray-200 dark:bg-white/5 animate-pulse rounded-xl" />
              ) : (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={false} />
                    <XAxis dataKey="name" stroke="#6B7280" tick={{ fill: '#6B7280', fontSize: 12 }} axisLine={false} tickLine={false} />
                    <YAxis stroke="#6B7280" tick={{ fill: '#6B7280', fontSize: 12 }} axisLine={false} tickLine={false} tickFormatter={(val) => `${val}%`} />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#1F2937', borderColor: '#374151', borderRadius: '8px', color: '#fff' }}
                      itemStyle={{ color: '#60A5FA' }}
                    />
                    <Line type="monotone" dataKey="value" stroke="#3B82F6" strokeWidth={3} dot={{ r: 4, fill: '#3B82F6', strokeWidth: 2, stroke: '#0B0F19' }} activeDot={{ r: 6 }} />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </div>
          </CardContent>
        </Card>

        <Card className="glass-card">
          <CardContent className="p-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">Distribution Summary</h3>
            <div className="grid grid-cols-3 gap-2 mb-8 border-b border-gray-200 dark:border-white/5 pb-6">
              <div>
                <p className="text-xs text-gray-600 dark:text-gray-400 mb-1">Students</p>
                <div className="flex items-center gap-2">
                  <div className="w-1.5 h-4 bg-primary rounded-full"></div>
                  <span className="text-xl font-bold text-gray-900 dark:text-white">{distribution?.students}%</span>
                </div>
              </div>
              <div>
                <p className="text-xs text-gray-600 dark:text-gray-400 mb-1">Teachers</p>
                <div className="flex items-center gap-2">
                  <div className="w-1.5 h-4 bg-emerald-500 rounded-full"></div>
                  <span className="text-xl font-bold text-gray-900 dark:text-white">{distribution?.teachers}</span>
                </div>
              </div>
              <div>
                <p className="text-xs text-gray-600 dark:text-gray-400 mb-1">Faculties</p>
                <div className="flex items-center gap-2">
                  <div className="w-1.5 h-4 bg-amber-500 rounded-full"></div>
                  <span className="text-xl font-bold text-gray-900 dark:text-white">{distribution?.faculties}</span>
                </div>
              </div>
            </div>

            <div className="h-[180px] w-full">
              {isLoading ? (
                <div className="w-full h-full bg-gray-200 dark:bg-white/5 animate-pulse rounded-xl" />
              ) : (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={[
                    { name: 'Students', value: distribution?.students || 0, fill: '#3B82F6' },
                    { name: 'Teachers', value: distribution?.teachers || 0, fill: '#10B981' },
                    { name: 'Faculties', value: distribution?.faculties || 0, fill: '#F59E0B' }
                  ]} margin={{ top: 0, right: 0, bottom: 0, left: -20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                    <XAxis dataKey="name" stroke="#6B7280" tick={{ fill: '#6B7280', fontSize: 11 }} axisLine={false} tickLine={false} />
                    <YAxis stroke="#6B7280" tick={{ fill: '#6B7280', fontSize: 11 }} axisLine={false} tickLine={false} />
                    <Tooltip cursor={{ fill: 'rgba(255,255,255,0.05)' }} contentStyle={{ backgroundColor: '#1F2937', borderColor: '#374151', borderRadius: '8px', color: '#fff' }} />
                    <Bar dataKey="value" radius={[4, 4, 0, 0]} barSize={30} />
                  </BarChart>
                </ResponsiveContainer>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main Table: Absence Ranking */}
      <Card className="glass-card">
        <div className="p-4 sm:p-6 border-b border-gray-100 dark:border-white/5 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
          <div className="flex items-center gap-2">
            <FileText className="text-primary" size={24} />
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white">Absence Ranking</h2>
          </div>
          <div className="flex items-center gap-4 w-full sm:w-auto">
            <div className="relative w-full sm:w-64">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" size={16} />
              <Input 
                placeholder="Search records..." 
                className="pl-9 h-10 border-gray-200 bg-white dark:bg-white/5 dark:border-white/10" 
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>
            <ExportButtons 
              className="hidden md:flex" 
              onExportPDF={handleExportPDF} 
              onExportCSV={handleExportCSV} 
              onPrint={handlePrint} 
            />
          </div>
        </div>
        <CardContent className="p-0">
           <ExportButtons 
             className="flex md:hidden p-4 border-b border-gray-200 dark:border-white/5 justify-end" 
             onExportPDF={handleExportPDF} 
             onExportCSV={handleExportCSV} 
             onPrint={handlePrint} 
           />
          <div ref={tableContainerRef} className="overflow-auto custom-scrollbar h-[400px] relative w-full">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>No.</TableHead>
                  <TableHead>Student Name</TableHead>
                  <TableHead>Type</TableHead>
                  <TableHead>Department / Course</TableHead>
                  <TableHead>Total Absences</TableHead>
                  <TableHead>Attendance %</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody
                style={{
                  height: `${rowVirtualizer.getTotalSize()}px`,
                  position: 'relative'
                }}
              >
                {isLoading ? (
                  Array.from({ length: 5 }).map((_, i) => (
                    <TableRow key={`skeleton-row-${i}`}>
                      <TableCell><div className="h-4 w-6 bg-gray-200 dark:bg-white/10 rounded animate-pulse" /></TableCell>
                      <TableCell><div className="h-4 w-24 bg-gray-200 dark:bg-white/10 rounded animate-pulse" /></TableCell>
                      <TableCell><div className="h-4 w-16 bg-gray-200 dark:bg-white/10 rounded animate-pulse" /></TableCell>
                      <TableCell><div className="h-4 w-32 bg-gray-200 dark:bg-white/10 rounded animate-pulse" /></TableCell>
                      <TableCell><div className="h-4 w-12 bg-gray-200 dark:bg-white/10 rounded animate-pulse" /></TableCell>
                      <TableCell><div className="h-4 w-12 bg-gray-200 dark:bg-white/10 rounded animate-pulse" /></TableCell>
                      <TableCell><div className="h-6 w-16 bg-gray-200 dark:bg-white/10 rounded-full animate-pulse" /></TableCell>
                      <TableCell className="text-right"><div className="h-8 w-8 ml-auto bg-gray-200 dark:bg-white/10 rounded animate-pulse" /></TableCell>
                    </TableRow>
                  ))
                ) : displayedRecords.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={8} className="h-24 text-center text-gray-500">
                      No records found.
                    </TableCell>
                  </TableRow>
                ) : (
(() => {
                    const virtualItems = rowVirtualizer.getVirtualItems();
                    const paddingTop = virtualItems.length > 0 ? virtualItems[0]?.start || 0 : 0;
                    const paddingBottom = virtualItems.length > 0
                      ? rowVirtualizer.getTotalSize() - (virtualItems[virtualItems.length - 1]?.end || 0)
                      : 0;

                    return (
                      <>
                        {paddingTop > 0 && (
                          <TableRow>
                            <TableCell style={{ height: `${paddingTop}px` }} colSpan={8} />
                          </TableRow>
                        )}
                        {virtualItems.map((virtualRow) => {
                          const index = virtualRow.index;
                          const record = displayedRecords[index];
                          return (
                            <TableRow key={record.id} style={{ height: `${virtualRow.size}px` }}>
                              <TableCell className="text-gray-500">{index + 1}</TableCell>
                              <TableCell className="font-medium text-gray-900 dark:text-white">
                                {record.studentName}
                              </TableCell>
                              <TableCell className="text-gray-500 dark:text-gray-400">
                                {record.type}
                              </TableCell>
                              <TableCell className="text-gray-500 dark:text-gray-400">
                                {record.facultyOrDepartment}
                              </TableCell>
                              <TableCell>
                                <span className="font-bold text-gray-900 dark:text-white">{record.totalAbsences} days</span>
                              </TableCell>
                              <TableCell className="text-gray-500 dark:text-gray-400">
                                {record.attendancePercentage}%
                              </TableCell>
                              <TableCell>
                                {getStatusBadge(record.status)}
                              </TableCell>
                              <TableCell className="text-right">
                                <Button 
                                  variant="ghost" 
                                  size="sm" 
                                  className="text-primary hover:text-primary-600 hover:bg-primary-50"
                                  onClick={() => handleDownloadSingle(record)}
                                >
                                  <Download size={16} />
                                </Button>
                              </TableCell>
                            </TableRow>
                          );
                        })}
                        {paddingBottom > 0 && (
                          <TableRow>
                            <TableCell style={{ height: `${paddingBottom}px` }} colSpan={8} />
                          </TableRow>
                        )}
                      </>
                    );
                  })()
                )}
              </TableBody>
            </Table>
          </div>
          <div className="p-4 border-t border-gray-100 dark:border-white/5 flex items-center justify-between">
             <span className="text-sm text-gray-500">Showing {displayedRecords.length} of {totalRecords} records</span>
             <div className="flex gap-2">
               <Button variant="secondary" size="sm" onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page === 1}>Previous</Button>
               <Button variant="secondary" size="sm" onClick={() => setPage(p => p + 1)} disabled={displayedRecords.length < limit}>Next</Button>
             </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
