import { useCallback, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Search, Filter, Download, ChevronDown, Check,
  X, ChevronRight, CalendarDays, Clock, FlaskConical, BookOpen, GraduationCap,
  BookOpenCheck, Info, FilePen, CheckCircle2, XCircle, AlertCircle,
} from 'lucide-react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { cn } from '@/utils/cn';
import dashboardService, {
  type StudentAttendanceCourse,
  type StudentSessionRecord,
  type ExcuseRequestItem,
} from '@/services/dashboardService';
import { useClickOutside } from '@/hooks/useClickOutside';

type StatusFilter = 'All' | 'Good' | 'Warning' | 'Low';

const STATUS_OPTIONS: StatusFilter[] = ['All', 'Good', 'Warning', 'Low'];

const STATUS_COLORS: Record<StatusFilter, string> = {
  All: 'bg-gray-400',
  Good: 'bg-emerald-500',
  Warning: 'bg-yellow-500',
  Low: 'bg-rose-500',
};

const getStatusBadge = (status: string) => {
  switch (status?.toString().toUpperCase()) {
    case 'PRESENT':
      return <Badge variant="success">Present</Badge>;
    case 'LATE':
      return <Badge variant="warning">Late</Badge>;
    case 'ABSENT':
      return <Badge variant="danger">Absent</Badge>;
    case 'EXCUSED':
      return <Badge variant="info">Excused</Badge>;
    default:
      return <Badge variant="default">{status ?? 'Unknown'}</Badge>;
  }
};

const getProgressColor = (percent: number) => {
  if (percent >= 85) return 'bg-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.5)]';
  if (percent >= 70) return 'bg-yellow-500 shadow-[0_0_10px_rgba(234,179,8,0.5)]';
  return 'bg-rose-500 shadow-[0_0_10px_rgba(244,63,94,0.5)]';
};

const SESSION_TYPE_ICON: Record<string, React.ReactNode> = {
  Lecture: <BookOpen size={13} />,
  Lab: <FlaskConical size={13} />,
  Tutorial: <GraduationCap size={13} />,
};

const SESSION_TYPE_COLOR: Record<string, string> = {
  Lecture: 'bg-blue-50 text-blue-700 dark:bg-blue-500/10 dark:text-blue-300 border border-blue-100 dark:border-blue-500/20',
  Lab: 'bg-purple-50 text-purple-700 dark:bg-purple-500/10 dark:text-purple-300 border border-purple-100 dark:border-purple-500/20',
  Tutorial: 'bg-teal-50 text-teal-700 dark:bg-teal-500/10 dark:text-teal-300 border border-teal-100 dark:border-teal-500/20',
};

function exportToCSV(rows: StudentAttendanceCourse[]) {
  const headers = ['Course', 'Code', 'Attended', 'Total', 'Absent', 'Excused', 'Percentage', 'Status', 'Last Updated'];
  const lines = rows.map((c) => [
    `"${(c.course_name ?? '').replace(/"/g, '""')}"`,
    `"${c.course_code ?? ''}"`,
    c.classes_attended ?? 0,
    c.total_classes ?? 0,
    c.classes_absent ?? 0,
    c.classes_excused ?? 0,
    `${Math.round(c.attendance_percentage ?? 0)}%`,
    `"${c.status ?? ''}"`,
    c.last_updated ? new Date(c.last_updated).toLocaleDateString() : '-',
  ].join(','));
  const csv = [headers.join(','), ...lines].join('\n');
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `attendance_${new Date().toISOString().slice(0, 10)}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

function formatDate(iso: string) {
  const d = new Date(iso);
  return d.toLocaleDateString(undefined, { weekday: 'short', month: 'short', day: 'numeric' });
}

// ── Excuse status helpers ───────────────────────────────────────────────────
const EXCUSE_STATUS_CONFIG = {
  PENDING: { label: 'Pending', icon: AlertCircle, color: 'text-amber-600 dark:text-amber-400', bg: 'bg-amber-50 dark:bg-amber-500/10 border-amber-200 dark:border-amber-500/20' },
  APPROVED: { label: 'Approved', icon: CheckCircle2, color: 'text-emerald-600 dark:text-emerald-400', bg: 'bg-emerald-50 dark:bg-emerald-500/10 border-emerald-200 dark:border-emerald-500/20' },
  DENIED: { label: 'Denied', icon: XCircle, color: 'text-rose-600 dark:text-rose-400', bg: 'bg-rose-50 dark:bg-rose-500/10 border-rose-200 dark:border-rose-500/20' },
} as const;

// ── Custom course picker (replaces native <select> for dark-mode support) ───
interface CoursPickerProps {
  courses: StudentAttendanceCourse[];
  value: string;
  onChange: (v: string) => void;
}

function CoursePicker({ courses, value, onChange }: CoursPickerProps) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const closeIt = useCallback(() => setOpen(false), []);
  useClickOutside(ref, closeIt);

  const selected = courses.find((c) => String(c.id) === value);
  const label = selected ? `${selected.course_code} — ${selected.course_name}` : 'All courses that day';

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between rounded-xl border border-gray-200 dark:border-white/10 bg-gray-50 dark:bg-white/5 px-3 py-2.5 text-sm text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-primary/40 transition-colors"
      >
        <span className="truncate">{label}</span>
        <ChevronDown size={14} className={cn('ml-2 shrink-0 text-gray-400 transition-transform', open && 'rotate-180')} />
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -4, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -4, scale: 0.97 }}
            transition={{ duration: 0.13 }}
            className="absolute left-0 right-0 top-full mt-1.5 z-[60] rounded-xl border border-gray-200 dark:border-white/10 bg-white dark:bg-[#1a1d26] shadow-xl overflow-hidden"
          >
            <div className="py-1 max-h-44 overflow-y-auto custom-scrollbar">
              {/* "All courses" option */}
              <button
                type="button"
                onClick={() => { onChange(''); setOpen(false); }}
                className={cn(
                  'w-full text-left px-3 py-2.5 text-sm transition-colors',
                  value === ''
                    ? 'bg-primary/10 text-primary dark:text-primary-accent font-semibold'
                    : 'text-gray-700 dark:text-gray-200 hover:bg-gray-50 dark:hover:bg-white/5',
                )}
              >
                All courses that day
              </button>
              {courses.map((c) => (
                <button
                  key={c.id}
                  type="button"
                  onClick={() => { onChange(String(c.id)); setOpen(false); }}
                  className={cn(
                    'w-full text-left px-3 py-2.5 text-sm transition-colors',
                    value === String(c.id)
                      ? 'bg-primary/10 text-primary dark:text-primary-accent font-semibold'
                      : 'text-gray-700 dark:text-gray-200 hover:bg-gray-50 dark:hover:bg-white/5',
                  )}
                >
                  <span className="font-medium text-primary dark:text-primary-accent">{c.course_code}</span>
                  <span className="text-gray-400 dark:text-gray-500 mx-1">—</span>
                  {c.course_name}
                </button>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ── Request Excuse modal ────────────────────────────────────────────────────
interface ExcuseModalProps {
  courses: StudentAttendanceCourse[];
  onClose: () => void;
  onSuccess: () => void;
}

function ExcuseModal({ courses, onClose, onSuccess }: ExcuseModalProps) {
  const today = new Date().toISOString().slice(0, 10);
  const [requestDate, setRequestDate] = useState(today);
  const [courseId, setCourseId] = useState<string>(''); // '' = all courses that day
  const [reason, setReason] = useState('');
  const [error, setError] = useState<string | null>(null);

  const mutation = useMutation({
    mutationFn: () =>
      dashboardService.submitExcuseRequest({
        request_date: requestDate,
        course_id: courseId ? Number(courseId) : null,
        reason: reason.trim() || null,
      }),
    onSuccess: () => {
      onSuccess();
      onClose();
    },
    onError: (e: unknown) => {
      const detail = (e as { response?: { data?: { detail?: unknown } } })?.response?.data?.detail;
      const msg = typeof detail === 'string' ? detail : 'Failed to submit request. Please try again.';
      setError(msg);
    },
  });

  return (
    /* Outer: scrollable layer — covers the viewport, lets tall content scroll */
    <div className="fixed inset-0 z-50 overflow-y-auto">
      {/* Backdrop — sits behind everything */}
      <motion.div
        initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black/40 backdrop-blur-sm"
        onClick={onClose}
      />
      {/* Inner: centering shell — min-h-full so short modals are vertically centred,
           but tall modals just start at the top and scroll rather than clipping */}
      <div className="flex min-h-full items-center justify-center p-4">
      {/* Modal card */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95, y: 16 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.95, y: 16 }}
        transition={{ duration: 0.18 }}
        className="relative w-full max-w-md flex flex-col max-h-[90vh] rounded-2xl bg-white dark:bg-dark-bg border border-gray-200 dark:border-white/10 shadow-2xl z-10"
      >
        {/* Header — fixed */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-100 dark:border-white/5 shrink-0">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-lg bg-amber-50 dark:bg-amber-500/10">
              <FilePen size={16} className="text-amber-600 dark:text-amber-400" />
            </div>
            <div>
              <h2 className="font-bold text-gray-900 dark:text-white text-base">Request Excuse</h2>
              <p className="text-xs text-gray-500 dark:text-gray-400">Submit to your faculty for review</p>
            </div>
          </div>
          <button onClick={onClose} className="p-1.5 rounded-lg text-gray-400 hover:bg-gray-100 dark:hover:bg-white/5 transition-colors">
            <X size={18} />
          </button>
        </div>

        {/* Body — scrollable */}
        <div className="px-6 py-5 space-y-4 overflow-y-auto flex-1">
          {error && (
            <div className="rounded-lg border border-rose-200 dark:border-rose-500/20 bg-rose-50 dark:bg-rose-500/10 px-3 py-2 text-sm text-rose-700 dark:text-rose-300">
              {error}
            </div>
          )}

          {/* Date */}
          <div className="space-y-1.5">
            <label className="text-xs font-semibold text-gray-600 dark:text-gray-400 uppercase tracking-wider">
              Absence Date
            </label>
            <input
              type="date"
              value={requestDate}
              max={today}
              onChange={(e) => setRequestDate(e.target.value)}
              className="w-full rounded-xl border border-gray-200 dark:border-white/10 bg-gray-50 dark:bg-white/5 px-3 py-2.5 text-sm text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-primary/40"
            />
          </div>

          {/* Course — custom picker, no native <select> so dark mode works */}
          <div className="space-y-1.5">
            <label className="text-xs font-semibold text-gray-600 dark:text-gray-400 uppercase tracking-wider">
              Course
            </label>
            <CoursePicker courses={courses} value={courseId} onChange={setCourseId} />
            <p className="text-xs text-gray-400 dark:text-gray-500">
              Select a specific course or leave as "All courses" for illness / full-day absence
            </p>
          </div>

          {/* Reason */}
          <div className="space-y-1.5">
            <label className="text-xs font-semibold text-gray-600 dark:text-gray-400 uppercase tracking-wider">
              Reason <span className="font-normal normal-case text-gray-400">(optional)</span>
            </label>
            <textarea
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              placeholder="e.g. Medical appointment, family emergency…"
              rows={3}
              className="w-full resize-none rounded-xl border border-gray-200 dark:border-white/10 bg-gray-50 dark:bg-white/5 px-3 py-2.5 text-sm text-gray-900 dark:text-white placeholder:text-gray-400 dark:placeholder:text-gray-500 focus:outline-none focus:ring-2 focus:ring-primary/40"
            />
          </div>
        </div>

        {/* Footer — fixed */}
        <div className="flex items-center justify-end gap-3 px-6 py-4 border-t border-gray-100 dark:border-white/5 shrink-0">
          <Button variant="secondary" onClick={onClose} disabled={mutation.isPending}>
            Cancel
          </Button>
          <Button
            onClick={() => mutation.mutate()}
            disabled={!requestDate || mutation.isPending}
            className="gap-2"
          >
            {mutation.isPending ? 'Submitting…' : 'Submit Request'}
          </Button>
        </div>
      </motion.div>
      </div>{/* end centering shell */}
    </div>
  );
}

// ── My excuse requests list ─────────────────────────────────────────────────
function MyExcuseRequests() {
  const { data: requests = [], isLoading } = useQuery<ExcuseRequestItem[]>({
    queryKey: ['myExcuseRequests'],
    queryFn: () => dashboardService.myExcuseRequests(),
    staleTime: 1000 * 60,
  });

  if (isLoading) {
    return (
      <div className="space-y-3 animate-pulse">
        {Array.from({ length: 2 }).map((_, i) => (
          <div key={i} className="h-14 rounded-xl bg-gray-100 dark:bg-white/5" />
        ))}
      </div>
    );
  }

  if (requests.length === 0) {
    return (
      <p className="text-sm text-gray-400 dark:text-gray-500 text-center py-6">
        You haven't submitted any excuse requests yet.
      </p>
    );
  }

  return (
    <div className="divide-y divide-gray-100 dark:divide-white/5">
      {requests.map((req) => {
        const cfg = EXCUSE_STATUS_CONFIG[req.status] ?? EXCUSE_STATUS_CONFIG.PENDING;
        const Icon = cfg.icon;
        return (
          <div key={req.id} className="flex items-center gap-4 py-3.5 px-1">
            <div className={cn('p-2 rounded-lg border shrink-0', cfg.bg)}>
              <Icon size={15} className={cfg.color} />
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-gray-800 dark:text-gray-200 truncate">
                {req.course_name ? `${req.course_code} — ${req.course_name}` : 'All courses'}
              </p>
              <p className="text-xs text-gray-400 dark:text-gray-500 mt-0.5">
                {formatDate(req.request_date)}
                {req.reason && <span className="ml-1.5 opacity-70">· {req.reason}</span>}
              </p>
            </div>
            <span className={cn('text-xs font-semibold shrink-0', cfg.color)}>
              {cfg.label}
            </span>
          </div>
        );
      })}
    </div>
  );
}

// ── Session drill-down panel ────────────────────────────────────────────────
interface SessionPanelProps {
  course: StudentAttendanceCourse;
  onClose: () => void;
}

function SessionPanel({ course, onClose }: SessionPanelProps) {
  const { data: sessions = [], isLoading } = useQuery<StudentSessionRecord[]>({
    queryKey: ['studentSessions', course.id],
    queryFn: () => dashboardService.studentSessionHistory(course.id),
    staleTime: 1000 * 60 * 2,
  });

  return (
    <motion.div
      key="session-panel"
      initial={{ x: '100%', opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      exit={{ x: '100%', opacity: 0 }}
      transition={{ type: 'spring', stiffness: 300, damping: 30 }}
      className="fixed inset-y-0 right-0 w-full sm:w-[420px] z-50 flex flex-col
                 bg-white dark:bg-[#0f1117] border-l border-gray-200 dark:border-white/10
                 shadow-2xl"
    >
      {/* Header */}
      <div className="flex items-start justify-between px-5 py-4 border-b border-gray-100 dark:border-white/5 shrink-0">
        <div>
          <p className="text-[11px] font-semibold uppercase tracking-wider text-gray-400 dark:text-gray-500 mb-0.5">
            Session History
          </p>
          <h3 className="font-bold text-gray-900 dark:text-white text-base leading-tight">
            {course.course_name}
          </h3>
          <p className="text-xs text-primary dark:text-primary-accent font-medium mt-0.5">
            {course.course_code}
          </p>
        </div>
        <button
          onClick={onClose}
          className="p-1.5 rounded-lg text-gray-400 hover:text-gray-600 dark:hover:text-gray-200
                     hover:bg-gray-100 dark:hover:bg-white/5 transition-colors mt-0.5 shrink-0"
        >
          <X size={18} />
        </button>
      </div>

      {/* Summary strip */}
      <div className="grid grid-cols-4 divide-x divide-gray-100 dark:divide-white/5 border-b border-gray-100 dark:border-white/5 shrink-0">
        {[
          { label: 'Attended', value: course.classes_attended, color: 'text-emerald-600 dark:text-emerald-400' },
          { label: 'Absent', value: course.classes_absent, color: 'text-rose-600 dark:text-rose-400' },
          { label: 'Excused', value: course.classes_excused, color: 'text-blue-600 dark:text-blue-400' },
          { label: 'Rate', value: `${Math.round(course.attendance_percentage)}%`, color:
              course.attendance_percentage >= 85
                ? 'text-emerald-600 dark:text-emerald-400'
                : course.attendance_percentage >= 70
                  ? 'text-yellow-600 dark:text-yellow-400'
                  : 'text-rose-600 dark:text-rose-400' },
        ].map((s) => (
          <div key={s.label} className="flex flex-col items-center py-3 px-2">
            <span className={cn('text-xl font-bold tabular-nums', s.color)}>{s.value}</span>
            <span className="text-[10px] text-gray-400 dark:text-gray-500 uppercase tracking-wider mt-0.5">{s.label}</span>
          </div>
        ))}
      </div>

      {/* Session list */}
      <div className="flex-1 overflow-y-auto custom-scrollbar">
        {isLoading ? (
          <div className="space-y-3 p-5">
            {Array.from({ length: 5 }).map((_, i) => (
              <div key={i} className="animate-pulse flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-gray-100 dark:bg-white/5 shrink-0" />
                <div className="flex-1 space-y-2">
                  <div className="h-3 w-32 rounded bg-gray-100 dark:bg-white/5" />
                  <div className="h-2.5 w-20 rounded bg-gray-100 dark:bg-white/5" />
                </div>
                <div className="h-5 w-16 rounded-full bg-gray-100 dark:bg-white/5" />
              </div>
            ))}
          </div>
        ) : sessions.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full gap-3 text-gray-400 dark:text-gray-500 py-16 px-8 text-center">
            <CalendarDays size={40} className="opacity-20" />
            <p className="text-sm">No sessions recorded for this course yet.</p>
          </div>
        ) : (
          <div className="divide-y divide-gray-100 dark:divide-white/5">
            {sessions.map((session) => {
              const type = session.session_type ?? 'Lecture';
              return (
                <div
                  key={session.record_id}
                  className="flex items-center gap-3.5 px-5 py-3.5 hover:bg-gray-50/70 dark:hover:bg-white/[0.03] transition-colors"
                >
                  {/* Date block */}
                  <div className="flex flex-col items-center justify-center w-11 h-11 rounded-xl bg-gray-100 dark:bg-white/5 border border-gray-200 dark:border-white/10 shrink-0">
                    <span className="text-[10px] font-semibold uppercase tracking-wider text-gray-400 dark:text-gray-500 leading-none">
                      {new Date(session.date).toLocaleDateString(undefined, { month: 'short' })}
                    </span>
                    <span className="text-lg font-bold text-gray-900 dark:text-white leading-none mt-0.5">
                      {new Date(session.date).getDate()}
                    </span>
                  </div>
                  {/* Info */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <p className="text-sm font-medium text-gray-700 dark:text-gray-200 shrink-0">
                        {formatDate(session.date)}
                      </p>
                      <span className={cn(
                        'inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[11px] font-medium shrink-0',
                        SESSION_TYPE_COLOR[type] ?? SESSION_TYPE_COLOR['Lecture'],
                      )}>
                        {SESSION_TYPE_ICON[type] ?? SESSION_TYPE_ICON['Lecture']}
                        {type}
                      </span>
                    </div>
                    <div className="flex items-center gap-1.5 mt-0.5 text-xs text-gray-400 dark:text-gray-500">
                      <Clock size={11} />
                      <span>{session.start_time}</span>
                      {session.recognized_at && (
                        <>
                          <span className="opacity-40">·</span>
                          <span>Scanned {session.recognized_at}</span>
                        </>
                      )}
                    </div>
                  </div>
                  {/* Status */}
                  <div className="shrink-0">
                    {getStatusBadge(session.status)}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </motion.div>
  );
}

// ── Main page ────────────────────────────────────────────────────────────────

export default function StudentAttendance() {
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('All');
  const [filterOpen, setFilterOpen] = useState(false);
  const [selectedCourse, setSelectedCourse] = useState<StudentAttendanceCourse | null>(null);
  const [excuseModalOpen, setExcuseModalOpen] = useState(false);
  const filterRef = useRef<HTMLDivElement>(null);
  const queryClient = useQueryClient();

  const closeFilter = useCallback(() => setFilterOpen(false), []);
  useClickOutside(filterRef, closeFilter);

  const { data: attendanceData = [], isLoading, isError, refetch } = useQuery<StudentAttendanceCourse[]>({
    queryKey: ['studentAttendance'],
    queryFn: () => dashboardService.studentAttendanceData(),
    staleTime: 1000 * 60 * 2,
  });

  const filteredData = attendanceData.filter((course) => {
    const matchesSearch =
      course.course_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      course.course_code.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesStatus = statusFilter === 'All' || course.status === statusFilter;
    return matchesSearch && matchesStatus;
  });

  return (
    <>
      {/* Backdrop for slide panel */}
      <AnimatePresence>
        {selectedCourse && (
          <motion.div
            key="backdrop"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="fixed inset-0 bg-black/30 backdrop-blur-[2px] z-40"
            onClick={() => setSelectedCourse(null)}
          />
        )}
      </AnimatePresence>

      {/* Session drill-down panel */}
      <AnimatePresence>
        {selectedCourse && (
          <SessionPanel course={selectedCourse} onClose={() => setSelectedCourse(null)} />
        )}
      </AnimatePresence>

      {/* Request Excuse modal */}
      <AnimatePresence>
        {excuseModalOpen && (
          <ExcuseModal
            courses={attendanceData}
            onClose={() => setExcuseModalOpen(false)}
            onSuccess={() => queryClient.invalidateQueries({ queryKey: ['myExcuseRequests'] })}
          />
        )}
      </AnimatePresence>

      <div className="space-y-6">
        {isError && (
          <div className="flex items-center justify-between rounded-xl border border-rose-200 dark:border-rose-500/20 bg-rose-50 dark:bg-rose-500/10 px-4 py-3 text-sm text-rose-700 dark:text-rose-300">
            <span>Failed to load attendance records. Check your connection.</span>
            <button
              onClick={() => void refetch()}
              className="ml-4 shrink-0 text-xs font-semibold underline hover:no-underline"
            >
              Retry
            </button>
          </div>
        )}

        <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white tracking-tight">Attendance Record</h1>
            <p className="text-gray-500 dark:text-gray-400 mt-1">Click any course row to see individual session history.</p>
          </div>

          <div className="flex items-center gap-3">
            {/* Request Excuse button */}
            <Button
              variant="secondary"
              className="shrink-0 gap-2 border-amber-300 dark:border-amber-500/40 text-amber-700 dark:text-amber-400 hover:bg-amber-50 dark:hover:bg-amber-500/10"
              onClick={() => setExcuseModalOpen(true)}
            >
              <FilePen size={16} />
              Request Excuse
            </Button>

            <div className="relative w-full sm:w-64">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" size={18} />
              <Input
                placeholder="Search courses..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10 glass-input"
              />
            </div>

            {/* Filter dropdown */}
            <div className="relative shrink-0" ref={filterRef}>
              <Button
                variant="secondary"
                className={cn('gap-2', statusFilter !== 'All' && 'border-primary/60 text-primary dark:text-primary-accent')}
                onClick={() => setFilterOpen((v) => !v)}
              >
                <Filter size={16} />
                {statusFilter === 'All' ? 'Filter' : statusFilter}
                <ChevronDown size={14} className={cn('transition-transform', filterOpen && 'rotate-180')} />
              </Button>

              <AnimatePresence>
                {filterOpen && (
                  <motion.div
                    initial={{ opacity: 0, y: -6, scale: 0.97 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: -6, scale: 0.97 }}
                    transition={{ duration: 0.15 }}
                    className="absolute right-0 mt-2 w-44 rounded-xl border border-gray-200 dark:border-white/10 bg-white dark:bg-dark-bg shadow-lg z-20 overflow-hidden"
                  >
                    <div className="p-1">
                      <p className="px-3 py-1.5 text-[11px] font-semibold uppercase tracking-wider text-gray-400 dark:text-gray-500">
                        Attendance Status
                      </p>
                      {STATUS_OPTIONS.map((opt) => (
                        <button
                          key={opt}
                          onClick={() => { setStatusFilter(opt); setFilterOpen(false); }}
                          className={cn(
                            'w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm transition-colors',
                            statusFilter === opt
                              ? 'bg-primary/10 text-primary dark:text-primary-accent font-medium'
                              : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-white/5',
                          )}
                        >
                          <span className={cn('w-2 h-2 rounded-full shrink-0', STATUS_COLORS[opt])} />
                          {opt === 'All' ? 'All Statuses' : opt}
                          {statusFilter === opt && <Check size={14} className="ml-auto" />}
                        </button>
                      ))}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            <Button
              variant="secondary"
              className="shrink-0 hidden sm:flex gap-2"
              onClick={() => exportToCSV(filteredData)}
              disabled={filteredData.length === 0}
            >
              <Download size={16} />
              Export
            </Button>
          </div>
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <Card className="glass-card border-gray-200 dark:border-white/5 overflow-hidden">
            <CardContent className="p-0">
              <div className="overflow-x-auto custom-scrollbar w-full">
                <table className="w-full min-w-[800px] text-sm text-left whitespace-nowrap">
                  <thead className="bg-gray-50/80 dark:bg-white/5 text-gray-500 dark:text-gray-400 font-medium border-b border-gray-100 dark:border-white/5">
                    <tr>
                      <th className="px-6 py-4">Course</th>
                      <th className="px-6 py-4">Code</th>
                      <th className="px-6 py-4 text-center">Classes Attended</th>
                      <th className="px-6 py-4 min-w-[200px]">Attendance Progress</th>
                      <th className="px-6 py-4">Last Updated</th>
                      <th className="px-6 py-4 text-center">Status</th>
                      <th className="px-6 py-4 w-10" />
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-100 dark:divide-white/5">
                    {isLoading ? (
                      Array.from({ length: 5 }).map((_, i) => (
                        <tr key={i} className="animate-pulse">
                          <td className="px-6 py-4"><div className="h-3.5 w-36 rounded bg-gray-100 dark:bg-white/5" /></td>
                          <td className="px-6 py-4"><div className="h-5 w-16 rounded-md bg-gray-100 dark:bg-white/5" /></td>
                          <td className="px-6 py-4 text-center"><div className="h-3.5 w-16 rounded bg-gray-100 dark:bg-white/5 mx-auto" /></td>
                          <td className="px-6 py-4">
                            <div className="space-y-2">
                              <div className="flex justify-between">
                                <div className="h-2.5 w-16 rounded bg-gray-100 dark:bg-white/5" />
                                <div className="h-2.5 w-8 rounded bg-gray-100 dark:bg-white/5" />
                              </div>
                              <div className="h-2 w-full rounded-full bg-gray-100 dark:bg-white/5" />
                            </div>
                          </td>
                          <td className="px-6 py-4"><div className="h-3.5 w-20 rounded bg-gray-100 dark:bg-white/5" /></td>
                          <td className="px-6 py-4 text-center"><div className="h-5 w-16 rounded-full bg-gray-100 dark:bg-white/5 mx-auto" /></td>
                          <td className="px-6 py-4" />
                        </tr>
                      ))
                    ) : filteredData.length === 0 ? (
                      <tr>
                        <td colSpan={7} className="px-6 py-14">
                          {statusFilter !== 'All' ? (
                            <div className="flex flex-col items-center gap-3 text-center text-gray-500 dark:text-gray-400">
                              <Filter size={36} className="opacity-20" />
                              <div>
                                <p className="font-medium text-gray-700 dark:text-gray-300">No "{statusFilter}" courses</p>
                                <p className="text-sm mt-0.5">None of your courses currently have this attendance status.</p>
                              </div>
                              <button onClick={() => setStatusFilter('All')} className="text-sm text-primary dark:text-primary-accent underline hover:no-underline">Clear filter</button>
                            </div>
                          ) : attendanceData.length === 0 ? (
                            <div className="flex flex-col items-center gap-4 text-center max-w-sm mx-auto">
                              <div className="w-14 h-14 rounded-full bg-blue-50 dark:bg-blue-500/10 flex items-center justify-center">
                                <BookOpenCheck size={28} className="text-blue-500 dark:text-blue-400" />
                              </div>
                              <div>
                                <p className="font-semibold text-gray-800 dark:text-gray-200 text-base">No attendance records yet</p>
                                <p className="text-sm text-gray-500 dark:text-gray-400 mt-1.5 leading-relaxed">
                                  Your attendance will appear here once your teacher starts a session and you have been scanned in.
                                </p>
                              </div>
                              <div className="flex items-start gap-2 rounded-lg bg-blue-50 dark:bg-blue-500/10 border border-blue-100 dark:border-blue-500/20 px-4 py-3 text-left text-xs text-blue-700 dark:text-blue-300">
                                <Info size={14} className="mt-0.5 shrink-0" />
                                <span>If you are already enrolled but see nothing here, make sure your face images have been submitted for recognition.</span>
                              </div>
                            </div>
                          ) : null}
                        </td>
                      </tr>
                    ) : (
                      filteredData.map((course) => (
                        <tr
                          key={course.id}
                          onClick={() => setSelectedCourse(course)}
                          className={cn(
                            'hover:bg-gray-50/50 dark:hover:bg-white/5 transition-colors cursor-pointer group',
                            selectedCourse?.id === course.id && 'bg-primary/5 dark:bg-primary/10',
                          )}
                        >
                          <td className="px-6 py-4">
                            <p className="font-semibold text-gray-900 dark:text-white">{course.course_name}</p>
                          </td>
                          <td className="px-6 py-4">
                            <p className="text-xs font-medium text-primary dark:text-primary-accent bg-primary/5 dark:bg-primary/10 px-2.5 py-1 rounded-md inline-block border border-primary/10 dark:border-primary/20">
                              {course.course_code}
                            </p>
                          </td>
                          <td className="px-6 py-4 text-center">
                            <span className="text-lg font-bold text-gray-900 dark:text-white">{course.classes_attended}</span>
                            <span className="text-gray-400 dark:text-gray-500 mx-1">/</span>
                            <span className="text-gray-500 dark:text-gray-400 font-medium">{course.total_classes}</span>
                          </td>
                          <td className="px-6 py-4">
                            <div className="flex flex-col gap-2">
                              <div className="flex justify-between items-center text-xs">
                                <span className="text-gray-500 dark:text-gray-400">Attendance</span>
                                <span className="font-bold text-gray-900 dark:text-white">{Math.round(course.attendance_percentage ?? 0)}%</span>
                              </div>
                              <div className="w-full bg-gray-200 dark:bg-gray-800 rounded-full h-2 overflow-hidden shadow-inner">
                                <motion.div
                                  initial={{ width: 0 }}
                                  animate={{ width: `${Math.round(course.attendance_percentage ?? 0)}%` }}
                                  transition={{ duration: 1, ease: 'easeOut' }}
                                  className={cn('h-full rounded-full', getProgressColor(Math.round(course.attendance_percentage ?? 0)))}
                                />
                              </div>
                            </div>
                          </td>
                          <td className="px-6 py-4">
                            <span className="text-gray-600 dark:text-gray-300">
                              {course.last_updated ? new Date(course.last_updated).toLocaleDateString() : '-'}
                            </span>
                          </td>
                          <td className="px-6 py-4 text-center">
                            {getStatusBadge(course.status)}
                          </td>
                          <td className="px-6 py-4 text-right">
                            <ChevronRight size={16} className="text-gray-400 group-hover:text-primary dark:group-hover:text-primary-accent transition-colors ml-auto" />
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>

              <div className="border-t border-gray-100 dark:border-white/5 px-6 py-4 flex items-center justify-between bg-gray-50/30 dark:bg-white/[0.02]">
                <span className="text-sm text-gray-500 dark:text-gray-400">
                  Showing{' '}
                  <span className="font-medium text-gray-900 dark:text-white">{filteredData.length}</span>
                  {' '}of{' '}
                  <span className="font-medium text-gray-900 dark:text-white">{attendanceData.length}</span>
                  {' '}courses
                  {statusFilter !== 'All' && (
                    <button onClick={() => setStatusFilter('All')} className="ml-2 text-primary dark:text-primary-accent hover:underline text-xs">
                      Clear filter
                    </button>
                  )}
                </span>
                <Button variant="secondary" size="sm" className="sm:hidden gap-2" onClick={() => exportToCSV(filteredData)} disabled={filteredData.length === 0}>
                  <Download size={14} />
                  Export
                </Button>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* My excuse requests history */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
          <Card className="glass-card border-gray-200 dark:border-white/5">
            <CardHeader className="border-b border-gray-100 dark:border-white/5 pb-4">
              <div className="flex items-center justify-between">
                <CardTitle className="text-base font-bold text-gray-900 dark:text-white flex items-center gap-2">
                  <FilePen size={16} className="text-amber-500" />
                  My Excuse Requests
                </CardTitle>
                <button
                  onClick={() => setExcuseModalOpen(true)}
                  className="text-xs font-semibold text-primary dark:text-primary-accent hover:underline"
                >
                  + New Request
                </button>
              </div>
            </CardHeader>
            <CardContent className="p-4">
              <MyExcuseRequests />
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </>
  );
}
