import { useEffect, useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { motion, AnimatePresence } from 'framer-motion';
import { Play, X, AlertCircle } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { useAttendanceStore } from '@/store/useAttendanceStore';
import { useAcademiaStore } from '@/store/useAcademiaStore';
import { useAuthStore } from '@/store/useAuthStore';
import ScannerInterface from '@/components/attendance/ScannerInterface';
import { cn } from '@/utils/cn';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import * as z from 'zod';
import attendanceService from '@/services/attendanceService';

const adminAttendanceSchema = z.object({
  faculty_id: z.string().min(1, 'Faculty is required'),
  department_id: z.string().min(1, 'Department is required'),
  course_id: z.string().min(1, 'Course is required'),
  schedule_id: z.string().min(1, 'Schedule is required'),
  class_id: z.string().min(1, 'Class is required'),
  session_type: z.enum(['Lecture', 'Lab', 'Tutorial'], {
    message: 'Session type is required',
  }),
  camera_index: z.coerce.number().int().min(0, 'Camera index must be 0 or greater'),
  notes: z.string().optional(),
});

type AdminAttendanceForm = z.infer<typeof adminAttendanceSchema>;

export default function Attendance() {
  const { sessionState, startSession, setActiveSession, resetSession } = useAttendanceStore();
  const { faculties, departments, courses, classes, classAssignments, courseAssignments, structures, isLoading: academiaLoading, error: academiaError, fetchData } = useAcademiaStore();
  const { user } = useAuthStore();
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Fetch all active sessions visible to this admin
  const { data: activeSessions } = useQuery({
    queryKey: ['adminActiveSessions'],
    queryFn: () => attendanceService.listActiveSessions(),
    refetchInterval: 15_000,
  });

  // Only the session THIS admin personally started (admin_id matches their user id)
  const myActiveSession = useMemo(() => {
    const sessions = activeSessions?.items ?? activeSessions ?? [];
    return (
      (sessions as any[]).find(
        (s: any) =>
          String(s.status ?? '').toUpperCase() === 'ACTIVE' &&
          s.admin_id === user?.id,
      ) ?? null
    );
  }, [activeSessions, user?.id]);

  const handleResumeSession = (session: any) => {
    const foundCourse = courses.find((c) => String(c.id) === String(session.course_id));
    const courseName = foundCourse?.title ?? foundCourse?.name ?? `Course ${session.course_id}`;
    setActiveSession(Number(session.id), courseName);
    startSession({ sessionId: Number(session.id), courseName });
  };

  // Reset session when navigating to this page
  useEffect(() => {
    resetSession();
  }, [resetSession]);

  useEffect(() => {
    if (!academiaLoading && faculties.length === 0) {
      fetchData();
    }
  }, [fetchData, faculties.length, academiaLoading]);

  const {
    register,
    handleSubmit,
    watch,
    setValue,
    reset,
    formState: { errors },
  } = useForm<AdminAttendanceForm>({
    resolver: zodResolver(adminAttendanceSchema),
    defaultValues: {
      faculty_id: '',
      department_id: '',
      course_id: '',
      schedule_id: '',
      class_id: '',
      session_type: '' as any,
      camera_index: 0,
      notes: '',
    },
  });

  const selectedFacultyId = watch('faculty_id');
  const selectedDepartmentId = watch('department_id');
  const selectedCourseId = watch('course_id');
  const selectedScheduleId = String(watch('schedule_id') ?? '');

  // Course IDs that belong to a currently-active semester
  const activeSemesterCourseIds = useMemo(() => {
    const activeStructureIds = new Set(
      structures.filter((s) => s.status === 'Active').map((s) => s.id),
    );
    return new Set(
      courseAssignments
        .filter((ca) => activeStructureIds.has(ca.academicYearId))
        .map((ca) => ca.courseId),
    );
  }, [structures, courseAssignments]);

  const filteredDepartments = departments.filter((d) => d.facultyId === selectedFacultyId);
  // Only show courses whose semester is currently active
  const filteredCourses = courses.filter(
    (c) => c.departmentId === selectedDepartmentId && activeSemesterCourseIds.has(c.id),
  );
  const filteredClasses = classes.filter((cls) => cls.departmentId === selectedDepartmentId || cls.facultyId === selectedFacultyId);

  // When a course is selected, restrict class dropdown to only classes assigned to that course.
  // This ensures the auto-filled class always appears as a valid option in the dropdown.
  const classesForSelectedCourse = (() => {
    if (!selectedCourseId) return filteredClasses;
    const assignedClassIds = classAssignments
      .filter((a) => a.courseId === selectedCourseId)
      .map((a) => a.classId);
    if (assignedClassIds.length === 0) return filteredClasses;
    return classes.filter((cls) => assignedClassIds.includes(cls.id));
  })();

  // Active session for the currently selected course (any creator)
  const { data: courseActiveSessions } = useQuery({
    queryKey: ['adminCourseActiveSession', selectedCourseId],
    queryFn: () => attendanceService.listActiveSessions({ course_id: Number(selectedCourseId) }),
    enabled: !!selectedCourseId,
    refetchInterval: 10_000,
  });

  const courseActiveSession = useMemo(() => {
    const sessions = courseActiveSessions?.items ?? courseActiveSessions ?? [];
    return Array.isArray(sessions)
      ? (sessions as any[]).find((s: any) => String(s.status ?? '').toUpperCase() === 'ACTIVE') ?? null
      : null;
  }, [courseActiveSessions]);

  // Is the selected course's active session one that I (this admin) started?
  const isMyActiveSessionForCourse = useMemo(() => {
    if (!courseActiveSession) return false;
    return courseActiveSession.admin_id === user?.id;
  }, [courseActiveSession, user?.id]);

  // Who started the selected course's active session — "teacher" | "admin" | null
  const courseSessionStarterRole = useMemo((): 'teacher' | 'admin' | null => {
    if (!courseActiveSession) return null;
    if (courseActiveSession.teacher_id != null) return 'teacher';
    if (courseActiveSession.admin_id != null) return 'admin';
    return null;
  }, [courseActiveSession]);

  const { data: courseSchedules, isLoading: schedulesLoading } = useQuery({
    queryKey: ['adminCourseSchedules', selectedCourseId],
    queryFn: () => attendanceService.getSchedulesForCourse(Number(selectedCourseId)),
    enabled: !!selectedCourseId,
    staleTime: 1000 * 60 * 5,
  });

  // Stable reference — prevents auto-select useEffect from firing on every render
  const scheduleOptions = useMemo(() => courseSchedules ?? [], [courseSchedules]);

  // Day-code → JS Date.getDay() map (EAT timezone)
  const DAY_CODE_TO_JS: Record<string, number> = {
    sat: 6, sun: 0, mon: 1, tue: 2, wed: 3, thu: 4, fri: 5,
  };
  const DAY_NAMES_FULL: Record<string, string> = {
    sat: 'Saturday', sun: 'Sunday', mon: 'Monday',
    tue: 'Tuesday', wed: 'Wednesday', thu: 'Thursday', fri: 'Friday',
  };

  // Auto-select schedule when course schedules load
  useEffect(() => {
    if (!selectedCourseId || schedulesLoading) return;
    const options = scheduleOptions as any[];
    if (options.length === 0) return;

    const nowEAT = new Date(new Date().toLocaleString('en-US', { timeZone: 'Africa/Mogadishu' }));
    const todayJs = nowEAT.getDay();

    const todayMatch = options.find((s: any) => {
      const raw: string[] = s.weekday_raw ?? [];
      return raw.some((code) => (DAY_CODE_TO_JS[code] ?? -1) === todayJs);
    });

    const best = todayMatch ?? options[0];
    setValue('schedule_id', String(best.id), { shouldValidate: true });
    setSubmitError(null);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scheduleOptions, selectedCourseId, schedulesLoading]);

  // Find selected schedule object for validation
  const selectedSchedule = selectedScheduleId
    ? (scheduleOptions as any[]).find((s: any) => String(s.id) === selectedScheduleId) ?? null
    : null;

  // Real-time day/time validation in Africa/Mogadishu (EAT, UTC+3)
  const scheduleTimeError = (() => {
    if (!selectedSchedule) return null;
    const nowEAT = new Date(new Date().toLocaleString('en-US', { timeZone: 'Africa/Mogadishu' }));
    const todayJs = nowEAT.getDay();

    const weekdayRaw: string[] = selectedSchedule.weekday_raw ?? [];
    if (weekdayRaw.length > 0) {
      const scheduledJsDays = weekdayRaw.map((c: string) => DAY_CODE_TO_JS[c] ?? -1).filter((d: number) => d >= 0);
      if (scheduledJsDays.length > 0 && !scheduledJsDays.includes(todayJs)) {
        const dayNames = weekdayRaw.map((c: string) => DAY_NAMES_FULL[c] ?? c.toUpperCase()).join(', ');
        return `This course is not scheduled for today. It runs on: ${dayNames}.`;
      }
    }

    const [sh = 0, sm = 0] = selectedSchedule.start_time.split(':').map(Number);
    const [eh = 0, em = 0] = selectedSchedule.end_time.split(':').map(Number);
    const nowMinutes = nowEAT.getHours() * 60 + nowEAT.getMinutes();

    if (nowMinutes < sh * 60 + sm) {
      return `This session cannot be started yet. The scheduled time begins at ${String(sh).padStart(2,'0')}:${String(sm).padStart(2,'0')}.`;
    }
    if (nowMinutes > eh * 60 + em) {
      return `The scheduled time for this course has passed. Sessions for this slot close at ${String(eh).padStart(2,'0')}:${String(em).padStart(2,'0')}.`;
    }
    return null;
  })();

  const handleCourseChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const courseId = e.target.value;
    setValue('course_id', courseId, { shouldValidate: true });
    setValue('schedule_id', '', { shouldValidate: false });
    setSubmitError(null);

    // Auto-fill Class/Section from class-course assignment table
    const assignment = classAssignments.find((a) => a.courseId === courseId);
    setValue('class_id', assignment ? assignment.classId : '', { shouldValidate: true });
  };

  const handleFacultyChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setValue('faculty_id', e.target.value, { shouldValidate: true });
    setValue('department_id', '', { shouldValidate: true });
    setValue('course_id', '', { shouldValidate: true });
    setValue('class_id', '', { shouldValidate: true });
  };

  const handleDepartmentChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setValue('department_id', e.target.value, { shouldValidate: true });
    setValue('course_id', '', { shouldValidate: true });
    setValue('class_id', '', { shouldValidate: true });
  };

  const onFormSubmit = async (data: AdminAttendanceForm) => {
    setSubmitError(null);
    setIsSubmitting(true);
    try {
      const selectedCourse = courses.find(c => String(c.id) === String(data.course_id));
      const courseName = selectedCourse?.title ?? selectedCourse?.name ?? `Course #${data.course_id}`;
      const response = await attendanceService.startSession({
        course_id: Number(data.course_id),
        session_type: data.session_type,
        schedule_id: data.schedule_id ? Number(data.schedule_id) : null,
      });
      const session = response?.session ?? response;
      const sessionId = session?.id ?? response?.id;
      if (!sessionId) throw new Error('No session ID returned from server');
      setActiveSession(Number(sessionId), courseName);
      startSession({ sessionId: Number(sessionId), courseName });
    } catch (err: any) {
      // Backend wraps errors as { error: { message: "..." } }; fall back to { detail: "..." }
      const d = err?.response?.data;
      const msg: string =
        (typeof d?.error?.message === 'string' ? d.error.message : null) ??
        (typeof d?.detail === 'string' ? d.detail : null) ??
        err?.message ??
        'Failed to start session. Please try again.';
      setSubmitError(msg);
    } finally {
      setIsSubmitting(false);
    }
  };

  const isScanningActive = sessionState === 'waiting_for_face';

  return (
    <div className="relative flex-1 flex flex-col items-center justify-center min-h-[calc(100vh-4rem)] px-4 py-2 overflow-hidden">
      
      {/* Dynamic Background Glow based on state */}
      <div 
        className={cn(
          "absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] rounded-full blur-[120px] opacity-20 pointer-events-none transition-colors duration-1000",
          sessionState === 'idle' && "bg-transparent",
          isScanningActive && "bg-primary",
          sessionState === 'success' && "bg-emerald-500",
          sessionState === 'failed' && "bg-rose-500",
          sessionState === 'already_marked' && "bg-yellow-400",
          sessionState === 'partial_face' && "bg-orange-500",
          sessionState === 'low_light' && "bg-amber-400"
        )} 
      />

      <AnimatePresence mode="wait">
        {sessionState === 'idle' ? (
          <motion.div
            key="start-screen"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9, filter: "blur(10px)" }}
            transition={{ duration: 0.4 }}
            className="w-full max-w-2xl z-10"
          >
            <Card className="glass-card shadow-2xl shadow-primary/10 border-white/5">
              <CardContent className="p-8">
                <div className="mb-6 space-y-2 text-center">
                  <h2 className="text-2xl font-bold text-gray-900 dark:text-white tracking-tight">
                    Start Attendance Session
                  </h2>
                  <p className="text-gray-500 dark:text-gray-400">
                    Configure the session details before starting the face recognition scanner.
                  </p>
                </div>

                {/* Amber: resume MY own active session (I started it as admin) */}
                {myActiveSession && (
                  <div className="mb-4 rounded-2xl border border-amber-200 dark:border-amber-500/20 bg-amber-50 dark:bg-amber-500/10 p-4 text-sm text-amber-800 dark:text-amber-200 space-y-3">
                    <div className="font-semibold">You have an active session running</div>
                    <div className="text-amber-700 dark:text-amber-300">
                      {courses.find((c) => String(c.id) === String(myActiveSession.course_id))?.title ??
                       courses.find((c) => String(c.id) === String(myActiveSession.course_id))?.name ??
                       `Course ${myActiveSession.course_id}`}
                      {' · '}Session #{myActiveSession.id}
                    </div>
                    <Button
                      type="button"
                      size="sm"
                      onClick={() => handleResumeSession(myActiveSession)}
                      className="bg-amber-500 hover:bg-amber-600 text-white border-0 flex items-center gap-2 mt-1"
                    >
                      <Play size={14} className="fill-white" />
                      Resume Session
                    </Button>
                  </div>
                )}

                {/* Rose: selected course has an active session started by someone ELSE */}
                {courseActiveSession && !isMyActiveSessionForCourse && selectedCourseId && (
                  <div className="mb-4 rounded-2xl border border-rose-200 dark:border-rose-500/20 bg-rose-50 dark:bg-rose-500/10 p-4 text-sm text-rose-800 dark:text-rose-200 space-y-1">
                    <div className="font-semibold">A session is already running for this course</div>
                    <div className="text-rose-600 dark:text-rose-300">
                      {courseSessionStarterRole === 'teacher'
                        ? 'A teacher has already started an attendance session for this course. You cannot start a duplicate session.'
                        : 'Another admin has already started an attendance session for this course. You cannot start a duplicate session.'}
                    </div>
                  </div>
                )}

                {submitError && (
                  <div className="mb-4 rounded-2xl border border-rose-300 dark:border-rose-500/30 bg-rose-50 dark:bg-rose-500/10 p-4 text-sm text-rose-800 dark:text-rose-200 space-y-1">
                    <div className="font-semibold flex items-center gap-2"><span>⛔</span><span>Unable to start session</span></div>
                    <div className="text-rose-600 dark:text-rose-300">{submitError}</div>
                  </div>
                )}

                {scheduleTimeError && (
                  <div className="mb-4 rounded-2xl border border-rose-300 dark:border-rose-500/30 bg-rose-50 dark:bg-rose-500/10 p-4 text-sm text-rose-800 dark:text-rose-200 space-y-1">
                    <div className="font-semibold flex items-center gap-2"><span>⛔</span><span>Session not allowed at this time</span></div>
                    <div className="text-rose-600 dark:text-rose-300">{scheduleTimeError}</div>
                  </div>
                )}

                <form onSubmit={handleSubmit(onFormSubmit)} className="space-y-6 text-left">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {/* Faculty */}
                    <div className="space-y-2">
                      <label className="text-sm font-medium text-gray-700 dark:text-gray-300 ml-1">
                        Faculty <span className="text-red-500">*</span>
                      </label>
                      <select
                        {...register('faculty_id')}
                        onChange={handleFacultyChange}
                        className={`flex h-12 w-full rounded-xl glass-input px-4 py-2 text-sm text-gray-900 dark:text-gray-100 transition-all appearance-none bg-no-repeat bg-[right_1rem_center] bg-[length:1em_1em] ${errors.faculty_id ? 'border-red-500 focus:border-red-500 focus:ring-red-500' : ''}`}
                        style={{ backgroundImage: `url("data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%2224%22%20height%3D%2224%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%236B7280%22%20stroke-width%3D%222%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpolyline%20points%3D%226%209%2012%2015%2018%209%22%3E%3C%2Fpolyline%3E%3C%2Fsvg%3E")`}}
                      >
                        <option value="" disabled className="text-gray-500">Select Faculty...</option>
                        {faculties.map(f => (
                          <option key={f.id} value={f.id} className="bg-white dark:bg-dark-bg text-gray-900 dark:text-white">{f.name}</option>
                        ))}
                      </select>
                      {errors.faculty_id && <p className="text-xs text-red-500 ml-1 mt-1">{errors.faculty_id.message}</p>}
                    </div>

                    {/* Department */}
                    <div className="space-y-2">
                      <label className="text-sm font-medium text-gray-700 dark:text-gray-300 ml-1">
                        Department <span className="text-red-500">*</span>
                      </label>
                      <select
                        {...register('department_id')}
                        onChange={handleDepartmentChange}
                        disabled={!selectedFacultyId}
                        className={`flex h-12 w-full rounded-xl glass-input px-4 py-2 text-sm text-gray-900 dark:text-gray-100 transition-all appearance-none bg-no-repeat bg-[right_1rem_center] bg-[length:1em_1em] disabled:opacity-50 disabled:cursor-not-allowed ${errors.department_id ? 'border-red-500 focus:border-red-500 focus:ring-red-500' : ''}`}
                        style={{ backgroundImage: `url("data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%2224%22%20height%3D%2224%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%236B7280%22%20stroke-width%3D%222%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpolyline%20points%3D%226%209%2012%2015%2018%209%22%3E%3C%2Fpolyline%3E%3C%2Fsvg%3E")`}}
                      >
                        <option value="" disabled className="text-gray-500">Select Department...</option>
                        {filteredDepartments.map(d => (
                          <option key={d.id} value={d.id} className="bg-white dark:bg-dark-bg text-gray-900 dark:text-white">{d.name}</option>
                        ))}
                      </select>
                      {errors.department_id && <p className="text-xs text-red-500 ml-1 mt-1">{errors.department_id.message}</p>}
                    </div>

                    {/* Course Name */}
                    <div className="space-y-2 md:col-span-2">
                      <label className="text-sm font-medium text-gray-700 dark:text-gray-300 ml-1">
                        Course Name <span className="text-red-500">*</span>
                      </label>
                      <select
                        {...register('course_id')}
                        onChange={handleCourseChange}
                        disabled={!selectedDepartmentId}
                        className={`flex h-12 w-full rounded-xl glass-input px-4 py-2 text-sm text-gray-900 dark:text-gray-100 transition-all appearance-none bg-no-repeat bg-[right_1rem_center] bg-[length:1em_1em] disabled:opacity-50 disabled:cursor-not-allowed ${errors.course_id ? 'border-red-500 focus:border-red-500 focus:ring-red-500' : ''}`}
                        style={{ backgroundImage: `url("data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%2224%22%20height%3D%2224%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%236B7280%22%20stroke-width%3D%222%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpolyline%20points%3D%226%209%2012%2015%2018%209%22%3E%3C%2Fpolyline%3E%3C%2Fsvg%3E")`}}
                      >
                        <option value="" disabled className="text-gray-500">Select a course...</option>
                        {filteredCourses.map((course) => (
                          <option key={course.id} value={course.id} className="bg-white dark:bg-dark-bg text-gray-900 dark:text-white">
                            {course.title || course.name}
                          </option>
                        ))}
                      </select>
                      {errors.course_id && (
                        <p className="text-xs text-red-500 ml-1 mt-1">{errors.course_id.message}</p>
                      )}
                    </div>

                    {/* Schedule / Time Slot */}
                    <div className="space-y-2 md:col-span-2">
                      <label className="text-sm font-medium text-gray-700 dark:text-gray-300 ml-1">
                        Schedule / Time Slot <span className="text-red-500">*</span>
                      </label>
                      <select
                        {...register('schedule_id')}
                        value={selectedScheduleId}
                        onChange={(e) => { register('schedule_id').onChange(e); setSubmitError(null); }}
                        disabled={!selectedCourseId || schedulesLoading}
                        className={`flex h-12 w-full rounded-xl glass-input px-4 py-2 text-sm text-gray-900 dark:text-gray-100 transition-all appearance-none bg-no-repeat bg-[right_1rem_center] bg-[length:1em_1em] disabled:opacity-50 disabled:cursor-not-allowed ${errors.schedule_id ? 'border-red-500 focus:border-red-500 focus:ring-red-500' : ''}`}
                        style={{ backgroundImage: `url("data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%2224%22%20height%3D%2224%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%236B7280%22%20stroke-width%3D%222%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpolyline%20points%3D%226%209%2012%2015%2018%209%22%3E%3C%2Fpolyline%3E%3C%2Fsvg%3E")`}}
                      >
                        <option value="" disabled className="text-gray-500">
                          {!selectedCourseId ? 'Select a course first' : schedulesLoading ? 'Loading schedules...' : scheduleOptions.length === 0 ? 'No schedules found' : 'Auto-filled — change if needed'}
                        </option>
                        {scheduleOptions.map((s: any) => (
                          <option key={s.id} value={String(s.id)} className="bg-white dark:bg-dark-bg text-gray-900 dark:text-white">
                            {s.weekday.toUpperCase()} · {s.start_time} – {s.end_time} ({s.grace_period_minutes} min grace)
                          </option>
                        ))}
                      </select>
                      {errors.schedule_id && <p className="text-xs text-red-500 ml-1 mt-1">{errors.schedule_id.message}</p>}
                    </div>

                    {/* Class / Section */}

                    <div className="space-y-2">
                      <label className="text-sm font-medium text-gray-700 dark:text-gray-300 ml-1">
                        Class / Section <span className="text-red-500">*</span>
                      </label>
                      <select
                        {...register('class_id')}
                        disabled={!selectedCourseId || academiaLoading}
                        className={`flex h-12 w-full rounded-xl glass-input px-4 py-2 text-sm text-gray-900 dark:text-gray-100 transition-all appearance-none bg-no-repeat bg-[right_1rem_center] bg-[length:1em_1em] disabled:opacity-50 disabled:cursor-not-allowed ${errors.class_id ? 'border-red-500 focus:border-red-500 focus:ring-red-500' : ''}`}
                        style={{ backgroundImage: 'url("data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%2224%22%20height%3D%2224%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%236B7280%22%20stroke-width%3D%222%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpolyline%20points%3D%226%209%2012%2015%2018%209%22%3E%3C%2Fpolyline%3E%3C%2Fsvg%3E")' }}
                      >
                        <option value="" disabled className="text-gray-500">
                          {!selectedCourseId ? 'Select a course first' : 'Select a class...'}
                        </option>
                        {classesForSelectedCourse.map((cls) => (
                          <option key={cls.id} value={cls.id} className="bg-white dark:bg-dark-bg text-gray-900 dark:text-white">
                            {cls.name}
                          </option>
                        ))}
                      </select>
                      {errors.class_id && <p className="text-xs text-red-500 ml-1 mt-1">{errors.class_id.message}</p>}
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {/* Session Type */}
                    <div className="space-y-2">
                      <label className="text-sm font-medium text-gray-700 dark:text-gray-300 ml-1">
                        Session Type <span className="text-red-500">*</span>
                      </label>
                      <select
                        {...register('session_type')}
                        className={`flex h-12 w-full rounded-xl glass-input px-4 py-2 text-sm text-gray-900 dark:text-gray-100 transition-all appearance-none bg-no-repeat bg-[right_1rem_center] bg-[length:1em_1em] ${errors.session_type ? 'border-red-500 focus:border-red-500 focus:ring-red-500' : ''}`}
                        style={{ backgroundImage: `url("data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%2224%22%20height%3D%2224%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%236B7280%22%20stroke-width%3D%222%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpolyline%20points%3D%226%209%2012%2015%2018%209%22%3E%3C%2Fpolyline%3E%3C%2Fsvg%3E")`}}
                      >
                        <option value="" disabled className="text-gray-500">Select Session Type...</option>
                        <option value="Lecture" className="bg-white dark:bg-dark-bg text-gray-900 dark:text-white">Lecture</option>
                        <option value="Lab" className="bg-white dark:bg-dark-bg text-gray-900 dark:text-white">Lab</option>
                        <option value="Tutorial" className="bg-white dark:bg-dark-bg text-gray-900 dark:text-white">Tutorial</option>
                      </select>
                      {errors.session_type && (
                        <p className="text-xs text-red-500 ml-1 mt-1">{errors.session_type.message}</p>
                      )}
                    </div>

                    {/* Camera Index */}
                    <div className="space-y-2">
                      <label className="text-sm font-medium text-gray-700 dark:text-gray-300 ml-1">
                        Camera Index <span className="text-red-500">*</span>
                      </label>
                      <Input
                        {...register('camera_index')}
                        type="number"
                        min="0"
                        placeholder="0"
                        className="glass-input"
                        error={errors.camera_index?.message}
                      />
                    </div>
                  </div>

                  {/* Notes */}
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-gray-700 dark:text-gray-300 ml-1">
                      Notes (Optional)
                    </label>
                    <textarea
                      {...register('notes')}
                      placeholder="Add any specific notes for this session..."
                      className="w-full rounded-xl glass-input px-4 py-3 text-sm text-gray-900 dark:text-gray-100 transition-all min-h-[100px] resize-y placeholder:text-gray-400 focus:border-primary focus:ring-primary dark:focus:border-primary-accent dark:focus:ring-primary-accent"
                    />
                  </div>

                  <div className="flex items-center justify-end gap-4 pt-4 border-t border-gray-100 dark:border-white/10">
                    <Button type="button" variant="secondary" onClick={() => { reset(); setSubmitError(null); }}>
                      <X size={18} className="mr-2" />
                      Cancel
                    </Button>
                    <Button
                      type="submit"
                      disabled={isSubmitting || !!scheduleTimeError || (!!courseActiveSession && !isMyActiveSessionForCourse)}
                      className="bg-gradient-brand hover:shadow-lg hover:shadow-primary/20 text-white border-0"
                    >
                      <Play size={18} className="mr-2 fill-white" />
                      {isSubmitting ? 'Starting...' : 'Start Session'}
                    </Button>
                  </div>
                </form>

              </CardContent>
            </Card>
          </motion.div>
        ) : (
          <ScannerInterface />
        )}
      </AnimatePresence>
    </div>
  );
}
