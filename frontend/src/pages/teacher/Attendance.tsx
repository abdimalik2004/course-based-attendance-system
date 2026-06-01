import { useEffect, useMemo, useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";
import { motion, AnimatePresence } from "framer-motion";
import { Play, X } from "lucide-react";
import { Card, CardContent } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { useAttendanceStore } from "@/store/useAttendanceStore";
import { useAcademiaStore } from "@/store/useAcademiaStore";
import ScannerInterface from "@/components/attendance/ScannerInterface";
import { cn } from "@/utils/cn";
import { useQuery } from "@tanstack/react-query";
import courseService from "@/services/courseService";
import attendanceService from "@/services/attendanceService";
import { useAuthStore } from '@/store/useAuthStore';

const attendanceSchema = z.object({
  course_id: z.string().min(1, "Course is required"),
  schedule_id: z.string().min(1, "Schedule is required"),
  class_id: z.string().optional(),
  session_type: z.enum(["Lecture", "Lab", "Tutorial"], {
    message: "Session type is required",
  }),
  camera_index: z.coerce
    .number()
    .int()
    .min(0, "Camera index must be 0 or greater"),
  notes: z.string().optional(),
});

type AttendanceForm = z.infer<typeof attendanceSchema>;

export default function TeacherAttendance() {
  const {
    sessionState,
    startSession,
    resetSession,
    setActiveSession,
  } = useAttendanceStore();
  const { structures, courseAssignments, fetchData: fetchAcademiaData } = useAcademiaStore();
  const [sessionError, setSessionError] = useState<string | null>(null);

  // Ensure academia store is populated (needed for semester filtering)
  useEffect(() => {
    if (structures.length === 0) fetchAcademiaData();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  const { data: activeSessions } = useQuery({
    queryKey: ["teacherActiveSessions"],
    queryFn: () => attendanceService.listActiveSessions(),
    refetchInterval: 15_000,
  });
  const { user } = useAuthStore();
  const teacherId = Number(user?.teacherId ?? user?.id ?? 0);

  const {
    data: coursesData,
    isLoading: coursesLoading,
    isError: coursesError,
    error: coursesErrorObj,
    refetch: refetchCourses,
  } = useQuery({
    queryKey: ["teacherAttendanceCourses", teacherId],
    queryFn: () =>
      courseService.listAssignments({ teacher_id: teacherId, skip: 0, limit: 200 }),
    enabled: !!teacherId,
    retry: false,
  });

  // Fetch the full course list (TEACHER role is scoped to their faculty by the backend)
  // so we can look up course titles from the assignment's course_id.
  // Note: GET /courses/{id} does not exist, so we fetch all courses in one call instead.
  const { data: allCoursesData } = useQuery({
    queryKey: ["teacherAllCourses"],
    queryFn: () => courseService.listCourses({ limit: 200 }),
    staleTime: 1000 * 60 * 5,
  });

  // Course IDs that have an active semester assignment
  const activeSemesterCourseIds = useMemo(() => {
    const activeIds = new Set(
      structures.filter((s) => s.status === 'Active').map((s) => s.id),
    );
    return new Set(
      courseAssignments
        .filter((ca) => activeIds.has(ca.academicYearId))
        .map((ca) => ca.courseId),
    );
  }, [structures, courseAssignments]);

  const teacherCourses = useMemo(() => {
    const assignments = coursesData?.items ?? coursesData ?? [];
    const allCourses: any[] = allCoursesData?.items ?? allCoursesData ?? [];

    return assignments
      .map((assignment: any) => {
        // assignment = { id, course_id, teacher_id, is_primary }
        const courseId = String(assignment.course_id ?? assignment.id);
        const courseDetail = allCourses.find((c: any) => String(c.id) === courseId);
        return {
          id: courseId,
          name: courseDetail?.title ?? courseDetail?.name ?? `Course ${courseId}`,
          class_name: courseDetail?.department_name ?? "",
        };
      })
      // Only show courses whose semester is currently active.
      // If the academia store hasn't loaded yet (activeSemesterCourseIds empty), show all.
      .filter((c: any) =>
        activeSemesterCourseIds.size === 0 || activeSemesterCourseIds.has(c.id),
      );
  }, [coursesData, allCoursesData, activeSemesterCourseIds]);

  // Only the session THIS teacher personally started (teacher_id matches their DB id)
  const myActiveSession = useMemo(() => {
    const sessions = activeSessions?.items ?? activeSessions ?? [];
    return (
      (sessions as any[]).find(
        (s: any) =>
          String(s.status ?? "").toUpperCase() === "ACTIVE" &&
          s.teacher_id === teacherId,
      ) ?? null
    );
  }, [activeSessions, teacherId]);

  const hasCourses = teacherCourses.length > 0;
  const courseQueryError = coursesError ? (coursesErrorObj as Error)?.message ?? 'Unable to load assigned courses.' : null;

  // Reset session when navigating to this page
  useEffect(() => {
    resetSession();
  }, [resetSession]);

  const {
    register,
    handleSubmit,
    setValue,
    reset,
    watch,
    formState: { errors, isSubmitting },
  } = useForm<AttendanceForm>({
    resolver: zodResolver(attendanceSchema),
    defaultValues: {
      course_id: "",
      schedule_id: "",
      class_id: "",
      session_type: "" as any,
      camera_index: 0,
      notes: "",
    },
  });

  const cameraIndex = Number(watch("camera_index") ?? 0);

  const selectedCourseId = String(watch("course_id") ?? "");
  const selectedScheduleId = String(watch("schedule_id") ?? "");

  // Active session for the currently selected course (any creator)
  const { data: courseActiveSessions } = useQuery({
    queryKey: ["courseActiveSession", selectedCourseId],
    queryFn: () =>
      attendanceService.listActiveSessions({ course_id: Number(selectedCourseId) }),
    enabled: !!selectedCourseId,
    refetchInterval: 10_000,
  });

  // The actual active session object for the selected course (if any)
  const courseActiveSession = useMemo(() => {
    const sessions = courseActiveSessions?.items ?? courseActiveSessions ?? [];
    return Array.isArray(sessions)
      ? (sessions as any[]).find((s: any) => String(s.status ?? "").toUpperCase() === "ACTIVE") ?? null
      : null;
  }, [courseActiveSessions]);

  // Is the selected course's active session mine (I started it)?
  const isMyActiveSessionForCourse = useMemo(() => {
    if (!courseActiveSession) return false;
    return courseActiveSession.teacher_id === teacherId;
  }, [courseActiveSession, teacherId]);

  // Who started the selected course's active session — for the blocked-course message
  const courseSessionStarterRole = useMemo((): "teacher" | "admin" | null => {
    if (!courseActiveSession) return null;
    if (courseActiveSession.teacher_id != null) return "teacher";
    if (courseActiveSession.admin_id != null) return "admin";
    return null;
  }, [courseActiveSession]);

  const handleResumeSession = (session: any) => {
    const allCourses: any[] = allCoursesData?.items ?? allCoursesData ?? [];
    const courseName =
      teacherCourses.find((c) => c.id === String(session.course_id))?.name ??
      allCourses.find((c: any) => String(c.id) === String(session.course_id))?.title ??
      `Course ${session.course_id}`;
    setActiveSession(Number(session.id), courseName);
    startSession({ sessionId: Number(session.id), courseName });
  };

  // Fetch schedules when a course is selected
  const { data: courseSchedules, isLoading: schedulesLoading } = useQuery({
    queryKey: ["courseSchedules", selectedCourseId],
    queryFn: () => attendanceService.getSchedulesForCourse(Number(selectedCourseId)),
    enabled: !!selectedCourseId,
    staleTime: 1000 * 60 * 5,
  });

  // Fetch classes linked to the selected course
  const { data: courseClasses } = useQuery({
    queryKey: ["courseClasses", selectedCourseId],
    queryFn: () => attendanceService.listClassesForCourse(Number(selectedCourseId)),
    enabled: !!selectedCourseId,
    staleTime: 1000 * 60 * 5,
  });
  // Stabilise the reference so the auto-select useEffect doesn't fire on every render
  const scheduleOptions = useMemo(() => courseSchedules ?? [], [courseSchedules]);

  // Map backend day codes (sat=1…fri=7, same as schedule_weekday_from_datetime)
  // to JavaScript Date.getDay() values (0=Sun,1=Mon,…,6=Sat).
  const DAY_CODE_TO_JS: Record<string, number> = {
    sat: 6, sun: 0, mon: 1, tue: 2, wed: 3, thu: 4, fri: 5,
  };
  const DAY_NAMES_FULL: Record<string, string> = {
    sat: "Saturday", sun: "Sunday", mon: "Monday",
    tue: "Tuesday", wed: "Wednesday", thu: "Thursday", fri: "Friday",
  };

  // Find the currently selected schedule object so we can validate timing.
  const selectedSchedule = selectedScheduleId
    ? (scheduleOptions as any[]).find((s: any) => String(s.id) === selectedScheduleId) ?? null
    : null;

  // ── Auto-fill class when course classes load ─────────────────────────────
  useEffect(() => {
    if (!selectedCourseId || !courseClasses) return;
    if (courseClasses.length === 0) {
      setValue("class_id", "", { shouldValidate: false });
      return;
    }
    // Join all class names — most courses are linked to one class, but show
    // all of them comma-separated when there are multiple.
    const label = courseClasses.map((c) => c.name).join(", ");
    setValue("class_id", label, { shouldValidate: false });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [courseClasses, selectedCourseId]);
  // ─────────────────────────────────────────────────────────────────────────

  // ── Auto-select schedule when course schedules load ──────────────────────
  // When a course is picked and its schedules arrive, automatically fill in
  // the schedule field. Priority: schedule whose weekday matches today (EAT);
  // fallback: the first schedule in the list.
  useEffect(() => {
    if (!selectedCourseId || schedulesLoading) return;
    const options = scheduleOptions as any[];
    if (options.length === 0) return;

    // Current day-of-week in Africa/Mogadishu (EAT = UTC+3)
    const nowEAT = new Date(
      new Date().toLocaleString("en-US", { timeZone: "Africa/Mogadishu" }),
    );
    const todayJs = nowEAT.getDay(); // 0=Sun … 6=Sat

    // Try to find a schedule whose weekday_raw includes today
    const todayMatch = options.find((s: any) => {
      const raw: string[] = s.weekday_raw ?? [];
      return raw.some((code) => (DAY_CODE_TO_JS[code] ?? -1) === todayJs);
    });

    const best = todayMatch ?? options[0];
    setValue("schedule_id", String(best.id), { shouldValidate: true });
    setSessionError(null);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scheduleOptions, selectedCourseId, schedulesLoading]);
  // ─────────────────────────────────────────────────────────────────────────

  // Compute a blocking error message if the selected schedule's day/time doesn't match now.
  // Uses Africa/Mogadishu (EAT, UTC+3) so it matches the backend's clock.
  // Re-evaluated on every render (no stale memoization) so it stays current.
  const scheduleTimeError = (() => {
    if (!selectedSchedule) return null;

    // Always use EAT (Africa/Mogadishu = UTC+3) to match the backend
    const nowEAT = new Date(
      new Date().toLocaleString("en-US", { timeZone: "Africa/Mogadishu" }),
    );
    const todayJs = nowEAT.getDay(); // 0=Sun … 6=Sat

    // 1. Day-of-week check
    const weekdayRaw: string[] = selectedSchedule.weekday_raw ?? [];
    if (weekdayRaw.length > 0) {
      const scheduledJsDays = weekdayRaw
        .map((code: string) => DAY_CODE_TO_JS[code] ?? -1)
        .filter((d: number) => d >= 0);
      if (scheduledJsDays.length > 0 && !scheduledJsDays.includes(todayJs)) {
        const dayNames = weekdayRaw
          .map((code: string) => DAY_NAMES_FULL[code] ?? code.toUpperCase())
          .join(", ");
        return `This course is not scheduled for today. It runs on: ${dayNames}.`;
      }
    }

    // 2. Time-window check (compare against EAT hours/minutes)
    const [sh = 0, sm = 0] = selectedSchedule.start_time.split(":").map(Number);
    const [eh = 0, em = 0] = selectedSchedule.end_time.split(":").map(Number);
    const startMinutes = sh * 60 + sm;
    const endMinutes = eh * 60 + em;
    const nowMinutes = nowEAT.getHours() * 60 + nowEAT.getMinutes();

    if (nowMinutes < startMinutes) {
      const startStr = `${String(sh).padStart(2, "0")}:${String(sm).padStart(2, "0")}`;
      return `This session cannot be started yet. The scheduled time begins at ${startStr}.`;
    }
    if (nowMinutes > endMinutes) {
      const endStr = `${String(eh).padStart(2, "0")}:${String(em).padStart(2, "0")}`;
      return `The scheduled time for this course has passed. Sessions for this slot close at ${endStr}.`;
    }

    return null;
  })();

  // Reset schedule and class when course is changed — the auto-fill effects
  // will re-populate them once their respective queries resolve.
  const handleCourseChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setSessionError(null);
    const courseId = e.target.value;
    setValue("course_id", courseId, { shouldValidate: true });
    setValue("schedule_id", "", { shouldValidate: false });
    setValue("class_id", "", { shouldValidate: false });
  };

  const onSubmit = async (formData: AttendanceForm) => {
    setSessionError(null);
    try {
      const resolvedCourseName =
        teacherCourses.find((c) => c.id === formData.course_id)?.name ?? "";

      const response = await attendanceService.startSession({
        course_id: Number(formData.course_id),
        schedule_id: formData.schedule_id ? Number(formData.schedule_id) : null,
        session_type: formData.session_type,
      });

      // Backend returns { session, ok, message } or the session object directly
      const session = response?.session ?? response;
      const sessionId = session?.id ?? response?.id;
      if (!sessionId) {
        startSession();
        return;
      }
      setActiveSession(Number(sessionId), resolvedCourseName);
      startSession({ sessionId: Number(sessionId), courseName: resolvedCourseName });
    } catch (err: any) {
      // Backend wraps errors as { error: { message: "..." } }; fall back to { detail: "..." }
      const data = err?.response?.data;
      const detail: string =
        (typeof data?.error?.message === "string" ? data.error.message : null) ??
        (typeof data?.detail === "string" ? data.detail : null) ??
        err?.message ??
        "Failed to start the attendance session. Please try again.";
      setSessionError(detail);
    }
  };

  const isScanningActive = sessionState === "waiting_for_face";

  return (
    <div className="relative flex-1 flex flex-col items-center justify-center min-h-[calc(100vh-4rem)] px-4 py-2 overflow-hidden">
      {/* Dynamic Background Glow based on state */}
      <div
        className={cn(
          "absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] rounded-full blur-[120px] opacity-20 pointer-events-none transition-colors duration-1000",
          sessionState === "idle" && "bg-transparent",
          isScanningActive && "bg-primary",
          sessionState === "success" && "bg-emerald-500",
          sessionState === "failed" && "bg-rose-500",
          sessionState === "already_marked" && "bg-yellow-400",
          sessionState === "partial_face" && "bg-orange-500",
          sessionState === "low_light" && "bg-amber-400",
        )}
      />

      <AnimatePresence mode="wait">
        {sessionState === "idle" ? (
          <motion.div
            key="start-screen"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9, filter: "blur(10px)" }}
            transition={{ duration: 0.4 }}
            className="w-full max-w-3xl z-10 mx-auto"
          >
            <div className="flex flex-col gap-2 mb-6">
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white tracking-tight">
                Start Attendance
              </h2>
              <p className="text-gray-500 dark:text-gray-400">
                Create a new attendance session for your class.
              </p>
            </div>

            {sessionError && (
              <div className="mb-6 rounded-2xl border border-rose-300 dark:border-rose-500/30 bg-rose-50 dark:bg-rose-500/10 p-4 text-sm text-rose-800 dark:text-rose-200 space-y-1">
                <div className="font-semibold flex items-center gap-2">
                  <span>⛔</span>
                  <span>Unable to start session</span>
                </div>
                <div className="text-rose-600 dark:text-rose-300">{sessionError}</div>
              </div>
            )}

            {/* Amber: resume MY own active session */}
            {myActiveSession && (
              <div className="mb-6 rounded-2xl border border-amber-200 dark:border-amber-500/20 bg-amber-50 dark:bg-amber-500/10 p-4 text-sm text-amber-800 dark:text-amber-200 space-y-3">
                <div className="font-semibold">You have an active session running</div>
                <div className="text-amber-700 dark:text-amber-300">
                  {teacherCourses.find((c) => c.id === String(myActiveSession.course_id))?.name ?? `Course ${myActiveSession.course_id}`}
                  {" · "}Session #{myActiveSession.id}
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
              <div className="mb-6 rounded-2xl border border-rose-200 dark:border-rose-500/20 bg-rose-50 dark:bg-rose-500/10 p-4 text-sm text-rose-800 dark:text-rose-200 space-y-1">
                <div className="font-semibold">A session is already running for this course</div>
                <div className="text-rose-600 dark:text-rose-300">
                  {courseSessionStarterRole === "admin"
                    ? "An admin has already started an attendance session for this course. You cannot start a duplicate session."
                    : "Another teacher has already started an attendance session for this course. You cannot start a duplicate session."}
                </div>
              </div>
            )}

            {scheduleTimeError && (
              <div className="mb-6 rounded-2xl border border-rose-300 dark:border-rose-500/30 bg-rose-50 dark:bg-rose-500/10 p-4 text-sm text-rose-800 dark:text-rose-200 space-y-1">
                <div className="font-semibold flex items-center gap-2">
                  <span>⛔</span>
                  <span>Session not allowed at this time</span>
                </div>
                <div className="text-rose-600 dark:text-rose-300">{scheduleTimeError}</div>
              </div>
            )}

            {courseQueryError ? (
              <div className="mb-6 rounded-2xl border border-rose-200 dark:border-rose-500/20 bg-rose-50 dark:bg-rose-500/10 p-4 text-sm text-rose-700 dark:text-rose-200 flex flex-col gap-3">
                <div>Unable to load your assigned courses. Please retry.</div>
                <button
                  type="button"
                  onClick={() => refetchCourses()}
                  className="inline-flex items-center justify-center rounded-full bg-rose-500 px-4 py-2 text-sm font-semibold text-white hover:bg-rose-600 transition"
                >
                  Retry
                </button>
              </div>
            ) : null}

            <Card className="glass-card shadow-2xl shadow-primary/10 border-white/5 relative overflow-hidden">
              {/* Top gradient accent */}
              <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-brand opacity-80" />

              <CardContent className="p-6 md:p-8">
                <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
                  <div className="grid grid-cols-1 gap-6">
                    <div className="space-y-2">
                      <label className="text-sm font-medium text-gray-700 dark:text-gray-300 ml-1">
                        Course Name <span className="text-red-500">*</span>
                      </label>
                      <select
                        {...register("course_id")}
                        onChange={handleCourseChange}
                        disabled={coursesLoading || !hasCourses}
                        className={`flex h-12 w-full rounded-xl glass-input px-4 py-2 text-sm text-gray-900 dark:text-gray-100 transition-all appearance-none bg-no-repeat bg-[right_1rem_center] bg-[length:1em_1em] ${errors.course_id ? "border-red-500 focus:border-red-500 focus:ring-red-500" : ""}`}
                        style={{
                          backgroundImage: `url("data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%2224%22%20height%3D%2224%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%236B7280%22%20stroke-width%3D%222%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpolyline%20points%3D%226%209%2012%2015%2018%209%22%3E%3C%2Fpolyline%3E%3C%2Fsvg%3E")`,
                        }}
                      >
                        <option value="" disabled className="text-gray-500">
                          {coursesLoading
                            ? 'Loading assigned courses...'
                            : hasCourses
                            ? 'Select a course...'
                            : 'No assigned courses available'}
                        </option>
                        {teacherCourses.map((course) => (
                          <option
                            key={course.id}
                            value={course.id}
                            className="bg-white dark:bg-dark-bg text-gray-900 dark:text-white"
                          >
                            {course.name}
                          </option>
                        ))}
                      </select>
                      {errors.course_id && (
                        <p className="text-xs text-red-500 ml-1 mt-1">
                          {errors.course_id.message}
                        </p>
                      )}
                      {!coursesLoading && !hasCourses && (
                        <p className="text-xs text-gray-500 dark:text-gray-400 ml-1 mt-1">
                          No assigned courses were found for your account.
                        </p>
                      )}
                    </div>

                    {/* Schedule / Time-slot selector */}
                    <div className="space-y-2">
                      <label className="text-sm font-medium text-gray-700 dark:text-gray-300 ml-1">
                        Schedule / Time Slot <span className="text-red-500">*</span>
                      </label>
                      <select
                        {...register("schedule_id")}
                        value={selectedScheduleId}
                        onChange={(e) => {
                          register("schedule_id").onChange(e);
                          setSessionError(null);
                        }}
                        disabled={!selectedCourseId || schedulesLoading}
                        className={`flex h-12 w-full rounded-xl glass-input px-4 py-2 text-sm text-gray-900 dark:text-gray-100 transition-all appearance-none bg-no-repeat bg-[right_1rem_center] bg-[length:1em_1em] ${errors.schedule_id ? "border-red-500" : ""}`}
                        style={{ backgroundImage: `url("data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%2224%22%20height%3D%2224%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%236B7280%22%20stroke-width%3D%222%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpolyline%20points%3D%226%209%2012%2015%2018%209%22%3E%3C%2Fpolyline%3E%3C%2Fsvg%3E")` }}
                      >
                        <option value="" disabled className="text-gray-500">
                          {!selectedCourseId
                            ? "Select a course first"
                            : schedulesLoading
                            ? "Loading schedules..."
                            : scheduleOptions.length === 0
                            ? "No schedules found for this course"
                            : "Auto-filled — change if needed"}
                        </option>
                        {scheduleOptions.map((s: any) => (
                          <option key={s.id} value={String(s.id)} className="bg-white dark:bg-dark-bg text-gray-900 dark:text-white">
                            {s.weekday.toUpperCase()} · {s.start_time} – {s.end_time} ({s.grace_period_minutes} min grace)
                          </option>
                        ))}
                      </select>
                      {errors.schedule_id && (
                        <p className="text-xs text-red-500 ml-1 mt-1">{errors.schedule_id.message}</p>
                      )}
                    </div>

                    <div className="space-y-2">
                      <label className="text-sm font-medium text-gray-700 dark:text-gray-300 ml-1">
                        Class / Section
                      </label>
                      <Input
                        {...register("class_id")}
                        readOnly
                        placeholder={
                          !selectedCourseId
                            ? "Select a course first"
                            : courseClasses === undefined
                            ? "Loading..."
                            : courseClasses.length === 0
                            ? "No class assigned to this course"
                            : "Auto-filled"
                        }
                        className="bg-gray-100/50 dark:bg-white/5 cursor-not-allowed text-gray-500 dark:text-gray-400"
                        error={errors.class_id?.message}
                      />
                    </div>

                    <div className="space-y-2">
                      <label className="text-sm font-medium text-gray-700 dark:text-gray-300 ml-1">
                        Session Type <span className="text-red-500">*</span>
                      </label>
                      <select
                        {...register("session_type")}
                        className={`flex h-12 w-full rounded-xl glass-input px-4 py-2 text-sm text-gray-900 dark:text-gray-100 transition-all appearance-none bg-no-repeat bg-[right_1rem_center] bg-[length:1em_1em] ${errors.session_type ? "border-red-500 focus:border-red-500 focus:ring-red-500" : ""}`}
                        style={{
                          backgroundImage: `url("data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%2224%22%20height%3D%2224%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%236B7280%22%20stroke-width%3D%222%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpolyline%20points%3D%226%209%2012%2015%2018%209%22%3E%3C%2Fpolyline%3E%3C%2Fsvg%3E")`,
                        }}
                      >
                        <option value="" disabled className="text-gray-500">
                          Select Session Type...
                        </option>
                        <option
                          value="Lecture"
                          className="bg-white dark:bg-dark-bg text-gray-900 dark:text-white"
                        >
                          Lecture
                        </option>
                        <option
                          value="Lab"
                          className="bg-white dark:bg-dark-bg text-gray-900 dark:text-white"
                        >
                          Lab
                        </option>
                        <option
                          value="Tutorial"
                          className="bg-white dark:bg-dark-bg text-gray-900 dark:text-white"
                        >
                          Tutorial
                        </option>
                      </select>
                      {errors.session_type && (
                        <p className="text-xs text-red-500 ml-1 mt-1">
                          {errors.session_type.message}
                        </p>
                      )}
                    </div>

                    {/* Camera Index */}
                    <div className="space-y-2">
                      <label className="text-sm font-medium text-gray-700 dark:text-gray-300 ml-1">
                        Camera Index <span className="text-red-500">*</span>
                      </label>
                      <Input
                        {...register("camera_index")}
                        type="number"
                        min="0"
                        placeholder="0"
                        className="glass-input"
                        error={errors.camera_index?.message}
                      />
                    </div>

                    <div className="space-y-2">
                      <label className="text-sm font-medium text-gray-700 dark:text-gray-300 ml-1">
                        Notes (Optional)
                      </label>
                      <textarea
                        {...register("notes")}
                        placeholder="Add any specific notes for this session..."
                        className="w-full rounded-xl glass-input px-4 py-3 text-sm text-gray-900 dark:text-gray-100 transition-all min-h-[120px] resize-y placeholder:text-gray-400 focus:border-primary focus:ring-primary dark:focus:border-primary-accent dark:focus:ring-primary-accent"
                      />
                    </div>
                  </div>

                  <div className="flex items-center justify-end gap-4 pt-4 border-t border-gray-100 dark:border-white/10">
                    <Button
                      type="button"
                      variant="secondary"
                      onClick={() => reset()}
                    >
                      <X size={18} className="mr-2" />
                      Cancel
                    </Button>
                    <Button
                      type="submit"
                      size="lg"
                      className="w-full sm:w-auto px-8"
                      isLoading={isSubmitting}
                      disabled={!hasCourses || coursesLoading || !watch("course_id") || (!!courseActiveSession && !isMyActiveSessionForCourse) || !!scheduleTimeError}
                    >
                      <Play size={18} className="mr-2 fill-white" />
                      Start Session
                    </Button>
                  </div>
                </form>
              </CardContent>
            </Card>
          </motion.div>
        ) : (
          <ScannerInterface cameraIndex={cameraIndex} />
        )}
      </AnimatePresence>
    </div>
  );
}
