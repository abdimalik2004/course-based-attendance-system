import { useEffect, useMemo } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";
import { motion, AnimatePresence } from "framer-motion";
import { Play, X } from "lucide-react";
import { Card, CardContent } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { useAttendanceStore } from "@/store/useAttendanceStore";
import ScannerInterface from "@/components/attendance/ScannerInterface";
import { cn } from "@/utils/cn";
import { useQuery } from "@tanstack/react-query";
import courseService from "@/services/courseService";
import attendanceService from "@/services/attendanceService";

const attendanceSchema = z.object({
  course_id: z.string().min(1, "Course is required"),
  class_id: z.string().min(1, "Class is required"),
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
    activeSessionId,
    activeCourseName,
  } = useAttendanceStore();
  const { data: activeSessions } = useQuery({
    queryKey: ["teacherActiveSessions"],
    queryFn: () => attendanceService.listSessions({ skip: 0, limit: 200 }),
  });
  const { data } = useQuery({
    queryKey: ["teacherAttendanceCourses"],
    queryFn: () => courseService.listCourses({ skip: 0, limit: 200 }),
  });

  const teacherCourses = useMemo(() => {
    const items = data?.items ?? data ?? [];
    return items.map((course: any) => ({
      id: String(course.id),
      name: course.title ?? course.name ?? `Course ${course.id}`,
      class_id: String(course.class_id ?? course.department_id ?? course.id),
      class_name:
        course.department_name ?? course.department ?? "Assigned class",
    }));
  }, [data]);

  const currentActiveSession = useMemo(() => {
    const sessions = activeSessions?.items ?? activeSessions ?? [];
    return (
      sessions.find(
        (session: any) =>
          String(session.status ?? "").toUpperCase() === "ACTIVE",
      ) ?? null
    );
  }, [activeSessions]);

  // Reset session when navigating to this page
  useEffect(() => {
    resetSession();
  }, [resetSession]);

  const {
    register,
    handleSubmit,
    setValue,
    reset,
    formState: { errors, isSubmitting },
  } = useForm<AttendanceForm>({
    resolver: zodResolver(attendanceSchema),
    defaultValues: {
      course_id: "",
      class_id: "",
      session_type: "" as any,
      camera_index: 0,
      notes: "",
    },
  });

  // Auto-fill class when course is selected
  const handleCourseChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const courseId = e.target.value;
    setValue("course_id", courseId, { shouldValidate: true });

    const course = teacherCourses.find((c) => c.id === courseId);
    if (course) {
      setValue("class_id", course.class_name, { shouldValidate: true });
    } else {
      setValue("class_id", "", { shouldValidate: true });
    }
  };

  const onSubmit = async (formData: AttendanceForm) => {
    const response = await attendanceService.startSession({
      course_id: Number(formData.course_id),
      schedule_id: null,
    });

    const session = response?.session;
    const selectedCourse = teacherCourses.find(
      (course) => course.id === formData.course_id,
    );
    if (session?.id) {
      setActiveSession(Number(session.id), selectedCourse?.name ?? "");
      startSession({
        sessionId: Number(session.id),
        courseName: selectedCourse?.name ?? "",
      });
      return;
    }

    startSession();
  };

  const isScanningActive =
    sessionState === "starting" ||
    sessionState === "waiting_for_face" ||
    sessionState === "face_detected" ||
    sessionState === "scanning";

  return (
    <div className="relative flex-1 flex flex-col items-center justify-center min-h-[calc(100vh-4rem)] p-4 overflow-hidden">
      {/* Dynamic Background Glow based on state */}
      <div
        className={cn(
          "absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] rounded-full blur-[120px] opacity-20 pointer-events-none transition-colors duration-1000",
          sessionState === "idle" && "bg-transparent",
          isScanningActive && "bg-primary",
          sessionState === "success" && "bg-emerald-500",
          sessionState === "failed" && "bg-rose-500",
          (sessionState === "low_light" ||
            sessionState === "partial_face" ||
            sessionState === "already_marked") &&
            "bg-yellow-500",
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

            {currentActiveSession ? (
              <div className="mb-6 rounded-2xl border border-amber-200 dark:border-amber-500/20 bg-amber-50 dark:bg-amber-500/10 p-4 text-sm text-amber-800 dark:text-amber-200 space-y-1">
                <div className="font-semibold">
                  Active session already running
                </div>
                <div>Session ID: {currentActiveSession.id}</div>
                <div>Course ID: {currentActiveSession.course_id}</div>
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
                        className={`flex h-12 w-full rounded-xl glass-input px-4 py-2 text-sm text-gray-900 dark:text-gray-100 transition-all appearance-none bg-no-repeat bg-[right_1rem_center] bg-[length:1em_1em] ${errors.course_id ? "border-red-500 focus:border-red-500 focus:ring-red-500" : ""}`}
                        style={{
                          backgroundImage: `url("data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%2224%22%20height%3D%2224%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%236B7280%22%20stroke-width%3D%222%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpolyline%20points%3D%226%209%2012%2015%2018%209%22%3E%3C%2Fpolyline%3E%3C%2Fsvg%3E")`,
                        }}
                      >
                        <option value="" disabled className="text-gray-500">
                          Select a course...
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
                    </div>

                    <div className="space-y-2">
                      <label className="text-sm font-medium text-gray-700 dark:text-gray-300 ml-1">
                        Class / Section
                      </label>
                      <Input
                        {...register("class_id")}
                        readOnly
                        placeholder="Auto-filled based on course"
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

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 rounded-2xl border border-gray-200 dark:border-white/10 bg-white/60 dark:bg-white/5 p-4 text-sm text-gray-600 dark:text-gray-300">
                    <div>
                      <span className="block text-xs uppercase tracking-wider text-gray-500 dark:text-gray-400">
                        Backend session
                      </span>
                      <span className="font-medium text-gray-900 dark:text-white">
                        {activeSessionId
                          ? `SES-${activeSessionId}`
                          : "Not started"}
                      </span>
                    </div>
                    <div>
                      <span className="block text-xs uppercase tracking-wider text-gray-500 dark:text-gray-400">
                        Selected course
                      </span>
                      <span className="font-medium text-gray-900 dark:text-white">
                        {activeCourseName || "Not selected"}
                      </span>
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
          <ScannerInterface />
        )}
      </AnimatePresence>
    </div>
  );
}
