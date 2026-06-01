import { useEffect, useState, useMemo } from "react";
import { useForm, Controller } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";
import { Calendar, Edit2, Trash2, Clock } from "lucide-react";
import { Card, CardContent } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/Table";
import { Input } from "@/components/ui/Input";
import { Select } from "@/components/ui/Select";
import { ViewButton } from "@/components/ui/ViewButton";
import { ViewModal } from "@/components/ui/ViewModal";
import { useFacultyStore } from "@/store/useFacultyStore";
import { ScheduleModal } from "@/components/faculty/ScheduleModal";
import { ConfirmDeleteModal } from "@/components/academia/ConfirmDeleteModal";
import type { CourseSchedule } from "@/store/useFacultyStore";
import { cn } from "@/utils/cn";

const weekdays = ["sat", "sun", "mon", "tue", "wed", "thu", "fri"] as const;

const scheduleSchema = z.object({
  courseId: z.string().min(1, "Course is required"),
  weekdays: z.array(z.string()).min(1, "Select at least one weekday"),
  startTime: z.string().min(1, "Start time is required"),
  endTime: z.string().min(1, "End time is required"),
  gracePeriod: z.number().min(0, "Grace period must be positive"),
});
type ScheduleForm = z.infer<typeof scheduleSchema>;

export default function ScheduleCourse() {
  const {
    schedules,
    courses,
    isLoading,
    error,
    fetchData,
    openModal,
    addSchedule,
    deleteSchedule,
  } = useFacultyStore();
  const [deleteId, setDeleteId] = useState<string | null>(null);
  const [viewSchedule, setViewSchedule] = useState<CourseSchedule | null>(null);
  const [searchTerm, setSearchTerm] = useState("");

  const getCourse = (id: string) => courses.find((c) => c.id === id);

  const filteredSchedules = useMemo(() => {
    return schedules.filter((schedule) => {
      const course = getCourse(schedule.courseId);
      const searchMatch =
        (course?.title || "")
          .toLowerCase()
          .includes(searchTerm.toLowerCase()) ||
        (course?.code || "").toLowerCase().includes(searchTerm.toLowerCase()) ||
        schedule.weekdays
          .join(" ")
          .toLowerCase()
          .includes(searchTerm.toLowerCase());
      return searchMatch;
    });
  }, [schedules, searchTerm, courses]);

  const {
    register,
    handleSubmit,
    control,
    reset,
    formState: { errors, isSubmitting },
  } = useForm<ScheduleForm>({
    resolver: zodResolver(scheduleSchema),
    defaultValues: {
      weekdays: [],
      gracePeriod: 15,
    },
  });

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const onSubmit = async (data: ScheduleForm) => {
    try {
      const payload = {
        course_id: Number(data.courseId),
        weekday: data.weekdays,
        start_time: data.startTime + ":00",
        end_time: data.endTime + ":00",
        grace_period_minutes: Number(data.gracePeriod),
      };

      console.log("SENDING PAYLOAD:", payload);

      await addSchedule(payload);

      reset({
        courseId: "",
        weekdays: [],
        startTime: "",
        endTime: "",
        gracePeriod: 15,
      });
    } catch (error) {
      console.error("Failed to save schedule:", error);
    }
  };

  // Collect the set of course IDs that already have at least one schedule entry.
  const scheduledCourseIds = useMemo(
    () => new Set(schedules.map((s) => s.courseId)),
    [schedules],
  );

  const courseOptions = [
    { value: "", label: "Select Course..." },
    ...courses.map((c) => {
      const alreadyScheduled = scheduledCourseIds.has(c.id);
      return {
        value: c.id,
        label: alreadyScheduled
          ? `${c.code} - ${c.title} (already scheduled)`
          : `${c.code} - ${c.title}`,
        disabled: alreadyScheduled,
      };
    }),
  ];

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-gray-900 dark:text-gray-100 flex items-center gap-3">
            <Calendar className="text-primary" size={28} />
            Schedule Course
          </h1>
          <p className="text-gray-500 dark:text-gray-400 mt-1">
            Create and manage course schedules.
          </p>
        </div>
      </div>

      {error ? (
        <div className="rounded-2xl border border-rose-200 dark:border-rose-500/20 bg-rose-50 dark:bg-rose-500/10 p-4 text-sm text-rose-700 dark:text-rose-200">
          {error}
        </div>
      ) : null}

      {/* Inline Form Section */}
      <Card className="glass-panel overflow-visible">
        <CardContent className="p-6">
          <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="col-span-1 md:col-span-2 space-y-1">
                <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Course
                </label>
                <Select
                  options={courseOptions}
                  error={errors.courseId?.message}
                  {...register("courseId")}
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">
                  Start Time
                </label>
                <div className="relative">
                  <Clock
                    className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400"
                    size={18}
                  />
                  <Input
                    type="time"
                    className="pl-10 text-gray-900 dark:text-white dark:[color-scheme:dark]"
                    {...register("startTime")}
                  />
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">
                  End Time
                </label>
                <div className="relative">
                  <Clock
                    className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400"
                    size={18}
                  />
                  <Input
                    type="time"
                    className="pl-10 text-gray-900 dark:text-white dark:[color-scheme:dark]"
                    {...register("endTime")}
                  />
                </div>
              </div>
            </div>

            <div className="space-y-2">
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                Weekdays
              </label>
              <Controller
                name="weekdays"
                control={control}
                render={({ field }) => (
                  <div className="flex flex-wrap gap-2">
                    {weekdays.map((day) => {
                      const isSelected = field.value.includes(day);
                      return (
                        <button
                          key={day}
                          type="button"
                          onClick={() => {
                            const newValue = isSelected
                              ? field.value.filter((d) => d !== day)
                              : [...field.value, day];
                            field.onChange(newValue);
                          }}
                          className={cn(
                            "px-4 py-2 rounded-full text-sm font-medium capitalize transition-all duration-300 border",
                            isSelected
                              ? "bg-purple-500/20 border-purple-500 text-purple-400 shadow-[0_0_10px_rgba(168,85,247,0.4)]"
                              : "bg-white/5 border-white/10 text-gray-400 hover:text-gray-200 hover:bg-white/10",
                          )}
                        >
                          {day}
                        </button>
                      );
                    })}
                  </div>
                )}
              />
              {errors.weekdays && (
                <p className="text-sm text-red-500 mt-1">
                  {errors.weekdays.message}
                </p>
              )}
            </div>

            <div className="flex items-end gap-4">
              <div className="w-48 space-y-1">
                <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Grace Period (Minutes)
                </label>
                <Input
                  type="number"
                  error={errors.gracePeriod?.message}
                  {...register("gracePeriod", { valueAsNumber: true })}
                />
              </div>
              <Button
                type="submit"
                isLoading={isSubmitting}
                className="ml-auto"
              >
                Save Schedule
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>

      {/* Schedule Table */}
      <Card>
        <CardContent className="p-0">
          <div className="p-4 border-b border-gray-200 dark:border-gray-700 flex justify-between items-center">
            <Input
              placeholder="Search schedules..."
              className="max-w-sm"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
          <div className="overflow-x-auto custom-scrollbar">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Course</TableHead>
                  <TableHead>Weekdays</TableHead>
                  <TableHead>Time</TableHead>
                  <TableHead>Grace Period</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {isLoading ? (
                  Array.from({ length: 3 }).map((_, i) => (
                    <TableRow key={`skeleton-${i}`}>
                      <TableCell colSpan={5}>
                        <div className="h-4 bg-gray-200 dark:bg-white/10 rounded animate-pulse" />
                      </TableCell>
                    </TableRow>
                  ))
                ) : filteredSchedules.length === 0 ? (
                  <TableRow>
                    <TableCell
                      colSpan={5}
                      className="text-center text-gray-500 py-8"
                    >
                      No schedules found matching your search.
                    </TableCell>
                  </TableRow>
                ) : (
                  filteredSchedules.map((schedule) => {
                    const course = getCourse(schedule.courseId);

                    return (
                      <TableRow
                        key={schedule.id}
                        className="group hover:bg-white/5 transition-colors"
                      >
                        <TableCell>
                          <div className="font-medium text-gray-900 dark:text-white">
                            {course?.title || "Unknown Course"}
                          </div>
                          <div className="text-sm text-gray-500">
                            {course?.code}
                          </div>
                        </TableCell>
                        <TableCell>
                          <div className="flex gap-1 flex-wrap">
                            {schedule.weekdays.map((d) => (
                              <span
                                key={d}
                                className="px-2 py-0.5 rounded text-xs font-medium bg-purple-500/10 text-purple-400 border border-purple-500/20 capitalize"
                              >
                                {d}
                              </span>
                            ))}
                          </div>
                        </TableCell>
                        <TableCell className="font-medium text-gray-900 dark:text-white">
                          {schedule.startTime} - {schedule.endTime}
                        </TableCell>
                        <TableCell className="text-gray-500">
                          {schedule.gracePeriod} mins
                        </TableCell>
                        <TableCell className="text-right">
                          <div className="flex items-center justify-end gap-2">
                            <ViewButton
                              onClick={() => setViewSchedule(schedule)}
                              tooltip="View"
                            />
                            <button
                              type="button"
                              onClick={() =>
                                openModal("schedule", "edit", schedule)
                              }
                              className="p-1.5 rounded-lg text-amber-500 hover:text-amber-600 hover:bg-amber-50 dark:hover:bg-amber-500/10 transition-colors"
                              title="Edit"
                            >
                              <Edit2 size={16} />
                            </button>
                            <button
                              type="button"
                              onClick={() => setDeleteId(schedule.id)}
                              className="p-1.5 rounded-lg text-red-500 hover:text-red-600 hover:bg-red-50 dark:hover:bg-red-500/10 transition-colors"
                              title="Delete"
                            >
                              <Trash2 size={16} />
                            </button>
                          </div>
                        </TableCell>
                      </TableRow>
                    );
                  })
                )}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>

      <ScheduleModal />

      <ConfirmDeleteModal
        isOpen={deleteId !== null}
        onClose={() => setDeleteId(null)}
        onConfirm={async () => {
          if (deleteId) {
            try {
              await deleteSchedule(deleteId);
              await fetchData();
            } finally {
              setDeleteId(null);
            }
          }
        }}
        title="Delete Schedule"
        message="Are you sure you want to delete this course schedule? This action cannot be undone."
      />

      <ViewModal
        isOpen={!!viewSchedule}
        onClose={() => setViewSchedule(null)}
        title="Course Schedule Details"
        data={
          viewSchedule
            ? [
                {
                  label: "Course",
                  value:
                    getCourse(viewSchedule.courseId)?.title || "Unknown Course",
                },
                {
                  label: "Course Code",
                  value: getCourse(viewSchedule.courseId)?.code,
                },
                {
                  label: "Weekdays",
                  value: (
                    <div className="flex gap-1 flex-wrap">
                      {viewSchedule.weekdays.map((d) => (
                        <span
                          key={d}
                          className="px-2 py-0.5 rounded text-xs font-medium bg-purple-500/10 text-purple-400 border border-purple-500/20 capitalize"
                        >
                          {d}
                        </span>
                      ))}
                    </div>
                  ),
                },
                {
                  label: "Time",
                  value: `${viewSchedule.startTime} - ${viewSchedule.endTime}`,
                },
                {
                  label: "Grace Period",
                  value: `${viewSchedule.gracePeriod} mins`,
                },
              ]
            : null
        }
      />
    </div>
  );
}
