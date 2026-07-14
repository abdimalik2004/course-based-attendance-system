import { useEffect, useState, useMemo } from "react";
import { Calendar, Edit2, Trash2, Plus } from "lucide-react";
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
import { ViewButton } from "@/components/ui/ViewButton";
import { ViewModal } from "@/components/ui/ViewModal";
import { useFacultyStore } from "@/store/useFacultyStore";
import { ScheduleModal } from "@/components/faculty/ScheduleModal";
import { ConfirmDeleteModal } from "@/components/academia/ConfirmDeleteModal";
import type { CourseSchedule } from "@/store/useFacultyStore";

export default function ScheduleCourse() {
  const {
    schedules,
    courses,
    isLoading,
    error,
    fetchData,
    openModal,
    deleteSchedule,
  } = useFacultyStore();
  const [deleteId, setDeleteId] = useState<string | null>(null);
  const [viewSchedule, setViewSchedule] = useState<CourseSchedule | null>(null);
  const [searchTerm, setSearchTerm] = useState("");

  const getCourse = (id: string) => courses.find((c) => c.id === id);

  const filteredSchedules = useMemo(() => {
    return schedules.filter((schedule) => {
      const course = getCourse(schedule.courseId);
      return (
        (course?.title || "").toLowerCase().includes(searchTerm.toLowerCase()) ||
        (course?.code || "").toLowerCase().includes(searchTerm.toLowerCase()) ||
        schedule.weekdays.join(" ").toLowerCase().includes(searchTerm.toLowerCase())
      );
    });
  }, [schedules, searchTerm, courses]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

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
        <Button onClick={() => openModal("schedule", "create")} className="gap-2">
          <Plus size={20} />
          Add Schedule
        </Button>
      </div>

      {error ? (
        <div className="rounded-2xl border border-rose-200 dark:border-rose-500/20 bg-rose-50 dark:bg-rose-500/10 p-4 text-sm text-rose-700 dark:text-rose-200">
          {error}
        </div>
      ) : null}

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
                              onClick={() => openModal("schedule", "edit", schedule)}
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
            await deleteSchedule(deleteId);
            setDeleteId(null);
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
                  value: getCourse(viewSchedule.courseId)?.title || "Unknown Course",
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
