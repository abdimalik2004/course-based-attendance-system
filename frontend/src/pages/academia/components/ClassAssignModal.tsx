import { useEffect, useState, useMemo } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { AlertTriangle } from "lucide-react";
import { Modal } from "@/components/ui/Modal";
import { Select } from "@/components/ui/Select";
import { Button } from "@/components/ui/Button";
import { useAcademiaStore } from "@/store/useAcademiaStore";

const schema = z.object({
  classId: z.string().min(1, "Class is required"),
  courseId: z.string().min(1, "Course is required"),
  facultyId: z.string().min(1, "Faculty is required"),
  departmentId: z.string().min(1, "Department is required"),
});

type FormData = z.infer<typeof schema>;

export function ClassAssignModal() {
  const {
    classAssignModal,
    closeModal,
    addClassAssignment,
    updateClassAssignment,
    classes,
    courses,
    faculties,
    departments,
    classAssignments,
  } = useAcademiaStore();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const isOpen = classAssignModal?.isOpen || false;
  const mode = classAssignModal?.mode || "create";
  const record = classAssignModal?.record;

  const {
    register,
    handleSubmit,
    formState: { errors },
    reset,
    watch,
    setValue,
  } = useForm<FormData>({
    resolver: zodResolver(schema),
  });

  const selectedFacultyId = watch("facultyId");
  const selectedClassId = watch("classId");
  const selectedCourseId = watch("courseId");

  // ── Derived lookups ──────────────────────────────────────────────────────
  const selectedCourse = useMemo(
    () => courses.find((c) => c.id === selectedCourseId) ?? null,
    [courses, selectedCourseId],
  );
  const selectedClass = useMemo(
    () => classes.find((c) => c.id === selectedClassId) ?? null,
    [classes, selectedClassId],
  );

  // ── Faculty mismatch detection ──────────────────────────────────────────
  // Fires when both a course AND a class are chosen but they belong to
  // different faculties — e.g. Agriculture class → Engineering course.
  const facultyMismatch = useMemo(() => {
    if (!selectedClass || !selectedCourse) return null;
    if (selectedClass.facultyId === selectedCourse.facultyId) return null;
    const classF =
      faculties.find((f) => f.id === selectedClass.facultyId)?.name ??
      "another faculty";
    const courseF =
      faculties.find((f) => f.id === selectedCourse.facultyId)?.name ??
      "the course's faculty";
    return `"${selectedClass.name}" belongs to ${classF}, but the selected course belongs to ${courseF}. A class can only be assigned to courses within the same faculty.`;
  }, [selectedClass, selectedCourse, faculties]);

  // ── Courses not yet assigned to any class (create mode only) ────────────
  const availableCourses = useMemo(() => {
    if (mode === "edit") return courses; // keep all options so the saved value still renders
    const assignedCourseIds = new Set(classAssignments.map((a) => a.courseId));
    return courses.filter((c) => !assignedCourseIds.has(c.id));
  }, [courses, classAssignments, mode]);

  // ── Department options (filtered by auto-filled faculty) ─────────────────
  const filteredDepartments = useMemo(
    () =>
      departments
        .filter((d) => d.facultyId === selectedFacultyId)
        .map((d) => ({ value: d.id, label: d.name })),
    [departments, selectedFacultyId],
  );

  // ── Class options ────────────────────────────────────────────────────────
  // In create mode: when a course is selected, restrict classes to the same
  // faculty so the user cannot accidentally pick a mismatched class.
  const classOptions = useMemo(() => {
    const pool =
      mode === "create" && selectedCourse
        ? classes.filter((c) => c.facultyId === selectedCourse.facultyId)
        : classes;
    return pool.map((c) => ({ value: c.id, label: c.name }));
  }, [classes, selectedCourse, mode]);

  // ── Reset on open ────────────────────────────────────────────────────────
  useEffect(() => {
    if (isOpen) {
      setSubmitError(null);
      if (record && mode !== "create") {
        reset({
          classId: record.classId,
          courseId: record.courseId,
          facultyId: record.facultyId,
          departmentId: record.departmentId,
        });
      } else {
        reset({ classId: "", courseId: "", facultyId: "", departmentId: "" });
      }
    }
  }, [isOpen, mode, record, reset]);

  // ── Auto-fill faculty + department from course (create mode) ─────────────
  useEffect(() => {
    if (selectedCourseId && mode === "create") {
      const course = courses.find((c) => c.id === selectedCourseId);
      if (course) {
        setValue("facultyId", course.facultyId, { shouldValidate: true });
        // Defer department setValue by one tick so filteredDepartments (which depends on
        // selectedFacultyId) has time to recompute with the new faculty before we try to
        // set the value — otherwise the <select> has no matching option yet and silently ignores it.
        const deptId = course.departmentId;
        setTimeout(() => {
          setValue("departmentId", deptId, { shouldValidate: true });
        }, 0);
        // Clear a previously selected class that no longer belongs to this faculty
        if (selectedClass && selectedClass.facultyId !== course.facultyId) {
          setValue("classId", "", { shouldValidate: false });
        }
      }
    }
  }, [selectedCourseId, courses, mode, setValue, selectedClass]);

  // ── Submit ───────────────────────────────────────────────────────────────
  const onSubmit = async (data: FormData) => {
    // Guard: block if class and course are from different faculties
    if (facultyMismatch) {
      setSubmitError(facultyMismatch);
      return;
    }
    setIsSubmitting(true);
    setSubmitError(null);
    try {
      if (mode === "edit" && record) {
        await updateClassAssignment(record.id, data);
      } else {
        await addClassAssignment(data);
      }
      closeModal("classAssign");
    } catch (error: any) {
      const msg =
        error?.response?.data?.detail ??
        error?.response?.data?.error?.message ??
        (error instanceof Error ? error.message : null) ??
        "Failed to assign class";
      setSubmitError(msg);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={() => closeModal("classAssign")}
      title="Assign Class to Course"
      className="md:max-w-md"
    >
      <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
        {/* Global submit / mismatch error */}
        {submitError && (
          <div className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700 dark:border-rose-500/20 dark:bg-rose-500/10 dark:text-rose-200">
            {submitError}
          </div>
        )}

        {/* ── 1. Course (first so faculty can auto-fill) ── */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">
            Course Title
          </label>
          <Select
            options={[
              { value: "", label: "Select Course..." },
              ...availableCourses.map((c) => {
                const alreadyAssigned = classAssignments.some(
                  (a) =>
                    a.classId === selectedClassId &&
                    a.courseId === c.id &&
                    !(mode === "edit" && record && a.id === record.id),
                );
                return {
                  value: c.id,
                  label: alreadyAssigned
                    ? `${c.code} – ${c.title} — already assigned`
                    : `${c.code} – ${c.title}`,
                  disabled: alreadyAssigned,
                };
              }),
            ]}
            {...register("courseId")}
            error={errors.courseId?.message}
            disabled={mode === "view"}
          />
          {mode === "create" && availableCourses.length === 0 && (
            <p className="text-xs text-orange-500 mt-1 ml-1">
              All courses have already been assigned to a class.
            </p>
          )}
          {selectedCourse && mode === "create" && (
            <p className="text-xs text-gray-500 mt-1 ml-1">
              Faculty and department will be filled automatically.
            </p>
          )}
        </div>

        {/* ── 2. Class (filtered to match course's faculty in create mode) ── */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">
            Class Name
          </label>
          <Select
            options={[
              {
                value: "",
                label:
                  mode === "create" && selectedCourse
                    ? `Select class from ${selectedCourse ? (faculties.find((f) => f.id === selectedCourse.facultyId)?.name ?? "same faculty") : "..."}...`
                    : "Select Class...",
              },
              ...classOptions,
            ]}
            {...register("classId")}
            error={errors.classId?.message}
            disabled={mode === "view"}
          />
          {/* Inline warning when class and course faculties don't match */}
          {facultyMismatch && (
            <div className="flex items-start gap-1.5 mt-1.5 px-1 text-xs text-amber-700 dark:text-amber-400">
              <AlertTriangle size={13} className="shrink-0 mt-0.5" />
              <span>{facultyMismatch}</span>
            </div>
          )}
          {mode === "create" && selectedCourse && classOptions.length === 0 && (
            <p className="text-xs text-orange-500 mt-1 ml-1">
              No classes found for this course's faculty.
            </p>
          )}
        </div>

        {/* ── 3. Faculty (auto-filled, read-only) ── */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">
            Faculty
          </label>
          <Select
            options={[
              { value: "", label: "Select Faculty..." },
              ...faculties.map((f) => ({ value: f.id, label: f.name })),
            ]}
            {...register("facultyId")}
            error={errors.facultyId?.message}
            disabled={mode === "view" || (mode === "create" && !!selectedCourseId)}
          />
          {mode === "create" && selectedCourseId && (
            <p className="text-xs text-gray-500 mt-1 ml-1">
              Auto-filled from course selection.
            </p>
          )}
        </div>

        {/* ── 4. Department (filtered by faculty, auto-filled) ── */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">
            Department
          </label>
          <Select
            options={[
              { value: "", label: "Select Department..." },
              ...filteredDepartments,
            ]}
            {...register("departmentId")}
            error={errors.departmentId?.message}
            disabled={
              !selectedFacultyId ||
              filteredDepartments.length === 0 ||
              mode === "view" ||
              (mode === "create" && !!selectedCourseId)
            }
          />
          {!selectedFacultyId && !selectedCourseId && (
            <p className="text-xs text-gray-500 mt-1 ml-1">
              Select a course to populate faculty and department.
            </p>
          )}
          {selectedFacultyId && filteredDepartments.length === 0 && (
            <p className="text-xs text-orange-500 mt-1 ml-1">
              No departments found for this faculty.
            </p>
          )}
        </div>

        <div className="flex items-center justify-end gap-3 pt-6 mt-6">
          <Button
            type="button"
            variant="ghost"
            onClick={() => closeModal("classAssign")}
          >
            {mode === "view" ? "Close" : "Cancel"}
          </Button>
          {mode !== "view" && (
            <Button
              type="submit"
              isLoading={isSubmitting}
              disabled={isSubmitting || !!facultyMismatch}
            >
              {mode === "edit" ? "Save Changes" : "Assign Class"}
            </Button>
          )}
        </div>
      </form>
    </Modal>
  );
}
