import { useEffect, useState, useMemo } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { Modal } from "@/components/ui/Modal";
import { Select } from "@/components/ui/Select";
import { Button } from "@/components/ui/Button";
import { useAcademiaStore } from "@/store/useAcademiaStore";

const schema = z.object({
  courseId: z.string().min(1, "Course is required"),
  facultyId: z.string().min(1, "Faculty is required"),
  departmentId: z.string().min(1, "Department is required"),
  semester: z.string().min(1, "Semester is required"),
});

type FormData = z.infer<typeof schema>;

/** Extract the numeric semester from a term name like "Semester 2" → 2 */
function semesterFromTerm(term: string): number {
  const match = term.match(/\d+/);
  return match ? parseInt(match[0], 10) : 0;
}

export function CourseAssignModal() {
  const {
    courseAssignModal,
    closeModal,
    addCourseAssignment,
    updateCourseAssignment,
    courses,
    faculties,
    departments,
    structures,
    courseAssignments,
  } = useAcademiaStore();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  // Local state — not part of form validation, just drives the semester dropdown
  const [selectedAcademicYear, setSelectedAcademicYear] = useState<string>("");

  const isOpen = courseAssignModal?.isOpen || false;
  const mode = courseAssignModal?.mode || "create";
  const record = courseAssignModal?.record;

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

  const selectedCourseId = watch("courseId");
  const selectedFacultyId = watch("facultyId");

  // ── Auto-fill faculty + department when course changes (create mode) ────────
  useEffect(() => {
    if (selectedCourseId) {
      const course = courses.find((c) => c.id === selectedCourseId);
      if (course) {
        setValue("facultyId", course.facultyId, { shouldValidate: true });
        setValue("departmentId", course.departmentId, { shouldValidate: true });
      }
    }
  }, [selectedCourseId, courses, setValue]);

  // ── Reset on modal open ───────────────────────────────────────────────────
  useEffect(() => {
    if (isOpen) {
      setSubmitError(null);
      if (record && mode !== "create") {
        // In edit mode, pre-select the academic year that owns this assignment
        const ownerStructure = structures.find((s) => s.id === record.academicYearId);
        setSelectedAcademicYear(ownerStructure?.academicYear ?? "");
        reset({
          courseId: record.courseId,
          facultyId: record.facultyId,
          departmentId: record.departmentId,
          semester: record.semester.toString(),
        });
      } else {
        setSelectedAcademicYear("");
        reset({ courseId: "", facultyId: "", departmentId: "", semester: "" });
      }
    }
  }, [isOpen, mode, record, reset, structures]);

  // ── Courses not yet assigned to any semester (create mode only) ─────────
  const availableCourses = useMemo(() => {
    if (mode === "edit") return courses; // dropdown is disabled in edit mode anyway
    const assignedCourseIds = new Set(courseAssignments.map((a) => a.courseId));
    return courses.filter((c) => !assignedCourseIds.has(c.id));
  }, [courses, courseAssignments, mode]);

  // ── Unique academic year labels ───────────────────────────────────────────
  const academicYearOptions = useMemo(() => {
    const unique = Array.from(new Set(structures.map((s) => s.academicYear))).sort();
    return unique.map((y) => ({ value: y, label: y }));
  }, [structures]);

  // ── Structures matching the chosen academic year ──────────────────────────
  const structuresForYear = useMemo(
    () => structures.filter((s) => s.academicYear === selectedAcademicYear),
    [structures, selectedAcademicYear],
  );

  // ── Semester options: terms of the chosen academic year, with conflict check ─
  const semesterOptions = useMemo(() => {
    const assignedSemesters = new Set(
      courseAssignments
        .filter((a) => {
          // Only check conflicts for the same course
          if (a.courseId !== selectedCourseId) return false;
          // In edit mode, don't count the current record
          if (mode === "edit" && record && a.id === record.id) return false;
          return true;
        })
        .map((a) => String(a.semester)),
    );

    return structuresForYear.map((s) => {
      const semNum = semesterFromTerm(s.term);
      const taken = assignedSemesters.has(String(semNum));
      return {
        value: String(semNum),
        label: taken ? `${s.term} — already assigned` : s.term,
        disabled: taken,
      };
    });
  }, [structuresForYear, courseAssignments, selectedCourseId, mode, record]);

  // ── Submit ────────────────────────────────────────────────────────────────
  const onSubmit = async (data: FormData) => {
    setIsSubmitting(true);
    setSubmitError(null);
    try {
      const semesterNum = parseInt(data.semester, 10);
      if (mode === "edit" && record) {
        await updateCourseAssignment(record.id, {
          courseId: data.courseId,
          facultyId: data.facultyId,
          departmentId: data.departmentId,
          semester: semesterNum,
        });
      } else {
        await addCourseAssignment({
          courseId: data.courseId,
          facultyId: data.facultyId,
          departmentId: data.departmentId,
          semester: semesterNum,
        });
      }
      closeModal("courseAssign");
    } catch (error: any) {
      const msg =
        error?.response?.data?.detail ??
        error?.response?.data?.error?.message ??
        (error instanceof Error ? error.message : null) ??
        "Failed to assign course";
      setSubmitError(msg);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={() => closeModal("courseAssign")}
      title="Assign Course to Semester"
      className="md:max-w-md"
    >
      <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
        {/* Error banner */}
        {submitError && (
          <div className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700 dark:border-rose-500/20 dark:bg-rose-500/10 dark:text-rose-200">
            {submitError}
          </div>
        )}

        {/* Edit-mode hint */}
        {mode === "edit" && (
          <div className="rounded-xl border border-blue-200 bg-blue-50 px-4 py-2.5 text-sm text-blue-700 dark:border-blue-500/20 dark:bg-blue-500/10 dark:text-blue-300">
            Only the <strong>Semester</strong> can be changed on an existing assignment. To move to a different course, delete this assignment and create a new one.
          </div>
        )}

        {/* ── 1. Course ── */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">
            Course
          </label>
          <Select
            options={[
              { value: "", label: "Select Course..." },
              ...availableCourses.map((c) => ({
                value: c.id,
                label: `${c.code} – ${c.title}`,
              })),
            ]}
            {...register("courseId")}
            error={errors.courseId?.message}
            disabled={mode === "view" || mode === "edit"}
          />
          {mode === "create" && availableCourses.length === 0 && (
            <p className="text-xs text-orange-500 mt-1 ml-1">
              All courses have already been assigned to a semester.
            </p>
          )}
          {selectedCourseId && mode === "create" && (
            <p className="text-xs text-gray-500 mt-1 ml-1">
              Faculty and department are filled automatically.
            </p>
          )}
        </div>

        {/* ── 2. Faculty (auto-filled, read-only) ── */}
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
            disabled={mode === "view" || mode === "edit" || !!selectedCourseId}
          />
        </div>

        {/* ── 3. Department (auto-filled, read-only) ── */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">
            Department
          </label>
          <Select
            options={[
              {
                value: "",
                label: selectedFacultyId
                  ? "Select Department..."
                  : "Auto-filled from course",
              },
              ...departments.map((d) => ({ value: d.id, label: d.name })),
            ]}
            {...register("departmentId")}
            error={errors.departmentId?.message}
            disabled={mode === "view" || mode === "edit" || !!selectedCourseId}
          />
        </div>

        {/* ── 4. Academic Year ── */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">
            Academic Year
          </label>
          <Select
            options={[
              { value: "", label: "Select Academic Year..." },
              ...academicYearOptions,
            ]}
            value={selectedAcademicYear}
            onChange={(e) => {
              setSelectedAcademicYear(e.target.value);
              // Clear the semester selection when the year changes
              setValue("semester", "", { shouldValidate: false });
            }}
            disabled={mode === "view" || mode === "edit"}
          />
          {academicYearOptions.length === 0 && mode !== "edit" && (
            <p className="text-xs text-orange-500 mt-1 ml-1">
              No academic years found. Create one in Academic Structure first.
            </p>
          )}
        </div>

        {/* ── 5. Semester (from structures for the chosen year) ── */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">
            Semester
          </label>
          <Select
            options={[
              {
                value: "",
                label:
                  !selectedAcademicYear && mode !== "edit"
                    ? "Select an academic year first..."
                    : "Select Semester...",
              },
              ...semesterOptions,
            ]}
            {...register("semester")}
            error={errors.semester?.message}
            disabled={
              mode === "view" ||
              (!selectedAcademicYear && mode !== "edit") ||
              semesterOptions.length === 0
            }
          />
          {selectedAcademicYear && semesterOptions.length === 0 && (
            <p className="text-xs text-orange-500 mt-1 ml-1">
              No semester terms found for this academic year.
            </p>
          )}
        </div>

        <div className="flex items-center justify-end gap-3 pt-6 mt-6">
          <Button
            type="button"
            variant="ghost"
            onClick={() => closeModal("courseAssign")}
          >
            {mode === "view" ? "Close" : "Cancel"}
          </Button>
          {mode !== "view" && (
            <Button type="submit" isLoading={isSubmitting}>
              {mode === "edit" ? "Save Changes" : "Assign Course"}
            </Button>
          )}
        </div>
      </form>
    </Modal>
  );
}
