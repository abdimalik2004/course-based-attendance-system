import { useEffect, useState } from "react";
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

export function CourseAssignModal() {
  const {
    courseAssignModal,
    closeModal,
    addCourseAssignment,
    updateCourseAssignment,
    courses,
    faculties,
    departments,
    courseAssignments,
  } = useAcademiaStore();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
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
  const [semesterOptions, setSemesterOptions] = useState<
    { value: string; label: string }[]
  >([]);

  useEffect(() => {
    if (selectedCourseId) {
      const selectedCourse = courses.find(
        (course) => course.id === selectedCourseId,
      );
      if (selectedCourse) {
        setValue("facultyId", selectedCourse.facultyId, {
          shouldValidate: true,
        });
        setValue("departmentId", selectedCourse.departmentId, {
          shouldValidate: true,
        });
      }
    }
  }, [selectedCourseId, courses, setValue]);

  useEffect(() => {
    if (isOpen) {
      setSubmitError(null);
      if (record && mode !== "create") {
        reset({
          courseId: record.courseId,
          facultyId: record.facultyId,
          departmentId: record.departmentId,
          semester: record.semester.toString(),
        });
      } else {
        reset({ courseId: "", facultyId: "", departmentId: "", semester: "" });
      }
    }
  }, [isOpen, mode, record, reset]);

  // Dynamic Semester calculation
  useEffect(() => {
    if (selectedFacultyId) {
      const faculty = faculties.find((f) => f.id === selectedFacultyId);
      if (faculty && faculty.years) {
        const totalSemesters = faculty.years * 2;

        // Build the set of semesters already taken for this course+faculty combo
        const assignedSemesters = new Set(
          courseAssignments
            .filter((a) => {
              if (a.courseId !== selectedCourseId || a.facultyId !== selectedFacultyId) return false;
              // In edit mode, don't count the current record as "already assigned"
              if (mode === 'edit' && record && a.id === record.id) return false;
              return true;
            })
            .map((a) => String(a.semester)),
        );

        const options = Array.from({ length: totalSemesters }, (_, i) => {
          const val = (i + 1).toString();
          const taken = assignedSemesters.has(val);
          return {
            value: val,
            label: taken ? `Semester ${i + 1} — already assigned` : `Semester ${i + 1}`,
            disabled: taken,
          };
        });
        setSemesterOptions(options);
      } else {
        setSemesterOptions([]);
      }
      // If we are in create mode, reset semester so a stale value isn't kept
      if (mode === "create") {
        setValue("semester", "");
      }
    } else {
      setSemesterOptions([]);
    }
  }, [selectedFacultyId, selectedCourseId, faculties, courseAssignments, mode, record, setValue]);

  const onSubmit = async (data: FormData) => {
    setIsSubmitting(true);
    setSubmitError(null);
    try {
      if (mode === "edit" && record) {
        await updateCourseAssignment(record.id, {
          courseId: data.courseId,
          facultyId: data.facultyId,
          departmentId: data.departmentId,
          semester: parseInt(data.semester, 10),
        });
      } else {
        await addCourseAssignment({
          courseId: data.courseId,
          facultyId: data.facultyId,
          departmentId: data.departmentId,
          semester: parseInt(data.semester, 10),
        });
      }
      closeModal("courseAssign");
    } catch (error: any) {
      console.error(error);
      const msg =
        error?.response?.data?.detail ??
        error?.response?.data?.error?.message ??
        (error instanceof Error ? error.message : null) ??
        'Failed to assign course';
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
        {submitError && (
          <div className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700 dark:border-rose-500/20 dark:bg-rose-500/10 dark:text-rose-200">
            {submitError}
          </div>
        )}
        {/* In edit mode show a hint: only semester can change */}
        {mode === "edit" && (
          <div className="rounded-xl border border-blue-200 bg-blue-50 px-4 py-2.5 text-sm text-blue-700 dark:border-blue-500/20 dark:bg-blue-500/10 dark:text-blue-300">
            Only the <strong>Semester</strong> can be changed on an existing assignment. To move to a different course, delete this assignment and create a new one.
          </div>
        )}

        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">
            Course Code
          </label>
          <Select
            options={[
              { value: "", label: "Select Course..." },
              ...courses.map((c) => ({
                value: c.id,
                label: `${c.code} - ${c.title}`,
              })),
            ]}
            {...register("courseId")}
            error={errors.courseId?.message}
            disabled={mode === "view" || mode === "edit"}
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">
            Faculty Name
          </label>
          <Select
            options={[
              { value: "", label: "Select Faculty..." },
              ...faculties.map((f) => ({ value: f.id, label: f.name })),
            ]}
            {...register("facultyId")}
            error={errors.facultyId?.message}
            disabled={mode === "view" || mode === "edit"}
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">
            Department
          </label>
          <Select
            options={[
              { value: "", label: "Department is selected by course" },
              ...departments.map((d) => ({ value: d.id, label: d.name })),
            ]}
            {...register("departmentId")}
            error={errors.departmentId?.message}
            disabled={mode === "view" || mode === "edit" || !!selectedCourseId}
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">
            Semester
          </label>
          <Select
            options={[
              { value: "", label: "Select Semester..." },
              ...semesterOptions,
            ]}
            {...register("semester")}
            error={errors.semester?.message}
            disabled={
              !selectedFacultyId ||
              semesterOptions.length === 0 ||
              mode === "view"
            }
          />
          {!selectedFacultyId && mode !== "edit" && (
            <p className="text-xs text-gray-500 mt-1 ml-1">
              Select a course to populate faculty and department.
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
