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
  } = useAcademiaStore();
  const [isSubmitting, setIsSubmitting] = useState(false);
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
        const options = Array.from({ length: totalSemesters }, (_, i) => ({
          value: (i + 1).toString(),
          label: `Semester ${i + 1}`,
        }));
        setSemesterOptions(options);
      } else {
        setSemesterOptions([]);
      }
      // If we are in edit/view mode and just mounted, dont reset semester yet to preserve the record
      if (mode === "create") {
        setValue("semester", "");
      }
    } else {
      setSemesterOptions([]);
    }
  }, [selectedFacultyId, faculties, setValue]);

  const onSubmit = async (data: FormData) => {
    setIsSubmitting(true);
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
    } catch (error) {
      console.error(error);
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
            disabled={mode === "view"}
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
            disabled={mode === "view"}
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
            disabled={mode === "view" || !!selectedCourseId}
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
          {!selectedFacultyId && (
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
