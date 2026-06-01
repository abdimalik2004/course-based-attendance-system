import { useEffect, useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
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
  const [filteredDepartments, setFilteredDepartments] = useState<
    { value: string; label: string }[]
  >([]);

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

  // Dynamic Department filtering
  useEffect(() => {
    if (selectedFacultyId) {
      const filtered = departments
        .filter((d) => d.facultyId === selectedFacultyId)
        .map((d) => ({ value: d.id, label: d.name }));

      setFilteredDepartments(filtered);
      if (mode === "create") {
        setValue("departmentId", "");
      }
    } else {
      setFilteredDepartments([]);
    }
  }, [selectedFacultyId, departments, setValue]);

  const onSubmit = async (data: FormData) => {
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
      console.error(error);
      const msg =
        error?.response?.data?.detail ??
        error?.response?.data?.error?.message ??
        (error instanceof Error ? error.message : null) ??
        'Failed to assign class';
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
        {submitError && (
          <div className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700 dark:border-rose-500/20 dark:bg-rose-500/10 dark:text-rose-200">
            {submitError}
          </div>
        )}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">
            Class Name
          </label>
          <Select
            options={[
              { value: "", label: "Select Class..." },
              ...classes.map((c) => ({ value: c.id, label: c.name })),
            ]}
            {...register("classId")}
            error={errors.classId?.message}
            disabled={mode === "view"}
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">
            Course Title
          </label>
          <Select
            options={[
              { value: "", label: "Select Course..." },
              ...courses.map((c) => {
                const alreadyAssigned = classAssignments.some(
                  (a) =>
                    a.classId === selectedClassId &&
                    a.courseId === c.id &&
                    !(mode === 'edit' && record && a.id === record.id),
                );
                return {
                  value: c.id,
                  label: alreadyAssigned ? `${c.title} — already assigned` : c.title,
                  disabled: alreadyAssigned,
                };
              }),
            ]}
            {...register("courseId")}
            error={errors.courseId?.message}
            disabled={mode === "view"}
          />
        </div>

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
            disabled={mode === "view"}
          />
        </div>

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
              mode === "view"
            }
          />
          {!selectedFacultyId && (
            <p className="text-xs text-gray-500 mt-1 ml-1">
              Select a faculty to filter departments.
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
            <Button type="submit" isLoading={isSubmitting}>
              {mode === "edit" ? "Save Changes" : "Assign Class"}
            </Button>
          )}
        </div>
      </form>
    </Modal>
  );
}
