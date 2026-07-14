import { useEffect } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";
import { Modal } from "@/components/ui/Modal";
import { Button } from "@/components/ui/Button";
import { Select } from "@/components/ui/Select";
import { useFacultyStore } from "@/store/useFacultyStore";

const assignSchema = z.object({
  courseId: z.string().min(1, "Course is required"),
  teacherId: z.string().min(1, "Teacher is required"),
  status: z.enum(["active", "inactive"]),
});

type AssignForm = z.infer<typeof assignSchema>;

export function AssignTeacherModal() {
  const {
    assignModal,
    closeModal,
    addAssignment,
    updateAssignment,
    courses,
    assignments,
    teachers,
  } = useFacultyStore();

  const { isOpen, mode, record } = assignModal;
  const isViewMode = mode === "view";

  const {
    register,
    handleSubmit,
    reset,
    formState: { errors, isSubmitting },
  } = useForm<AssignForm>({
    resolver: zodResolver(assignSchema),
    defaultValues: {
      courseId: "",
      teacherId: "",
      status: "active",
    },
  });

  useEffect(() => {
    if (isOpen && record) {
      reset({
        courseId: record.courseId,
        teacherId: record.teacherId,
        status: record.isPrimary ? "active" : "inactive",
      });
    } else if (isOpen && mode === "create") {
      reset({ courseId: "", teacherId: "", status: "active" });
    }
  }, [isOpen, mode, record, reset]);

  const onSubmit = async (data: AssignForm) => {
    try {
      if (mode === "create") {
        await addAssignment(data);
      } else if (mode === "edit" && record) {
        await updateAssignment(record.id, data);
      }
      closeModal("assign");
    } catch (error) {
      console.error("Failed to save assignment:", error);
    }
  };

  // Courses already assigned — exclude the one being edited so it stays selectable
  const assignedCourseIds = new Set(
    assignments
      .filter((a) => mode !== "edit" || a.id !== record?.id)
      .map((a) => a.courseId),
  );

  const courseOptions = courses.map((c) => {
    const alreadyAssigned = assignedCourseIds.has(c.id);
    return {
      value: c.id,
      label: alreadyAssigned
        ? `${c.code} - ${c.title} (already assigned)`
        : `${c.code} - ${c.title}`,
      disabled: alreadyAssigned,
    };
  });

  const teacherOptions = teachers.map((t) => ({
    value: t.id,
    label: t.fullName,
  }));

  return (
    <Modal
      isOpen={isOpen}
      onClose={() => closeModal("assign")}
      title={
        mode === "create"
          ? "Assign Teacher"
          : mode === "edit"
            ? "Edit Assignment"
            : "View Assignment"
      }
    >
      <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
        <div className="space-y-4">
          <div className="space-y-1">
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Course
            </label>
            <Select
              placeholder="Select Course"
              options={courseOptions}
              error={errors.courseId?.message}
              disabled={isViewMode}
              {...register("courseId")}
            />
          </div>

          <div className="space-y-1">
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Teacher
            </label>
            <Select
              placeholder="Select Teacher"
              options={teacherOptions}
              error={errors.teacherId?.message}
              disabled={isViewMode}
              {...register("teacherId")}
            />
          </div>

          <div className="space-y-1">
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Role
            </label>
            <Select
              options={[
                { value: "active", label: "Primary" },
                { value: "inactive", label: "Secondary" },
              ]}
              error={errors.status?.message}
              disabled={isViewMode}
              {...register("status")}
            />
          </div>
        </div>

        <div className="flex justify-end gap-3 pt-6 border-t border-gray-200 dark:border-gray-800">
          <Button
            type="button"
            variant="secondary"
            onClick={() => closeModal("assign")}
          >
            {isViewMode ? "Close" : "Cancel"}
          </Button>
          {!isViewMode && (
            <Button type="submit" isLoading={isSubmitting}>
              {mode === "create" ? "Assign Teacher" : "Save Changes"}
            </Button>
          )}
        </div>
      </form>
    </Modal>
  );
}
