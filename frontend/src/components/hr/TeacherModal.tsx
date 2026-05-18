import { useEffect, useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { Modal } from "@/components/ui/Modal";
import { Input } from "@/components/ui/Input";
import { Select } from "@/components/ui/Select";
import { Button } from "@/components/ui/Button";
import { useHrStore } from "@/store/useHrStore";
import type { Teacher } from "@/services/hrService";

const teacherSchema = z.object({
  fullName: z.string().min(2, "Full name is required"),
  facultyId: z.string().min(1, "Faculty is required"),
  departmentId: z.string().min(1, "Department is required"),
  role: z.string().min(1, "Role is required"),
  status: z.enum(["Active", "Inactive", "On Leave"]).optional(),
});

type TeacherFormData = z.infer<typeof teacherSchema>;

export interface TeacherModalProps {
  isOpen: boolean;
  onClose: () => void;
  mode: "create" | "edit" | "view";
  record: Teacher | null;
}

export function TeacherModal({
  isOpen,
  onClose,
  mode,
  record,
}: TeacherModalProps) {
  const {
    faculties,
    departments,
    fetchFaculties,
    fetchDepartments,
    addTeacher,
    updateTeacher,
  } = useHrStore();
  const [isSubmitting, setIsSubmitting] = useState(false);

  const isViewMode = mode === "view";

  const {
    register,
    handleSubmit,
    formState: { errors },
    reset,
    watch,
  } = useForm<TeacherFormData>({
    resolver: zodResolver(teacherSchema),
  });

  const selectedFacultyId = watch("facultyId");
  const facultyOptions = faculties.map((faculty) => ({
    value: faculty.id,
    label: faculty.name,
  }));
  const departmentOptions = departments
    .filter((department) => department.facultyId === selectedFacultyId)
    .map((department) => ({ value: department.id, label: department.name }));

  useEffect(() => {
    if (isOpen) {
      fetchFaculties();
      fetchDepartments();
      if (record && (mode === "edit" || mode === "view")) {
        reset({
          fullName: record.fullName,
          facultyId: record.facultyId,
          departmentId: record.departmentId,
          role: record.role,
          status: record.status as any,
        });
      } else {
        reset({
          fullName: "",
          facultyId: "",
          departmentId: "",
          role: "",
          status: "Active",
        });
      }
    }
  }, [isOpen, record, mode, reset, fetchFaculties, fetchDepartments]);

  const onSubmit = async (data: TeacherFormData) => {
    if (isViewMode) return;
    setIsSubmitting(true);
    try {
      if (mode === "edit" && record) {
        await updateTeacher(record.id, data);
      } else {
        await addTeacher(data);
      }
      onClose();
    } catch (error) {
      console.error(error);
    } finally {
      setIsSubmitting(false);
    }
  };

  const titles = {
    create: "Add New Teacher",
    edit: "Edit Teacher",
    view: "Teacher Details",
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title={titles[mode]}
      className="md:max-w-lg"
    >
      <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">
            Full Name
          </label>
          <Input
            placeholder="e.g. Dr. Ahmed Ali"
            {...register("fullName")}
            error={errors.fullName?.message}
            disabled={isViewMode}
            className={
              isViewMode ? "bg-gray-50 dark:bg-dark-bg text-gray-500" : ""
            }
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">
            Faculty
          </label>
          <Select
            {...register("facultyId")}
            options={facultyOptions}
            placeholder="Select Faculty"
            error={errors.facultyId?.message}
            disabled={isViewMode}
            className={
              isViewMode
                ? "bg-gray-50 dark:bg-dark-bg text-gray-500 cursor-not-allowed"
                : ""
            }
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">
            Department
          </label>
          <Select
            {...register("departmentId")}
            options={departmentOptions}
            placeholder={
              selectedFacultyId ? "Select Department" : "Select Faculty First"
            }
            error={errors.departmentId?.message}
            disabled={!selectedFacultyId || isViewMode}
            className={
              !selectedFacultyId || isViewMode
                ? "bg-gray-50 dark:bg-dark-bg text-gray-500 cursor-not-allowed opacity-70"
                : ""
            }
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">
            Role
          </label>
          <select
            {...register("role")}
            disabled={isViewMode}
            className={`w-full rounded-xl border bg-white dark:bg-white/5 px-4 py-2.5 text-sm outline-none transition-all
              ${isViewMode ? "bg-gray-50 dark:bg-dark-bg text-gray-500 cursor-not-allowed" : ""}
              ${
                errors.role
                  ? "border-red-500 focus:border-red-500 focus:ring-4 focus:ring-red-500/10"
                  : "border-gray-300 dark:border-white/10 focus:border-primary focus:ring-4 focus:ring-primary/10"
              } text-gray-900 dark:text-white appearance-none`}
          >
            <option
              value=""
              disabled
              className="bg-white dark:bg-dark-bg text-gray-900 dark:text-white"
            >
              Select Role
            </option>
            <option
              value="Professor"
              className="bg-white dark:bg-dark-bg text-gray-900 dark:text-white"
            >
              Professor
            </option>
            <option
              value="Associate Professor"
              className="bg-white dark:bg-dark-bg text-gray-900 dark:text-white"
            >
              Associate Professor
            </option>
            <option
              value="Assistant Professor"
              className="bg-white dark:bg-dark-bg text-gray-900 dark:text-white"
            >
              Assistant Professor
            </option>
            <option
              value="Lecturer"
              className="bg-white dark:bg-dark-bg text-gray-900 dark:text-white"
            >
              Lecturer
            </option>
          </select>
          {errors.role && (
            <p className="text-xs text-red-500 mt-1 ml-1">
              {errors.role.message}
            </p>
          )}
        </div>

        {mode === "edit" && (
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">
              Status
            </label>
            <select
              {...register("status")}
              className={`w-full rounded-xl border bg-white dark:bg-white/5 px-4 py-2.5 text-sm outline-none transition-all
                ${
                  errors.status
                    ? "border-red-500 focus:border-red-500 focus:ring-4 focus:ring-red-500/10"
                    : "border-gray-300 dark:border-white/10 focus:border-primary focus:ring-4 focus:ring-primary/10"
                } text-gray-900 dark:text-white appearance-none`}
            >
              <option
                value="Active"
                className="bg-white dark:bg-dark-bg text-gray-900 dark:text-white"
              >
                Active
              </option>
              <option
                value="Inactive"
                className="bg-white dark:bg-dark-bg text-gray-900 dark:text-white"
              >
                Inactive
              </option>
              <option
                value="On Leave"
                className="bg-white dark:bg-dark-bg text-gray-900 dark:text-white"
              >
                On Leave
              </option>
            </select>
            {errors.status && (
              <p className="text-xs text-red-500 mt-1 ml-1">
                {errors.status.message}
              </p>
            )}
          </div>
        )}

        <div className="flex items-center justify-end gap-3 pt-6 mt-6">
          <Button type="button" variant="ghost" onClick={onClose}>
            {isViewMode ? "Close" : "Cancel"}
          </Button>
          {!isViewMode && (
            <Button type="submit" isLoading={isSubmitting}>
              {mode === "edit" ? "Save Changes" : "Save"}
            </Button>
          )}
        </div>
      </form>
    </Modal>
  );
}
