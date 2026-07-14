import { useEffect, useState } from "react";
import { useForm, useWatch } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { Link2, Link2Off } from "lucide-react";
import { Modal } from "@/components/ui/Modal";
import { Input } from "@/components/ui/Input";
import { Select } from "@/components/ui/Select";
import { Button } from "@/components/ui/Button";
import { useHrStore } from "@/store/useHrStore";
import type { Teacher } from "@/services/hrService";
import { LinkUserModal } from "./LinkUserModal";

const teacherSchema = z.object({
  fullName: z.string().min(2, "Full name is required"),
  facultyId: z.string().min(1, "Faculty is required"),
  departmentId: z.string().min(1, "Department is required"),
  role: z.string().min(1, "Role is required"),
  status: z.enum(["Active", "Inactive", "On Leave"]).optional(),
  phone: z.string().max(30).optional().or(z.literal("")),
  email: z.string().email("Invalid email address").optional().or(z.literal("")),
  hireDate: z.string().optional().or(z.literal("")),
});

type TeacherFormData = z.infer<typeof teacherSchema>;

export interface TeacherModalProps {
  isOpen: boolean;
  onClose: () => void;
  mode: "create" | "edit" | "view";
  record: Teacher | null;
  onSave?: (updated: Teacher) => void;
}

export function TeacherModal({
  isOpen,
  onClose,
  mode,
  record,
  onSave,
}: TeacherModalProps) {
  const {
    faculties,
    departments,
    addTeacher,
    updateTeacher,
    linkUser,
  } = useHrStore();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [linkModalOpen, setLinkModalOpen] = useState(false);

  const isViewMode = mode === "view";

  const {
    register,
    handleSubmit,
    formState: { errors },
    reset,
    control,
    setValue,
  } = useForm<TeacherFormData>({
    resolver: zodResolver(teacherSchema),
  });

  const selectedFacultyId = useWatch({ control, name: "facultyId" });
  const facultyOptions = faculties.map((faculty) => ({
    value: faculty.id,
    label: faculty.name,
  }));
  const departmentOptions = departments
    .filter((department) => department.facultyId === selectedFacultyId)
    .map((department) => ({ value: department.id, label: department.name }));

  useEffect(() => {
    if (isOpen) {
      setSubmitError(null);
      if (record && (mode === "edit" || mode === "view")) {
        reset({
          fullName: record.fullName,
          facultyId: record.facultyId,
          departmentId: record.departmentId,
          role: record.role,
          status: record.status as any,
          phone: record.phone ?? "",
          email: record.email ?? "",
          hireDate: record.hireDate ?? "",
        });
      } else {
        reset({
          fullName: "",
          facultyId: "",
          departmentId: "",
          role: "",
          status: "Active",
          phone: "",
          email: "",
          hireDate: "",
        });
      }
    }
  }, [isOpen, record, mode, reset]);

  // Reset department when faculty changes (create mode only)
  useEffect(() => {
    if (isOpen && !record && selectedFacultyId) {
      setValue("departmentId", "");
    }
  }, [selectedFacultyId, isOpen, record, setValue]);

  const onSubmit = async (data: TeacherFormData) => {
    if (isViewMode) return;
    setIsSubmitting(true);
    setSubmitError(null);
    try {
      if (mode === "edit" && record) {
        const updated = await updateTeacher(record.id, {
          ...data,
          phone: data.phone || null,
          email: data.email || null,
          hireDate: data.hireDate || null,
        });
        onSave?.(updated);
      } else {
        await addTeacher({
          ...data,
          phone: data.phone || null,
          email: data.email || null,
          hireDate: data.hireDate || null,
        });
      }
      onClose();
    } catch (error: any) {
      const msg =
        error?.response?.data?.detail ??
        error?.response?.data?.error?.message ??
        (error instanceof Error ? error.message : null) ??
        (mode === "edit" ? "Failed to update teacher" : "Failed to create teacher");
      setSubmitError(msg);
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
        {submitError && (
          <div className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700 dark:border-rose-500/20 dark:bg-rose-500/10 dark:text-rose-200">
            {submitError}
          </div>
        )}
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

        {/* Contact Information */}
        <div className="rounded-xl border border-gray-200 dark:border-white/10 bg-gray-50/30 dark:bg-white/[0.02] px-4 py-3 space-y-3">
          <p className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide">
            Contact Information
          </p>

          {/* Phone + Email side by side */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">
                Phone
              </label>
              <Input
                type="tel"
                placeholder="e.g. +252 61 234 5678"
                {...register("phone")}
                error={errors.phone?.message}
                disabled={isViewMode}
                className={isViewMode ? "bg-gray-50 dark:bg-dark-bg text-gray-500" : ""}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">
                Email
              </label>
              <Input
                type="email"
                placeholder="e.g. teacher@university.edu"
                {...register("email")}
                error={errors.email?.message}
                disabled={isViewMode}
                className={isViewMode ? "bg-gray-50 dark:bg-dark-bg text-gray-500" : ""}
              />
            </div>
          </div>

          {/* Hire Date */}
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">
              Hire Date
            </label>
            <Input
              type="date"
              {...register("hireDate")}
              error={errors.hireDate?.message}
              disabled={isViewMode}
              className={`text-gray-900 dark:text-white dark:[color-scheme:dark]${isViewMode ? " bg-gray-50 dark:bg-dark-bg text-gray-500" : ""}`}
            />
          </div>
        </div>

        {/* Login Account section — shown in view and edit modes */}
        {(mode === "view" || mode === "edit") && record && (
          <div className="rounded-xl border border-gray-200 dark:border-white/10 bg-gray-50/50 dark:bg-white/5 px-4 py-3">
            <p className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-2 uppercase tracking-wide">
              Login Account
            </p>
            {record.linkedUsername ? (
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="flex h-7 w-7 items-center justify-center rounded-full bg-primary/10 text-primary text-xs font-semibold">
                    {record.linkedUsername[0].toUpperCase()}
                  </span>
                  <span className="text-sm font-medium text-gray-900 dark:text-white">
                    @{record.linkedUsername}
                  </span>
                </div>
                {mode === "edit" && (
                  <button
                    type="button"
                    onClick={() => setLinkModalOpen(true)}
                    className="flex items-center gap-1.5 text-xs text-primary hover:text-primary/80 transition-colors font-medium"
                  >
                    <Link2 size={13} />
                    Change
                  </button>
                )}
              </div>
            ) : (
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-sm text-gray-400 dark:text-gray-500">
                  <Link2Off size={14} />
                  <span>No account linked</span>
                </div>
                {mode === "edit" && (
                  <button
                    type="button"
                    onClick={() => setLinkModalOpen(true)}
                    className="flex items-center gap-1.5 text-xs text-primary hover:text-primary/80 transition-colors font-medium"
                  >
                    <Link2 size={13} />
                    Link Account
                  </button>
                )}
              </div>
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

      {record && (
        <LinkUserModal
          isOpen={linkModalOpen}
          onClose={() => setLinkModalOpen(false)}
          teacherName={record.fullName}
          currentLinkedUsername={record.linkedUsername}
          onLink={async (userId) => {
            await linkUser(record.id, userId);
          }}
          onUnlink={async () => {
            await linkUser(record.id, null);
          }}
        />
      )}
    </Modal>
  );
}
