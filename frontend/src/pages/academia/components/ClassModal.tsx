import { useEffect, useState } from "react";
import { useForm, useWatch } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { Modal } from "@/components/ui/Modal";
import { Input } from "@/components/ui/Input";
import { Select } from "@/components/ui/Select";
import { Button } from "@/components/ui/Button";
import { useAcademiaStore } from "@/store/useAcademiaStore";

const classSchema = z.object({
  facultyId: z.string().min(1, "Faculty is required"),
  departmentId: z.string().min(1, "Department is required"),
  year: z.coerce
    .number()
    .min(1, "Year must be at least 1")
    .max(10, "Year must not exceed 10"),
});

type ClassFormData = z.infer<typeof classSchema>;

export function ClassModal() {
  const {
    classModal,
    closeModal,
    addClass,
    updateClass,
    faculties,
    departments,
  } = useAcademiaStore();
  const [isSubmitting, setIsSubmitting] = useState(false);

  const { isOpen, mode, record } = classModal;
  const isViewMode = mode === "view";

  const {
    register,
    handleSubmit,
    formState: { errors },
    reset,
    control,
    setValue,
  } = useForm<ClassFormData>({
    resolver: zodResolver(classSchema),
  });

  const selectedFacultyId = useWatch({ control, name: "facultyId" });

  // Dynamic filter dependency
  const availableDepartments = departments.filter(
    (d) => d.facultyId === selectedFacultyId,
  );

  useEffect(() => {
    if (isOpen) {
      if (record) {
        reset({
          facultyId: record.facultyId,
          departmentId: record.departmentId,
          year: record.year,
        });
      } else {
        reset({ facultyId: "", departmentId: "", year: 1 });
      }
    }
  }, [isOpen, record, reset]);

  // When faculty changes, invalidate the department
  useEffect(() => {
    if (isOpen && !record && selectedFacultyId) {
      setValue("departmentId", "");
    }
  }, [selectedFacultyId, isOpen, record, setValue]);

  const onSubmit = async (data: ClassFormData) => {
    if (isViewMode) return;
    setIsSubmitting(true);
    try {
      if (mode === "edit" && record) {
        await updateClass(record.id, data);
      } else {
        await addClass(data);
      }
      closeModal("class");
    } catch (error) {
      console.error(error);
    } finally {
      setIsSubmitting(false);
    }
  };

  const titles = {
    create: "Create Class",
    edit: "Edit Class",
    view: "Class Details",
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={() => closeModal("class")}
      title={titles[mode]}
      className="md:max-w-md"
    >
      <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
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
            disabled={isViewMode}
          />
        </div>
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
                  : "Select Faculty first...",
              },
              ...availableDepartments.map((d) => ({
                value: d.id,
                label: d.name,
              })),
            ]}
            {...register("departmentId")}
            error={errors.departmentId?.message}
            disabled={isViewMode || !selectedFacultyId}
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">
            Year
          </label>
          <Input
            type="number"
            placeholder="e.g. 1"
            {...register("year")}
            error={errors.year?.message}
            disabled={isViewMode}
            className={
              isViewMode ? "bg-gray-50 dark:bg-dark-bg text-gray-500" : ""
            }
          />
        </div>

        <div className="flex items-center justify-end gap-3 pt-6 mt-6">
          <Button
            type="button"
            variant="ghost"
            onClick={() => closeModal("class")}
          >
            {isViewMode ? "Close" : "Cancel"}
          </Button>
          {!isViewMode && (
            <Button type="submit" isLoading={isSubmitting}>
              {mode === "edit" ? "Save Changes" : "Create Class"}
            </Button>
          )}
        </div>
      </form>
    </Modal>
  );
}
