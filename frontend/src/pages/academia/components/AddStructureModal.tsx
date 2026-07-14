import { useEffect, useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { Modal } from "@/components/ui/Modal";
import { Input } from "@/components/ui/Input";
import { Button } from "@/components/ui/Button";
import { useAcademiaStore } from "@/store/useAcademiaStore";

const structureSchema = z.object({
  academicYear: z.string().min(4, "Year must be e.g. 2026-2027"),
  term: z.string().min(3, "Term name required"),
  startDate: z.string().min(1, "Start Date is required"),
  endDate: z.string().min(1, "End Date is required"),
});

type StructureFormData = z.infer<typeof structureSchema>;

/** Convert an ISO date string to "YYYY-MM-DD" for <input type="date"> */
function toDateInput(value: string | undefined): string {
  if (!value) return "";
  if (/^\d{4}-\d{2}-\d{2}$/.test(value)) return value;
  const d = new Date(value);
  if (isNaN(d.getTime())) return "";
  return d.toISOString().slice(0, 10);
}

export function AddStructureModal() {
  const { structureModal, closeModal, addStructure, updateStructure } = useAcademiaStore();
  const isOpen = structureModal?.isOpen || false;
  const mode = structureModal?.mode || "create";
  const record = structureModal?.record;
  const isEdit = mode === "edit";

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const {
    register,
    handleSubmit,
    formState: { errors },
    reset,
  } = useForm<StructureFormData>({
    resolver: zodResolver(structureSchema),
  });

  // Pre-fill form when editing, clear when creating
  useEffect(() => {
    if (!isOpen) {
      setSubmitError(null);
      return;
    }
    if (isEdit && record) {
      reset({
        academicYear: record.academicYear ?? "",
        term: record.term ?? "",
        startDate: toDateInput(record.startDate),
        endDate: toDateInput(record.endDate),
      });
    } else {
      reset({ academicYear: "", term: "", startDate: "", endDate: "" });
    }
  }, [isOpen, isEdit, record, reset]);

  const onSubmit = async (data: StructureFormData) => {
    setIsSubmitting(true);
    setSubmitError(null);
    try {
      if (isEdit && record) {
        await updateStructure(record.id, data);
      } else {
        await addStructure(data);
      }
      closeModal("structure");
    } catch (error: any) {
      const msg =
        error?.response?.data?.detail ??
        error?.response?.data?.error?.message ??
        (error instanceof Error ? error.message : null) ??
        (isEdit ? "Failed to update term" : "Failed to create term");
      setSubmitError(msg);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={() => closeModal("structure")}
      title={isEdit ? "Edit Academic Term" : "Create Academic Term"}
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
            Academic Year
          </label>
          <Input
            placeholder="e.g. 2026-2027"
            {...register("academicYear")}
            error={errors.academicYear?.message}
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">
            Term Name
          </label>
          <Input
            placeholder="e.g. Semester 1"
            {...register("term")}
            error={errors.term?.message}
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">
              Start Date
            </label>
            <Input
              type="date"
              className="dark:[color-scheme:dark]"
              {...register("startDate")}
              error={errors.startDate?.message}
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">
              End Date
            </label>
            <Input
              type="date"
              className="dark:[color-scheme:dark]"
              {...register("endDate")}
              error={errors.endDate?.message}
            />
          </div>
        </div>

        <p className="text-xs text-gray-500 dark:text-gray-400 px-1">
          Status is set automatically: <strong>Active</strong> while today falls between the dates,{" "}
          <strong>Inactive</strong> once the end date passes, <strong>Draft</strong> before the start date.
        </p>

        <div className="flex items-center justify-end gap-3 pt-4 mt-2">
          <Button type="button" variant="ghost" onClick={() => closeModal("structure")}>
            Cancel
          </Button>
          <Button type="submit" isLoading={isSubmitting}>
            {isEdit ? "Save Changes" : "Create Term"}
          </Button>
        </div>
      </form>
    </Modal>
  );
}
