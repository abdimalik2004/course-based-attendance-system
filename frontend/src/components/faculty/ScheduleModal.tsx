import { useEffect, useState } from "react";
import { useForm, Controller } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";
import { Modal } from "@/components/ui/Modal";
import { Button } from "@/components/ui/Button";
import { Select } from "@/components/ui/Select";
import { Input } from "@/components/ui/Input";
import { useFacultyStore } from "@/store/useFacultyStore";
import { cn } from "@/utils/cn";

const weekdays = ["sat", "sun", "mon", "tue", "wed", "thu", "fri"] as const;

const scheduleSchema = z.object({
  courseId: z.string().min(1, "Course is required"),
  weekdays: z.array(z.string()).min(1, "Select at least one weekday"),
  startTime: z.string().min(1, "Start time is required"),
  endTime: z.string().min(1, "End time is required"),
  gracePeriod: z.number().min(0, "Grace period must be positive"),
});

type ScheduleForm = z.infer<typeof scheduleSchema>;

// Strip seconds from "HH:MM:SS" → "HH:MM" for HTML time inputs
const toTimeInput = (t: string | undefined): string => {
  if (!t) return "";
  const parts = t.split(":");
  return parts.length >= 2 ? `${parts[0]}:${parts[1]}` : t;
};

export function ScheduleModal() {
  const { scheduleModal, closeModal, updateSchedule, courses } =
    useFacultyStore();

  const { isOpen, mode, record } = scheduleModal;
  const isViewMode = mode === "view";

  const [submitError, setSubmitError] = useState<string | null>(null);

  const {
    register,
    handleSubmit,
    control,
    reset,
    formState: { errors, isSubmitting },
  } = useForm<ScheduleForm>({
    resolver: zodResolver(scheduleSchema),
  });

  useEffect(() => {
    if (isOpen && record) {
      setSubmitError(null);
      reset({
        courseId: record.courseId,
        weekdays: record.weekdays,
        startTime: toTimeInput(record.startTime),
        endTime: toTimeInput(record.endTime),
        gracePeriod: record.gracePeriod,
      });
    }
  }, [isOpen, record, reset]);

  const onSubmit = async (data: ScheduleForm) => {
    setSubmitError(null);
    try {
      if (mode === "edit" && record) {
        await updateSchedule(record.id, data);
      }
      closeModal("schedule");
    } catch (error: any) {
      const msg =
        error?.response?.data?.error?.message ??
        error?.response?.data?.detail ??
        error?.message ??
        "Failed to save schedule. Please try again.";
      setSubmitError(msg);
    }
  };

  const courseOptions = [
    { value: "", label: "Select Course..." },
    ...courses.map((c) => ({ value: c.id, label: `${c.code} - ${c.title}` })),
  ];

  if (mode === "create") return null;

  return (
    <Modal
      isOpen={isOpen}
      onClose={() => closeModal("schedule")}
      title={isViewMode ? "View Schedule" : "Edit Schedule"}
    >
      <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="col-span-1 md:col-span-2 space-y-1">
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Course
            </label>
            <Select
              options={courseOptions}
              error={errors.courseId?.message}
              disabled={isViewMode}
              {...register("courseId")}
            />
          </div>

          <div className="space-y-1">
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Start Time
            </label>
            <Input
              type="time"
              error={errors.startTime?.message}
              disabled={isViewMode}
              {...register("startTime")}
            />
          </div>

          <div className="space-y-1">
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
              End Time
            </label>
            <Input
              type="time"
              error={errors.endTime?.message}
              disabled={isViewMode}
              {...register("endTime")}
            />
          </div>
        </div>

        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
            Weekdays
          </label>
          <Controller
            name="weekdays"
            control={control}
            render={({ field }) => (
              <div className="flex flex-wrap gap-2">
                {weekdays.map((day) => {
                  const isSelected = field.value?.includes(day);
                  return (
                    <button
                      key={day}
                      type="button"
                      disabled={isViewMode}
                      onClick={() => {
                        if (isViewMode) return;
                        const newValue = isSelected
                          ? field.value.filter((d) => d !== day)
                          : [...(field.value || []), day];
                        field.onChange(newValue);
                      }}
                      className={cn(
                        "px-4 py-2 rounded-full text-sm font-medium capitalize transition-all duration-300 border",
                        isSelected
                          ? "bg-purple-500/20 border-purple-500 text-purple-400 shadow-[0_0_10px_rgba(168,85,247,0.4)]"
                          : "bg-white/5 border-white/10 text-gray-400 hover:text-gray-200 hover:bg-white/10",
                        isViewMode &&
                          !isSelected &&
                          "opacity-50 cursor-not-allowed",
                        isViewMode && isSelected && "cursor-default",
                      )}
                    >
                      {day}
                    </button>
                  );
                })}
              </div>
            )}
          />
          {errors.weekdays && (
            <p className="text-sm text-red-500 mt-1">
              {errors.weekdays.message}
            </p>
          )}
        </div>

        <div className="w-48 space-y-1">
          <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Grace Period (Minutes)
          </label>
          <Input
            type="number"
            error={errors.gracePeriod?.message}
            disabled={isViewMode}
            {...register("gracePeriod", { valueAsNumber: true })}
          />
        </div>

        {submitError && (
          <div className="rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-400">
            {submitError}
          </div>
        )}

        <div className="flex justify-end gap-3 pt-6 border-t border-gray-200 dark:border-gray-800">
          <Button
            type="button"
            variant="secondary"
            onClick={() => closeModal("schedule")}
          >
            {isViewMode ? "Close" : "Cancel"}
          </Button>
          {!isViewMode && (
            <Button type="submit" isLoading={isSubmitting}>
              Save Changes
            </Button>
          )}
        </div>
      </form>
    </Modal>
  );
}
