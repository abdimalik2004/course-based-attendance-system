import { useEffect, useState } from 'react';
import { Modal } from '@/components/ui/Modal';
import { Button } from '@/components/ui/Button';
import { AlertTriangle } from 'lucide-react';

interface ConfirmDeleteModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => Promise<void> | void;
  title?: string;
  message?: string;
}

export function ConfirmDeleteModal({
  isOpen,
  onClose,
  onConfirm,
  title = "Delete Confirmation",
  message = "Are you sure you want to delete this record? This action cannot be undone."
}: ConfirmDeleteModalProps) {
  const [isDeleting, setIsDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Reset state whenever the modal opens or closes
  useEffect(() => {
    if (!isOpen) {
      setError(null);
      setIsDeleting(false);
    }
  }, [isOpen]);

  const handleConfirm = async () => {
    setIsDeleting(true);
    setError(null);
    try {
      await onConfirm();
      onClose();
    } catch (err: any) {
      const msg =
        err?.response?.data?.detail ??
        err?.response?.data?.error?.message ??
        (err instanceof Error ? err.message : null) ??
        'Delete failed. Please try again.';
      setError(msg);
    } finally {
      setIsDeleting(false);
    }
  };

  return (
    <Modal isOpen={isOpen} onClose={isDeleting ? undefined : onClose} title={title} className="max-w-sm">
      <div className="flex flex-col items-center text-center space-y-4 pt-4">
        <div className="h-12 w-12 rounded-full bg-red-100 dark:bg-red-500/10 flex items-center justify-center">
          <AlertTriangle className="h-6 w-6 text-red-600 dark:text-red-500" />
        </div>
        <p className="text-sm text-gray-600 dark:text-gray-300">
          {message}
        </p>

        {error && (
          <div className="w-full rounded-xl border border-rose-200 bg-rose-50 px-4 py-2.5 text-sm text-rose-700 dark:border-rose-500/20 dark:bg-rose-500/10 dark:text-rose-200 text-left">
            {error}
          </div>
        )}

        <div className="flex w-full gap-3 mt-6 pt-4">
          <Button
            variant="outline"
            className="w-full"
            onClick={onClose}
            disabled={isDeleting}
          >
            No, Cancel
          </Button>
          <Button
            className="w-full bg-red-600 hover:bg-red-700 text-white border-transparent"
            onClick={handleConfirm}
            isLoading={isDeleting}
            disabled={isDeleting}
          >
            Yes, Delete
          </Button>
        </div>
      </div>
    </Modal>
  );
}
