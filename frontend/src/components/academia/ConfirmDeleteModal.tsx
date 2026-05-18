import { Modal } from '@/components/ui/Modal';
import { Button } from '@/components/ui/Button';
import { AlertTriangle } from 'lucide-react';

interface ConfirmDeleteModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
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
  
  return (
    <Modal isOpen={isOpen} onClose={onClose} title={title} className="max-w-sm">
      <div className="flex flex-col items-center text-center space-y-4 pt-4">
        <div className="h-12 w-12 rounded-full bg-red-100 dark:bg-red-500/10 flex items-center justify-center">
          <AlertTriangle className="h-6 w-6 text-red-600 dark:text-red-500" />
        </div>
        <p className="text-sm text-gray-600 dark:text-gray-300">
          {message}
        </p>
        <div className="flex w-full gap-3 mt-6 pt-4">
          <Button variant="outline" className="w-full" onClick={onClose}>
            No, Cancel
          </Button>
          <Button 
            className="w-full bg-red-600 hover:bg-red-700 text-white border-transparent" 
            onClick={() => {
              onConfirm();
              onClose();
            }}
          >
            Yes, Delete
          </Button>
        </div>
      </div>
    </Modal>
  );
}
