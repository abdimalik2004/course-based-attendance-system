import { ReactNode } from 'react';
import { Modal } from './Modal';
import { Button } from './Button';

export interface ViewModalField {
  label: string;
  value: ReactNode;
}

export interface ViewModalProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  data: ViewModalField[] | null;
}

export function ViewModal({ isOpen, onClose, title, data }: ViewModalProps) {
  return (
    <Modal isOpen={isOpen} onClose={onClose} title={title} className="max-w-2xl">
      <div className="space-y-6">
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 bg-gray-50/50 dark:bg-white/5 p-6 rounded-xl border border-gray-100 dark:border-white/10 max-h-[60vh] overflow-y-auto custom-scrollbar">
          {data?.map((field, index) => (
            <div key={index} className="space-y-1.5">
              <span className="text-sm font-medium text-gray-500 dark:text-gray-400">
                {field.label}
              </span>
              <div className="text-base font-semibold text-gray-900 dark:text-white break-words">
                {field.value === undefined || field.value === null || field.value === '' ? '-' : field.value}
              </div>
            </div>
          ))}
        </div>
        
        <div className="flex justify-end pt-4 border-t border-gray-100 dark:border-white/10">
          <Button 
            onClick={onClose}
            className="bg-blue-600 hover:bg-blue-700 text-white shadow-lg shadow-blue-500/20 px-6"
          >
            Close
          </Button>
        </div>
      </div>
    </Modal>
  );
}
