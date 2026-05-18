import { Eye } from 'lucide-react';
import { Button } from './Button';

interface ViewButtonProps {
  onClick: () => void;
  tooltip?: string;
  className?: string;
}

export function ViewButton({ onClick, tooltip = 'View', className }: ViewButtonProps) {
  return (
    <Button
      variant="ghost"
      size="sm"
      className={`h-8 w-8 p-0 text-blue-500 hover:text-blue-600 hover:bg-blue-50 dark:text-blue-400 dark:hover:text-blue-300 dark:hover:bg-blue-500/10 transition-colors ${className || ''}`}
      onClick={onClick}
      title={tooltip}
    >
      <Eye size={16} />
    </Button>
  );
}
