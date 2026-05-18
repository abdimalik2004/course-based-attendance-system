import { Download, Printer, FileSpreadsheet } from 'lucide-react';
import { cn } from '@/utils/cn';

interface ExportButtonsProps {
  onExportPDF?: () => void;
  onExportCSV?: () => void;
  onPrint?: () => void;
  className?: string;
}

export function ExportButtons({ onExportPDF, onExportCSV, onPrint, className }: ExportButtonsProps) {
  return (
    <div className={cn("flex items-center gap-2", className)}>
      <button
        onClick={onExportPDF}
        className="flex items-center gap-2 px-3 py-1.5 text-sm font-medium text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors rounded-lg hover:bg-gray-100 dark:hover:bg-white/5 border border-transparent hover:border-gray-200 dark:hover:border-white/10"
        title="Export to PDF"
      >
        <Download size={16} />
        <span className="hidden sm:inline">Export PDF</span>
      </button>
      <button
        onClick={onExportCSV}
        className="flex items-center gap-2 px-3 py-1.5 text-sm font-medium text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors rounded-lg hover:bg-gray-100 dark:hover:bg-white/5 border border-transparent hover:border-gray-200 dark:hover:border-white/10"
        title="Export to CSV"
      >
        <FileSpreadsheet size={16} />
        <span className="hidden sm:inline">Export CSV</span>
      </button>
      <div className="w-px h-4 bg-gray-300 dark:bg-gray-700 mx-1" />
      <button
        onClick={onPrint}
        className="flex items-center gap-2 px-3 py-1.5 text-sm font-medium text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors rounded-lg hover:bg-gray-100 dark:hover:bg-white/5 border border-transparent hover:border-gray-200 dark:hover:border-white/10"
        title="Print"
      >
        <Printer size={16} />
        <span className="hidden sm:inline">Print</span>
      </button>
    </div>
  );
}
