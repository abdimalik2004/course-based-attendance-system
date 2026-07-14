import { ChevronLeft, ChevronRight } from "lucide-react";
import { cn } from "@/utils/cn";

interface PaginationProps {
  page: number;
  totalPages: number;
  totalItems: number;
  pageSize: number;
  onPageChange: (page: number) => void;
}

export function Pagination({
  page,
  totalPages,
  totalItems,
  pageSize,
  onPageChange,
}: PaginationProps) {
  if (totalItems === 0) return null;

  const from = (page - 1) * pageSize + 1;
  const to = Math.min(page * pageSize, totalItems);

  /** Build a compact list of page numbers with ellipsis gaps */
  const getPageNumbers = (): (number | "...")[] => {
    if (totalPages <= 7) {
      return Array.from({ length: totalPages }, (_, i) => i + 1);
    }
    const pages: (number | "...")[] = [1];
    if (page > 3) pages.push("...");
    for (
      let p = Math.max(2, page - 1);
      p <= Math.min(totalPages - 1, page + 1);
      p++
    ) {
      pages.push(p);
    }
    if (page < totalPages - 2) pages.push("...");
    pages.push(totalPages);
    return pages;
  };

  const btnBase =
    "inline-flex items-center justify-center h-8 min-w-[2rem] px-2 rounded-lg text-sm font-medium transition-colors focus:outline-none";
  const btnActive =
    "bg-primary text-white shadow-sm";
  const btnInactive =
    "text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-white/10";
  const btnDisabled =
    "text-gray-300 dark:text-gray-600 cursor-not-allowed";

  return (
    <div className="flex flex-col sm:flex-row items-center justify-between gap-3 px-4 py-3 border-t border-gray-200 dark:border-white/10">
      <p className="text-sm text-gray-500 dark:text-gray-400">
        Showing{" "}
        <span className="font-medium text-gray-700 dark:text-gray-200">
          {from}
        </span>
        {" "}–{" "}
        <span className="font-medium text-gray-700 dark:text-gray-200">
          {to}
        </span>{" "}
        of{" "}
        <span className="font-medium text-gray-700 dark:text-gray-200">
          {totalItems}
        </span>{" "}
        results
      </p>

      <div className="flex items-center gap-1">
        <button
          onClick={() => onPageChange(page - 1)}
          disabled={page === 1}
          className={cn(btnBase, page === 1 ? btnDisabled : btnInactive)}
          aria-label="Previous page"
        >
          <ChevronLeft size={16} />
        </button>

        {getPageNumbers().map((p, i) =>
          p === "..." ? (
            <span
              key={`ellipsis-${i}`}
              className="inline-flex items-center justify-center h-8 w-8 text-sm text-gray-400"
            >
              …
            </span>
          ) : (
            <button
              key={p}
              onClick={() => onPageChange(p as number)}
              className={cn(btnBase, p === page ? btnActive : btnInactive)}
              aria-current={p === page ? "page" : undefined}
            >
              {p}
            </button>
          ),
        )}

        <button
          onClick={() => onPageChange(page + 1)}
          disabled={page === totalPages}
          className={cn(
            btnBase,
            page === totalPages ? btnDisabled : btnInactive,
          )}
          aria-label="Next page"
        >
          <ChevronRight size={16} />
        </button>
      </div>
    </div>
  );
}
