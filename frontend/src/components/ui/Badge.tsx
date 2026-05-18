import { forwardRef } from 'react';
import type { HTMLAttributes } from 'react';
import { cn } from '@/utils/cn';

export interface BadgeProps extends HTMLAttributes<HTMLDivElement> {
  variant?: 'success' | 'warning' | 'danger' | 'info' | 'default';
}

const Badge = forwardRef<HTMLDivElement, BadgeProps>(
  ({ className, variant = 'default', children, ...props }, ref) => {
    const variants = {
      success: 'bg-emerald-500/10 text-emerald-500 border-emerald-500/20',
      warning: 'bg-amber-500/10 text-amber-500 border-amber-500/20',
      danger: 'bg-rose-500/10 text-rose-500 border-rose-500/20',
      info: 'bg-blue-500/10 text-blue-500 border-blue-500/20',
      default: 'bg-gray-500/10 text-gray-500 border-gray-500/20 dark:text-gray-400',
    };

    return (
      <div
        ref={ref}
        className={cn(
          'inline-flex items-center justify-center px-2.5 py-0.5 text-xs font-semibold rounded-full border',
          variants[variant],
          className
        )}
        {...props}
      >
        {children}
      </div>
    );
  }
);
Badge.displayName = 'Badge';

export { Badge };
