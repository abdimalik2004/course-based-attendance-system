import { Card, CardContent } from './Card';
import { cn } from '@/utils/cn';

interface StatCardProps {
  title: string;
  value: string | number;
  icon: React.ElementType;
  iconColor?: 'primary' | 'success' | 'warning' | 'danger';
}

export function StatCard({ title, value, icon: Icon, iconColor = 'primary' }: StatCardProps) {
  const colorStyles = {
    primary: 'text-primary bg-primary/10 border-primary/20 shadow-[0_0_15px_rgba(37,99,235,0.15)]',
    success: 'text-emerald-500 bg-emerald-500/10 border-emerald-500/20 shadow-[0_0_15px_rgba(16,185,129,0.15)]',
    warning: 'text-amber-500 bg-amber-500/10 border-amber-500/20 shadow-[0_0_15px_rgba(245,158,11,0.15)]',
    danger: 'text-rose-500 bg-rose-500/10 border-rose-500/20 shadow-[0_0_15px_rgba(244,63,94,0.15)]',
  };

  return (
    <Card className="glass-card transition-all duration-300 hover:shadow-lg hover:-translate-y-1 group border-white/5">
      <CardContent className="p-6">
        <div className="flex items-center gap-4">
          <div className={cn(
            'flex h-12 w-12 shrink-0 items-center justify-center rounded-xl border transition-colors',
            colorStyles[iconColor],
            'group-hover:scale-110 transition-transform duration-300'
          )}>
            <Icon size={24} />
          </div>
          <div>
            <p className="text-sm font-medium text-gray-400 dark:text-gray-400">
              {title}
            </p>
            <h3 className="text-2xl font-bold tracking-tight text-gray-900 dark:text-white mt-1">
              {value}
            </h3>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
