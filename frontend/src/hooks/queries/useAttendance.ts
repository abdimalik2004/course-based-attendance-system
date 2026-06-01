// src/hooks/queries/useAttendance.ts
import { useQuery } from '@tanstack/react-query';
import { attendanceService } from '@/services/attendanceService';

export const attendanceKeys = {
  all: ['attendance'] as const,
  list: (params?: any) => [...attendanceKeys.all, 'list', params] as const,
};

export function useAttendanceList(params?: { page?: number; limit?: number; search?: string; faculty?: string; department?: string; course?: string; status?: string }) {
  return useQuery({
    queryKey: attendanceKeys.list(params),
    queryFn: () => attendanceService.getAttendanceList(params),
    refetchInterval: 15_000,
  });
}
