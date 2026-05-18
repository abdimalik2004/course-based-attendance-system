import { useQuery } from '@tanstack/react-query';
import { reportsService } from '@/services/reportsService';

export const reportKeys = {
  all: ['reports'] as const,
  summary: () => [...reportKeys.all, 'summary'] as const,
  absenceRanking: (params?: any) => [...reportKeys.all, 'absenceRanking', params] as const,
  chartData: () => [...reportKeys.all, 'chartData'] as const,
  distribution: () => [...reportKeys.all, 'distribution'] as const,
};

export function useReportSummary() {
  return useQuery({
    queryKey: reportKeys.summary(),
    queryFn: () => reportsService.getSummaryMetrics(),
  });
}

export function useAbsenceRanking(params?: { page?: number; limit?: number; search?: string; type?: string; faculty?: string; department?: string; course?: string }) {
  return useQuery({
    queryKey: reportKeys.absenceRanking(params),
    queryFn: () => reportsService.getAbsenceRanking(params),
  });
}

export function useAttendanceChartData() {
  return useQuery({
    queryKey: reportKeys.chartData(),
    queryFn: () => reportsService.getAttendanceChartData(),
  });
}

export function useDistributionSummary() {
  return useQuery({
    queryKey: reportKeys.distribution(),
    queryFn: () => reportsService.getDistributionSummary(),
  });
}
