import { create } from 'zustand';
import type { ReportSummary, AbsenceRecord, ChartDataPoint, DistributionData } from '../types/reports.types';
import { reportsService } from '../services/reportsService';

interface ReportsState {
  summary: ReportSummary | null;
  absenceRecords: AbsenceRecord[];
  chartData: ChartDataPoint[];
  distribution: DistributionData | null;
  isLoading: boolean;
  error: string | null;

  fetchReportsData: () => Promise<void>;
}

export const useReportsStore = create<ReportsState>((set) => ({
  summary: null,
  absenceRecords: [],
  chartData: [],
  distribution: null,
  isLoading: false,
  error: null,

  fetchReportsData: async () => {
    set({ isLoading: true, error: null });
    try {
      const [summary, absenceRecords, chartData, distribution] = await Promise.all([
        reportsService.getSummaryMetrics(),
        reportsService.getAbsenceRanking(),
        reportsService.getAttendanceChartData(),
        reportsService.getDistributionSummary()
      ]);
      set({ summary, absenceRecords, chartData, distribution, isLoading: false });
    } catch {
      set({ error: 'Failed to fetch report data', isLoading: false });
    }
  }
}));
