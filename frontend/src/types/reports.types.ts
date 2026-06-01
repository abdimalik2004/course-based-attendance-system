export interface ReportSummary {
  totalStudents: number;
  totalTeachers: number;
  totalFaculties: number;
  attendanceRate: number; // percentage
}

export interface AbsenceRecord {
  id: string;
  studentName: string;
  type: string; // e.g. 'Student', 'Faculty', 'Teacher'
  facultyOrDepartment: string;
  totalAbsences: number;
  attendancePercentage: number;
  status: 'Low' | 'Normal' | 'Good'; // Based on attendance percentage
}

export interface ChartDataPoint {
  name: string; // e.g. 'Jan', 'Feb'
  value: number; // e.g. percentage or count
}

export interface DistributionData {
  students: number; // percentage of total
  teachers: number; // percentage or count
  faculties: number; // percentage or count
}
