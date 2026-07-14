/**
 * Shared schedule utilities used by Dashboard.tsx and Schedule.tsx.
 * Centralised here to avoid duplicating WD_TO_DAY and formatTime across files.
 */

/** Weekday code (short or full) → JS Date.getDay() (0=Sun … 6=Sat) */
export const WD_TO_DAY: Record<string, number> = {
  sat: 6, sun: 0, mon: 1, tue: 2, wed: 3, thu: 4, fri: 5,
  saturday: 6, sunday: 0, monday: 1, tuesday: 2, wednesday: 3, thursday: 4, friday: 5,
};

/**
 * Same mapping under the alias used in Attendance.tsx / Schedule.tsx for
 * the short-code-only variant (DAY_CODE_TO_JS).
 */
export const DAY_CODE_TO_JS = WD_TO_DAY;

/** Short day code → full display name */
export const DAY_NAMES_FULL: Record<string, string> = {
  sat: "Saturday", sun: "Sunday", mon: "Monday",
  tue: "Tuesday", wed: "Wednesday", thu: "Thursday", fri: "Friday",
};

/**
 * Convert a 24-hour "HH:MM" string to a 12-hour "h:MM AM/PM" label.
 * Returns the original string unchanged if it cannot be parsed.
 */
export function formatTime(timeStr: string): string {
  if (!timeStr || timeStr === "TBA") return "TBA";
  const [hh, mm] = timeStr.split(":").map(Number);
  if (isNaN(hh)) return timeStr;
  const period = hh >= 12 ? "PM" : "AM";
  const h = hh % 12 || 12;
  return `${h}:${String(mm ?? 0).padStart(2, "0")} ${period}`;
}

/**
 * Compute the Saturday-anchored week bounds (Sat → Fri) for the given date.
 * Returns { weekStart, weekEnd, today } with times zeroed on weekStart/today
 * and end-of-day on weekEnd.
 */
export function getWeekBounds(now = new Date()) {
  const today = new Date(now);
  today.setHours(0, 0, 0, 0);

  const day = today.getDay(); // 0=Sun … 6=Sat
  const daysFromSat = day === 6 ? 0 : day + 1;

  const weekStart = new Date(today);
  weekStart.setDate(today.getDate() - daysFromSat);

  const weekEnd = new Date(weekStart);
  weekEnd.setDate(weekStart.getDate() + 6);
  weekEnd.setHours(23, 59, 59, 999);

  return { weekStart, weekEnd, today };
}
