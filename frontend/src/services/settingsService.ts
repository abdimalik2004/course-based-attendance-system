import { api } from './api';

/** Fetch all system settings as a flat key→value record. */
export async function fetchSettings(): Promise<Record<string, string>> {
  const res = await api.get<Record<string, string>>('/settings');
  return res.data;
}

/** Bulk-upsert settings — only send the keys you want to change. */
export async function saveSettings(
  payload: Record<string, string | boolean | number>
): Promise<Record<string, string>> {
  // Coerce everything to strings so the backend always receives string values
  const stringified: Record<string, string> = {};
  for (const [k, v] of Object.entries(payload)) {
    stringified[k] = String(v);
  }
  const res = await api.put<Record<string, string>>('/settings', stringified);
  return res.data;
}
