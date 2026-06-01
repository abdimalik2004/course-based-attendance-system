/**
 * Activity Log Service
 *
 * Handles fetching recent activity logs and maintaining real-time
 * WebSocket connection for activity updates.
 */

import { api } from "./api";

export interface ActivityLog {
  id: number;
  username: string;
  action: string;
  status: "Success" | "Failed" | "Pending";
  created_at: string;
}

export interface ActivityStats {
  total_activities: number;
  success_count: number;
  failed_count: number;
  pending_count: number;
  unique_users: number;
}

/**
 * Format relative time difference
 * Examples: "2 mins ago", "1 hour ago", "just now"
 */
export function formatTimeAgo(dateStr: string): string {
  const diff = Date.now() - new Date(dateStr).getTime();
  const mins = Math.floor(diff / 60000);

  if (mins < 1) return "just now";
  if (mins < 60) return `${mins} min${mins > 1 ? "s" : ""} ago`;

  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs} hr${hrs > 1 ? "s" : ""} ago`;

  const days = Math.floor(hrs / 24);
  return `${days} day${days > 1 ? "s" : ""} ago`;
}

/**
 * Format time for display
 * Examples: "14:30", "09:15"
 */
export function formatTime(dateStr: string): string {
  return new Date(dateStr).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
}

/**
 * Format full datetime for tooltips
 * Examples: "May 22, 2024 2:30 PM"
 */
export function formatFullDateTime(dateStr: string): string {
  return new Date(dateStr).toLocaleString([], {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

class ActivityService {
  private wsConnection: WebSocket | null = null;
  private wsListeners: ((activity: ActivityLog) => void)[] = [];
  private wsReconnectAttempts = 0;
  private wsMaxReconnectAttempts = 5;
  private wsReconnectDelay = 3000; // 3 seconds

  /**
   * Fetch recent activities from the last N hours
   */
  async getRecentActivities(
    limit: number = 30,
    hours: number = 2,
  ): Promise<ActivityLog[]> {
    try {
      const response = await api.get("/activity/recent", {
        params: { limit, hours },
      });
      return response.data;
    } catch (error) {
      console.error("Failed to fetch recent activities:", error);
      throw error;
    }
  }

  /**
   * Get activity statistics for the last N hours
   */
  async getActivityStats(hours: number = 24): Promise<ActivityStats> {
    try {
      const response = await api.get("/activity/stats", {
        params: { hours },
      });
      return response.data;
    } catch (error) {
      console.error("Failed to fetch activity stats:", error);
      throw error;
    }
  }

  /**
   * Connect to WebSocket for real-time activity updates
   */
  connectWebSocket(
    onActivity: (activity: ActivityLog) => void,
    onError?: (error: Event) => void,
    onClose?: (event: CloseEvent) => void,
  ): void {
    // Don't create multiple connections
    if (this.wsConnection && this.wsConnection.readyState === WebSocket.OPEN) {
      console.log("WebSocket already connected");
      this.wsListeners.push(onActivity);
      return;
    }

    // Register listener
    this.wsListeners.push(onActivity);

    // Build WebSocket URL.
    // In production the frontend and backend share the same origin, so window.location.host works.
    // In development the Vite dev server (e.g. :5173) and the FastAPI backend (e.g. :8000) are on
    // different ports. We derive the backend host from VITE_API_URL so the WebSocket always lands
    // on the correct server regardless of environment.
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const apiBase: string = import.meta.env.VITE_API_URL ?? "";
    // Strip scheme (http:// or https://) from the env var — we only need host[:port][/path].
    // If VITE_API_URL is empty we fall back to the current window host (production).
    const backendHost = apiBase.replace(/^https?:\/\//, "") || window.location.host;
    const wsUrl = `${protocol}//${backendHost}/activity/ws/recent`;

    console.log("Connecting to WebSocket:", wsUrl);

    try {
      this.wsConnection = new WebSocket(wsUrl);

      this.wsConnection.onopen = () => {
        console.log("WebSocket connected");
        this.wsReconnectAttempts = 0;
      };

      this.wsConnection.onmessage = (event: MessageEvent) => {
        try {
          const message = JSON.parse(event.data);

          if (message.type === "activity" && message.data) {
            // Notify all listeners
            this.wsListeners.forEach((listener) => {
              try {
                listener(message.data);
              } catch (e) {
                console.error("Error in activity listener:", e);
              }
            });
          }
        } catch (e) {
          console.error("Error parsing WebSocket message:", e);
        }
      };

      this.wsConnection.onerror = (event: Event) => {
        console.error("WebSocket error:", event);
        onError?.(event);
      };

      this.wsConnection.onclose = (event: CloseEvent) => {
        console.log("WebSocket closed:", event);
        this.wsConnection = null;
        onClose?.(event);

        // Attempt to reconnect
        this.attemptReconnect(onActivity, onError, onClose);
      };
    } catch (error) {
      console.error("Failed to create WebSocket:", error);
    }
  }

  /**
   * Attempt to reconnect to WebSocket with exponential backoff
   */
  private attemptReconnect(
    onActivity: (activity: ActivityLog) => void,
    onError?: (error: Event) => void,
    onClose?: (event: CloseEvent) => void,
  ): void {
    if (this.wsReconnectAttempts >= this.wsMaxReconnectAttempts) {
      console.warn("Max WebSocket reconnection attempts reached");
      return;
    }

    this.wsReconnectAttempts++;
    const delay =
      this.wsReconnectDelay * Math.pow(2, this.wsReconnectAttempts - 1);

    console.log(
      `Reconnecting to WebSocket in ${delay}ms (attempt ${this.wsReconnectAttempts}/${this.wsMaxReconnectAttempts})`,
    );

    setTimeout(() => {
      this.connectWebSocket(onActivity, onError, onClose);
    }, delay);
  }

  /**
   * Disconnect from WebSocket
   */
  disconnectWebSocket(): void {
    if (this.wsConnection) {
      this.wsConnection.close();
      this.wsConnection = null;
    }
    this.wsListeners = [];
    this.wsReconnectAttempts = 0;
  }

  /**
   * Check if WebSocket is connected
   */
  isWebSocketConnected(): boolean {
    return (
      this.wsConnection !== null &&
      this.wsConnection.readyState === WebSocket.OPEN
    );
  }

  /**
   * Get the number of registered listeners
   */
  getListenerCount(): number {
    return this.wsListeners.length;
  }

  /**
   * Remove a specific listener
   */
  removeListener(listener: (activity: ActivityLog) => void): void {
    const index = this.wsListeners.indexOf(listener);
    if (index > -1) {
      this.wsListeners.splice(index, 1);
    }
  }

  /**
   * Clear all listeners
   */
  clearListeners(): void {
    this.wsListeners = [];
  }
}

// Export singleton instance
export const activityService = new ActivityService();

export default activityService;
