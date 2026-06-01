import axios, {
  type AxiosError,
  type AxiosInstance,
  type AxiosRequestConfig,
} from "axios";
import { useAuthStore } from "@/store/useAuthStore";
import { toast } from "@/store/useToastStore";

const BASE_URL = import.meta.env.VITE_API_URL || "/api";

// Core axios instance used throughout the app
export const api: AxiosInstance = axios.create({
  baseURL: BASE_URL,
  withCredentials: true,
  headers: {
    "Content-Type": "application/json",
  },
});

// Lightweight client used for auth calls to avoid interceptor loops
const authClient = axios.create({
  baseURL: BASE_URL,
  withCredentials: true,
  headers: {
    "Content-Type": "application/json",
  },
});

let isRefreshing = false;
let refreshSubscribers: Array<(token: string) => void> = [];

function subscribeTokenRefresh(cb: (token: string) => void) {
  refreshSubscribers.push(cb);
}

function onRefreshed(token: string) {
  refreshSubscribers.forEach((cb) => cb(token));
  refreshSubscribers = [];
}

// Attach access token to requests
api.interceptors.request.use((config: AxiosRequestConfig) => {
  const token = useAuthStore.getState().accessToken;
  if (token && config.headers) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Response interceptor with refresh token handling
api.interceptors.response.use(
  (res) => res,
  async (error: AxiosError) => {
    const originalRequest = error.config as AxiosRequestConfig & {
      _retry?: boolean;
    };
    const status = error.response?.status;

    // If unauthorized and we haven't retried yet, attempt refresh
    if (status === 401 && !originalRequest?._retry) {
      originalRequest._retry = true;

      if (isRefreshing) {
        // Queue the request until token refreshed
        return new Promise((resolve, reject) => {
          subscribeTokenRefresh((token: string) => {
            if (!originalRequest.headers) originalRequest.headers = {};
            originalRequest.headers.Authorization = `Bearer ${token}`;
            resolve(api(originalRequest));
          });
        });
      }

      isRefreshing = true;
      try {
        const resp = await authClient.post("/auth/refresh");
        const data = resp.data as { access_token: string };
        useAuthStore.getState().setTokens(data.access_token);
        onRefreshed(data.access_token);

        if (!originalRequest.headers) originalRequest.headers = {};
        originalRequest.headers.Authorization = `Bearer ${data.access_token}`;
        return api(originalRequest);
      } catch (err) {
        useAuthStore.getState().logout();
        window.location.href = "/login";
        return Promise.reject(err);
      } finally {
        isRefreshing = false;
      }
    }

    // Handle forbidden — show a non-blocking toast instead of alert()
    if (status === 403) {
      const data = error.response?.data as any;
      // Backend wraps errors as { error: { message: "..." } }; fall back to legacy { detail: "..." }
      const message =
        (typeof data?.error?.message === "string" ? data.error.message : null) ??
        (typeof data?.detail === "string" ? data.detail : null) ??
        "You don't have permission to perform this action.";
      toast.error(message);
    }

    return Promise.reject(error);
  },
);

export { authClient };
