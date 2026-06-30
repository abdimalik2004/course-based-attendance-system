import { api, authClient } from "./api";
import { useAuthStore } from "@/store/useAuthStore";

type TokenResp = { access_token: string };

export const authService = {
  login: async (username: string, password: string) => {
    const params = new URLSearchParams();
    params.append("username", username);
    params.append("password", password);

    // Server will set refresh token as httpOnly cookie; response contains access_token
    const res = await authClient.post<TokenResp>("/auth/token", params, {
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
    });

    const { access_token } = res.data;
    useAuthStore.getState().setTokens(access_token);

    const meResp = await authClient.get("/auth/me", {
      headers: { Authorization: `Bearer ${access_token}` },
    });
    const me = meResp.data;
    useAuthStore.getState().login(me, access_token);
    return me;
  },

  refresh: async () => {
    // Server reads refresh token from cookie and returns a fresh access token
    const res = await authClient.post<TokenResp>("/auth/refresh");
    const { access_token } = res.data;
    useAuthStore.getState().setTokens(access_token);
    return access_token;
  },

  initialize: async () => {
    try {
      const access = await authService.refresh();
      const meResp = await authClient.get("/auth/me", {
        headers: { Authorization: `Bearer ${access}` },
      });
      useAuthStore.getState().login(meResp.data, access);
      return meResp.data;
    } catch (err) {
      useAuthStore.getState().logout();
      return null;
    }
  },

  logout: async () => {
    try {
      await authClient.post("/auth/logout");
    } catch {
      // ignore
    }
    useAuthStore.getState().logout();
  },

  // ── Forgot Password flow ─────────────────────────────────────────────────

  /**
   * Step 1 — request a 6-digit code to be emailed.
   * In development (SMTP not configured) the backend returns `dev_code`
   * so the flow can be tested without a real Gmail account.
   */
  forgotPassword: async (email: string): Promise<{ devCode?: string }> => {
    const res = await authClient.post<{ ok: boolean; dev_code?: string }>(
      "/auth/forgot-password",
      { email },
    );
    return { devCode: res.data.dev_code };
  },

  /**
   * Step 2 — submit the code the user received.
   * Returns a one-time `reset_token` valid for 15 minutes.
   */
  verifyResetCode: async (email: string, code: string): Promise<string> => {
    const res = await authClient.post<{ ok: boolean; reset_token: string }>(
      "/auth/verify-reset-code",
      { email, code },
    );
    return res.data.reset_token;
  },

  /** Step 3 — set a new password using the reset_token from step 2 */
  resetPassword: async (resetToken: string, newPassword: string): Promise<void> => {
    await authClient.post("/auth/reset-password", {
      reset_token: resetToken,
      new_password: newPassword,
    });
  },
};

export default authService;
