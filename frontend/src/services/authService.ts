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
};

export default authService;
