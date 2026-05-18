import { api } from "./api";

export const fileService = {
  uploadProfileImage: async (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    const res = await api.post("/users/me/profile-image", fd, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    return res.data;
  },

  sendAttendanceFrame: async (session_id: number, base64Image: string) => {
    const payload = { session_id, image: base64Image };
    const res = await api.post("/attendance/frame", payload);
    return res.data;
  },
};

export default fileService;
