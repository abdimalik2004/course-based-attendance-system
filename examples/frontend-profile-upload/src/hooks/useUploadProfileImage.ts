import { useState } from "react";

type UploadResult = {
  profile_image_url: string;
};

export function useUploadProfileImage() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function upload(file: File, token: string): Promise<UploadResult> {
    setLoading(true);
    setError(null);
    const form = new FormData();
    form.append("file", file, file.name);

    const resp = await fetch("/users/me/profile-image", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
      },
      body: form,
    });

    setLoading(false);
    if (!resp.ok) {
      const json = await resp.json().catch(() => null);
      const msg =
        json?.error?.message ||
        (json?.detail ?? resp.statusText) ||
        "Upload failed";
      setError(msg);
      throw new Error(msg);
    }

    const data = (await resp.json()) as UploadResult;
    return data;
  }

  return { upload, loading, error } as const;
}
