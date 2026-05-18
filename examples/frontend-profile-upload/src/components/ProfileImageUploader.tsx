import React, { useState, ChangeEvent } from "react";
import { useUploadProfileImage } from "../hooks/useUploadProfileImage";

type Props = {
  token: string; // auth token
  onUploaded?: (url: string) => void;
};

export const ProfileImageUploader: React.FC<Props> = ({
  token,
  onUploaded,
}) => {
  const { upload, loading, error } = useUploadProfileImage();
  const [preview, setPreview] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);

  function onFileChange(e: ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0] ?? null;
    setFile(f);
    if (f) setPreview(URL.createObjectURL(f));
  }

  async function onSubmit() {
    if (!file) return;
    try {
      const result = await upload(file, token);
      onUploaded?.(result.profile_image_url);
      // revoke preview URL
      if (preview) URL.revokeObjectURL(preview);
      setPreview(null);
      setFile(null);
    } catch (err) {
      // error state from hook
    }
  }

  return (
    <div>
      <div>
        {preview ? (
          <img
            src={preview}
            alt="preview"
            style={{ width: 120, height: 120, objectFit: "cover" }}
          />
        ) : (
          <div style={{ width: 120, height: 120, background: "#eee" }} />
        )}
      </div>
      <input type="file" accept="image/*" onChange={onFileChange} />
      <button onClick={onSubmit} disabled={!file || loading}>
        {loading ? "Uploading..." : "Upload"}
      </button>
      {error && <div style={{ color: "red" }}>{error}</div>}
    </div>
  );
};
