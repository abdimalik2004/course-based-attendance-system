# Frontend example: Profile image upload

This folder contains a minimal React + TypeScript example showing how to upload a profile image to the backend endpoint `/users/me/profile-image` and update the user's profile URL.

Files:

- `src/hooks/useUploadProfileImage.ts` — React hook using `fetch` to POST `FormData`.
- `src/components/ProfileImageUploader.tsx` — Small component showing preview and upload UI.

Usage: adapt these files into your frontend project. They assume you already have authentication (Bearer token) available.
