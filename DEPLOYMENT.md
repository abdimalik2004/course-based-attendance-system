# Deployment Guide — Vercel (Frontend) + Railway (Backend + MySQL)

## What you'll have when done

```
Browser  ──►  Vercel CDN  ──►  React SPA
                  │
                  ▼ API calls (HTTPS)
             Railway Service  ──►  FastAPI
                  │
                  ▼
             Railway MySQL Plugin
```

---

## Prerequisites

- GitHub account (your code must be in a GitHub repo)
- Railway account at railway.app — subscribe to the $20 Pro plan
- Vercel account at vercel.com (free tier is fine for frontend)

If your code is not on GitHub yet:

```bash
cd attendance_system
git init
git add .
git commit -m "initial commit"
# create a repo on github.com, then:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

---

## Part 1 — Railway (Backend + Database)

### Step 1 — Create a new Railway project

1. Go to https://railway.app and log in
2. Click **New Project**
3. Choose **Empty Project**
4. Click the project name and rename it (e.g., "heegan-attendance")

---

### Step 2 — Add MySQL database

1. Inside your Railway project, click **+ New**
2. Choose **Database → MySQL**
3. Railway creates the MySQL service instantly
4. Click the MySQL service → go to the **Variables** tab
5. You'll see: `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DATABASE`
   — keep this tab open, you'll need these values next

---

### Step 3 — Add the Python backend service

1. Click **+ New** again → choose **GitHub Repo**
2. Connect your GitHub account if prompted
3. Select your repository
4. Railway will detect the Dockerfile automatically

**Configure the service:**

- Click the new service → go to **Settings**
- Under **Root Directory**: type `attendance_system`
  (the Dockerfile lives here and it copies the full project tree)
- Under **Service Name**: rename to "backend"

---

### Step 4 — Set environment variables

1. Click your backend service → go to the **Variables** tab
2. Click **Add Variable** and add each one from `.env.production.example`

The most important ones:

```
APP_ENV                    production
SECRET_KEY                 (generate: python -c "import secrets; print(secrets.token_hex(32))")
MYSQL_HOST                 ${{MySQL.MYSQL_HOST}}
MYSQL_PORT                 ${{MySQL.MYSQL_PORT}}
MYSQL_USER                 ${{MySQL.MYSQL_USER}}
MYSQL_PASSWORD             ${{MySQL.MYSQL_PASSWORD}}
MYSQL_DB                   ${{MySQL.MYSQL_DATABASE}}
DB_TYPE                    mysql
CORS_ALLOW_ORIGINS         https://your-project.vercel.app   ← update after Vercel deploy
SMTP_EMAIL                 mahadalleabdimalik@gmail.com
SMTP_PASSWORD              ritc nneg zocd pqlq
FACE_CONFIDENCE_THRESHOLD  0.58
FACE_TIMEOUT_SECONDS       2.0
```

The `${{MySQL.MYSQL_HOST}}` syntax tells Railway to automatically inject the
MySQL plugin's internal hostname — you don't type the actual value.

---

### Step 5 — Deploy the backend

1. Go to the **Deployments** tab of your backend service
2. Click **Deploy** (or it may auto-deploy after variables are set)
3. Watch the build logs — the first build takes 10–20 minutes because it:
   - Installs PyTorch (~200 MB, CPU-only)
   - Installs OpenCV, InsightFace, ONNX Runtime
   - Downloads the InsightFace buffalo_l model (~600 MB)
4. When you see `Application startup complete` in the logs, it's ready

**Get your Railway backend URL:**

- Click the backend service → **Settings → Networking → Public Networking**
- Click **Generate Domain** — Railway gives you a URL like:
  `https://backend-production-xxxx.up.railway.app`
- Save this URL — you need it for Vercel

---

### Step 6 — Verify the backend is working

Open in your browser:

```
https://backend-production-xxxx.up.railway.app/docs
```

You should see the FastAPI Swagger UI. If you do, the backend is live.

---

## Part 2 — Vercel (Frontend)

### Step 7 — Deploy the frontend

1. Go to https://vercel.com and log in
2. Click **Add New → Project**
3. Import your GitHub repository
4. Vercel asks you to configure the project:

   | Setting          | Value                        |
   | ---------------- | ---------------------------- |
   | Framework Preset | Vite                         |
   | Root Directory   | `attendance_system/frontend` |
   | Build Command    | `npm run build`              |
   | Output Directory | `dist`                       |

5. Before clicking Deploy, click **Environment Variables** and add:

   ```
   VITE_API_URL    https://backend-production-xxxx.up.railway.app
   ```

   (replace with your actual Railway URL from Step 5)

6. Click **Deploy** — Vercel builds in ~1–2 minutes

7. Vercel gives you a URL like: `https://your-project.vercel.app`

---

### Step 8 — Update CORS on Railway

Now that you have your Vercel URL, go back to Railway:

1. Backend service → **Variables**
2. Update `CORS_ALLOW_ORIGINS` to your Vercel URL:
   ```
   CORS_ALLOW_ORIGINS    https://your-project.vercel.app
   ```
3. Railway auto-redeploys with the new variable

---

### Step 9 — Test the full system

Open your Vercel URL in the browser and try logging in. Check that:

- Login works
- Face recognition pages load
- Images upload correctly

---

## Part 3 — Persistent Storage (Dataset + Models + Uploads)

Your project has three folders that must survive redeploys:

| Folder | What's in it |
|--------|-------------|
| `dataset/` | Student face photos, organised as `dataset/{FACULTY}/{STUDENT_ID}/` |
| `models/` | Trained embeddings (`face_embeddings.npz`, `label_map.json`, etc.) |
| `backend/backend/static/uploads/` | Profile images uploaded via the UI |

All three are redirected to a single Railway Volume at `/data` via environment
variables that the Dockerfile already sets. Here's how to add the volume:

1. Railway backend service → **Volumes** tab → **Add Volume**
2. **Mount path**: `/data`
3. **Size**: 10 GB (adjustable later — covers dataset, models, and uploads)
4. Click **Add** — Railway redeploys with the volume attached

**What happens on first boot:**
The `entrypoint.sh` script detects that `/data` is empty and automatically
copies your existing `dataset/` and `models/` (baked into the Docker image)
into the volume. Your current students and trained embeddings carry over
without any manual work.

On every subsequent boot `/data` is already populated, so the copy is skipped.

**Adding students after go-live:**
When staff captures face photos for a new student via the Admission page,
the images go directly to `/data/dataset/`. Then clicking "Retrain All" in
Settings → Face Model updates the embedding file at `/data/models/`. Everything
stays on the volume and persists across all future redeploys.

---

## Part 4 — Custom Domain (Optional)

### Frontend (Vercel)

1. Buy a domain on Namecheap, Cloudflare, etc. (e.g., `heeganattend.com`)
2. In Vercel: **Settings → Domains → Add Domain** → type your domain
3. Vercel shows DNS records to add (usually an A record and CNAME)
4. Go to your domain registrar and add those records
5. Wait 5–30 minutes for DNS to propagate

### Backend (Railway) — for API subdomain

1. In Railway: backend service → **Settings → Networking → Add Custom Domain**
2. Type: `api.heeganattend.com`
3. Railway shows a CNAME record to add at your registrar
4. Add the CNAME at your registrar
5. Update `VITE_API_URL` on Vercel to `https://api.heeganattend.com`
6. Update `CORS_ALLOW_ORIGINS` on Railway to `https://heeganattend.com`

---

## Troubleshooting

| Symptom                              | Likely cause                                  | Fix                                                                    |
| ------------------------------------ | --------------------------------------------- | ---------------------------------------------------------------------- |
| Login fails with CORS error          | CORS_ALLOW_ORIGINS doesn't include Vercel URL | Update the Railway variable                                            |
| `/docs` returns 502                  | Build still in progress or startup error      | Check Railway deployment logs                                          |
| Face images disappear after redeploy | No Railway Volume attached                    | Add volume mounted to `/app/static/uploads`                            |
| "Internal server error" on login     | DB not connected                              | Check MYSQL\_\* vars, check Railway MySQL service is running           |
| Frontend shows blank page            | Vercel routing issue                          | Confirm `vercel.json` is present in frontend folder                    |
| Build times out                      | First build too large                         | Normal — buffalo_l model download takes time. Pro tier has no timeout. |

---

## Redeployment (everyday use)

After this initial setup, every time you `git push` to your main branch:

- **Railway** auto-detects the push and rebuilds the backend
- **Vercel** auto-detects the push and rebuilds the frontend

No manual steps needed.
