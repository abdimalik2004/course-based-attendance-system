const express = require("express");
const bodyParser = require("body-parser");
const cookieParser = require("cookie-parser");

const app = express();
const port = 8000;

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
app.use(cookieParser());

const CORS_ORIGIN = "http://localhost:5173";

// CORS middleware
app.use((req, res, next) => {
  res.header("Access-Control-Allow-Origin", CORS_ORIGIN);
  res.header("Access-Control-Allow-Credentials", "true");
  res.header(
    "Access-Control-Allow-Headers",
    "Origin, X-Requested-With, Content-Type, Accept, Authorization",
  );
  res.header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");

  // Handle preflight requests
  if (req.method === "OPTIONS") {
    return res.sendStatus(200);
  }

  next();
});

// Seeded mock users for local dev
const MOCK_USERS = {
  admin: { password: "admin", roles: ["SUPER_ADMIN"], full_name: "Admin" },
  academia: {
    password: "academia",
    roles: ["ACADEMIA"],
    full_name: "Academia",
  },
  hr: { password: "hr", roles: ["HR"], full_name: "HR" },
  admission: {
    password: "admission",
    roles: ["ADMISSIONS"],
    full_name: "Admission",
  },
  science: {
    password: "science",
    roles: ["FACULTY"],
    full_name: "Science Faculty",
  },
  teacher: { password: "teacher", roles: ["TEACHER"], full_name: "Teacher" },
  student: { password: "student", roles: ["STUDENT"], full_name: "Student" },
};

app.post("/auth/token", (req, res) => {
  const username = (req.body.username || req.body.email || "").trim();
  const password = (req.body.password || "").trim();

  const entry = MOCK_USERS[username];
  if (!entry || entry.password !== password) {
    return res.status(401).json({ detail: "Invalid username or password" });
  }

  // return a mock token that encodes the username
  const access = `mock-access-${username}`;
  const refresh = `mock-refresh-${username}`;
  // set refresh token as httpOnly cookie to emulate server behavior
  res.cookie("refresh_token", refresh, {
    httpOnly: true,
    sameSite: "lax",
    maxAge: 3600 * 1000,
  });
  return res.json({ access_token: access });
});

app.post("/auth/refresh", (req, res) => {
  // read refresh token from cookie (emulating httpOnly cookie behavior)
  const cookie = req.cookies?.refresh_token || "";
  const username = cookie.replace("mock-refresh-", "") || "admin";
  const access = `mock-access-${username}-refreshed`;
  const refresh = `mock-refresh-${username}`;
  // rotate cookie
  res.cookie("refresh_token", refresh, {
    httpOnly: true,
    sameSite: "lax",
    maxAge: 3600 * 1000,
  });
  return res.json({ access_token: access });
});

app.get("/auth/me", (req, res) => {
  const auth = (req.headers.authorization || "").toString();
  const parts = auth.split(" ");
  const token = parts.length === 2 ? parts[1] : "";
  // token format: mock-access-<username> or mock-access-<username>-refreshed
  const uname =
    token.replace(/^mock-access-/, "").replace(/-refreshed$/, "") || "admin";
  const entry = MOCK_USERS[uname] || MOCK_USERS["admin"];
  return res.json({
    id: uname,
    username: uname,
    full_name: entry.full_name,
    role_names: entry.roles,
  });
});

app.get("/students", (req, res) => {
  return res.json({ total: 120, items: [{ id: 1, name: "Student A" }] });
});
app.get("/teachers", (req, res) => {
  return res.json({ total: 12, items: [{ id: 1, name: "Teacher A" }] });
});
app.get("/faculties", (req, res) => {
  return res.json({ total: 3, items: [{ id: 1, name: "CIS" }] });
});
app.get("/courses", (req, res) => {
  return res.json({
    total: 5,
    items: [
      { id: 1, name: "Course 1" },
      { id: 2, name: "Course 2" },
      { id: 3, name: "Course 3" },
      { id: 4, name: "Course 4" },
      { id: 5, name: "Course 5" },
    ],
  });
});
app.get("/reports/course/:id", (req, res) => {
  return res.json({ total_records: 50, present: 40 });
});
app.get("/reports", (req, res) => {
  return res.json({ total: 100, present: 80 });
});
app.get("/student-portal/students/:id/attendance", (req, res) => {
  return res.json([{ date: "2026-05-01", status: "present" }]);
});
app.get("/student-portal/students/:id/schedule", (req, res) => {
  return res.json([{ course: "Course 1", time: "09:00" }]);
});

app.listen(port, () => {
  console.log(`Mock API server listening at http://localhost:${port}`);
});
