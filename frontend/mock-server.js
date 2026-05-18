const express = require("express");
const bodyParser = require("body-parser");
const app = express();
const port = 8000;
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
app.use((req, res, next) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET,POST,PUT,DELETE,OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
  if (req.method === "OPTIONS") return res.sendStatus(200);
  next();
});

app.post("/auth/token", (req, res) => {
  const username = req.body.username || req.body.email || "user";
  const role = username.includes("admin") ? "SUPER_ADMIN" : "STUDENT";
  return res.json({
    access_token: "mock-access-token-" + role.toLowerCase(),
    refresh_token: "mock-refresh-token",
    token_type: "bearer",
    expires_in: 3600,
  });
});

app.post("/auth/refresh", (req, res) => {
  return res.json({
    access_token: "mock-access-token-refreshed",
    refresh_token: "mock-refresh-token",
    token_type: "bearer",
    expires_in: 3600,
  });
});

app.get("/auth/me", (req, res) => {
  // return a simple user
  return res.json({
    id: 1,
    username: "admin",
    full_name: "Admin User",
    role_names: ["SUPER_ADMIN"],
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
