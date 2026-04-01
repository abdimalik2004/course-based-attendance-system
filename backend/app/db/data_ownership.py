from __future__ import annotations

# Central DB remains the source of truth for global platform/auth metadata.
CENTRAL_PLATFORM_TABLE_KEYS = (
    "faculties",
    "roles",
    "users",
    "user_role_links",
)

# Faculty-scoped operational data lives in tenant DBs when tenant routing is enabled.
TENANT_OPERATIONAL_TABLE_KEYS = (
    "departments",
    "class_batches",
    "teachers",
    "students",
    "courses",
    "course_assignments",
    "course_schedules",
    "enrollments",
    "attendance_sessions",
    "attendance_records",
)