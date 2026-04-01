# Project Progress Report

## Course-Based Attendance System

### Reporting Period: March 4, 2026 to March 19, 2026

## 1. Introduction

The Course-Based Attendance System has evolved into a hybrid platform that combines:

- AI-powered facial recognition for attendance capture
- Automated attendance session lifecycle management
- A production-oriented FastAPI backend
- Role-based and faculty-scoped access control
- Central and tenant-aware data architecture for academic operations

The system supports both desktop recognition workflows and API-driven operations suitable for web/mobile or institutional integrations, with emphasis on data integrity, readiness, and multi-role governance.

## 2. Development Timeline (From March 4, 2026)

### March 4 to March 7, 2026

- Foundation period in this reporting window.
- No major timestamped commit in this date range, but this marks the beginning of the tracked phase.

### March 8, 2026

- Project documentation expanded and refined across multiple updates.
- Initial backend schema foundation appears in migration chain dated March 8 (core entities, roles, attendance, sessions, schedules, enrollments).

### March 12, 2026

- Major backend milestone delivered:
- FastAPI router surface expanded significantly (classes, courses, faculties, schedules, students, teachers and related CRUD/validation logic).
- Production profile and startup-readiness hardening added.
- Large integration step completed between backend API and recognition workflows.
- README overhaul completed, documenting deployment, endpoints, architecture, and operations.

### March 13, 2026

- Schema migration introduced weekday storage transition from legacy numeric format to day-code format, with backward-compatible conversion logic.

### March 15, 2026

- Departments model introduced and integrated into institutional hierarchy.
- Department references added to class batches, students, and teachers.
- Faculty tenant metadata introduced (tenant database name and provisioning timestamp).
- Tenant provisioning and synchronization workflows formalized.

### March 19, 2026 (Present)

- Class-batch uniqueness scope fix migration added to resolve legacy false-duplicate behavior.
- Report logic consolidated so LATE contributes to PRESENT totals where required.
- Tenant-first and scheduler reliability tests expanded and strengthened.
- Current state reflects a significantly matured backend and multi-tenant-ready academic structure.

## 3. Features Added

### Backend Platform and API

- FastAPI backend with modular routers for:
- Authentication
- Faculties, departments, classes, students, teachers
- Courses and assignments
- Schedules and sessions
- Attendance frame ingestion
- Reports
- Unified JSON-style error payload strategy and health endpoints:
- Health
- Liveness
- Readiness
- Tenant scheduler health report

### Authentication and Authorization

- JWT access and refresh token flow.
- Role model with key roles:
- ACADEMIA
- FACULTY_ADMIN
- TEACHER
- Role-based endpoint protection.
- Faculty-scope enforcement and role-scoped DB session routing.

### Attendance Automation

- Automated session creation via scheduler based on course schedules.
- Session closure with automatic absent marking for non-attending enrolled students.
- Duplicate attendance protection within a session.
- Present/late classification based on grace period logic.
- Timeout-aware face processing response behavior.

### Face Recognition and AI Pipeline

- Continued use of SCRFD-based detection and keypoint handling.
- Embedding-based recognition path with confidence thresholding.
- Anti-spoof gating and occlusion checks with stability windows.
- Quality checks for blur and brightness constraints.
- Runtime and preview pipeline improvements for camera processing.

### Data and Tenant Operations

- Tenant database provisioning utility.
- Tenant synchronization utility (metadata-only default, legacy operational sync explicitly gated).
- Tenant scheduler observability and unhealthy-tenant detection thresholds.

### Deployment and Operations

- Dockerfile and docker-compose backend deployment path.
- Environment profile strategy with production profile support.
- Startup checks for:
- DB connectivity
- Required model availability
- Secret strength
- Schema readiness for upgraded features

### Testing Infrastructure

- Broad backend test coverage for:
- Permissions and role behavior
- Attendance logic and performance
- Scheduler behavior and idempotency
- Tenant-first writes and scope enforcement
- Tenant provisioning/sync behavior
- Reporting correctness

## 4. Features Modified / Improved

### API Validation and Conflict Handling

- Duplicate detection hardened for departments and class batches with normalized comparison (case/whitespace robustness).
- More explicit 400 vs 409 handling based on foreign key vs uniqueness conflict type.
- Improved user-facing detail messages for integrity errors.

### Scheduling and Weekday Handling

- Weekday storage modernized from legacy integer/bitmask behavior to readable day-code strings.
- Conversion logic added to preserve compatibility and reduce migration risk.
- Schedule overlap protection and validation flows improved.

### Reporting Logic

- Attendance reports refined so late attendance contributes to present totals where policy requires it.
- Aggregation consistency improved across:
- Overall course summary
- Date range summary
- Student-level breakdown
- Session-level breakdown

### Academic Structure

- Department layer introduced and propagated to major entities.
- Faculty-department-class consistency checks added to prevent invalid cross-organization operations.

### Security and Runtime Hardening

- Startup readiness checks expanded.
- Secret-key policy enforcement strengthened.
- Health telemetry improved for production diagnostics.

## 5. Features Removed (If Any)

No major user-facing modules were fully removed in this period, but key legacy behaviors were deprecated or replaced:

- Legacy class-batch uniqueness at faculty scope was replaced by department-scoped uniqueness to avoid false conflicts.
- Legacy weekday integer storage was replaced by day-code storage.
- Unsafe fallback behavior for faculty-scoped operations was intentionally blocked in tenant-first mode (operations now fail fast when tenant metadata/provisioning is missing instead of silently using central operational data).

## 6. Bugs Encountered and Fixes

### A. Duplicate Department/Class Errors (HTTP 409)

**Issue**

- Department and class creation/update produced duplicate conflicts in real-world normalization scenarios or under legacy constraints.

**Root causes**

- Inconsistent normalization (case, spacing) between payloads and existing rows.
- Legacy unique constraint scope mismatch in class batches (faculty-level uniqueness remained in some environments).

**Fixes**

- Added normalized duplicate checks before write operations.
- Improved integrity-error classification and deterministic 409 responses for duplicates.
- Added migration to enforce correct uniqueness scope by department.
- Added tests for same-name-allowed-across-departments and case/whitespace duplicate rejection.

### B. Invalid Faculty Reference Errors (HTTP 400)

**Issue**

- Requests referencing invalid or mismatched faculty/department/class relationships failed unexpectedly or with unclear semantics.

**Root causes**

- Foreign key mismatch and cross-faculty payload inconsistencies.
- Missing/incorrect scope linkage for faculty-scoped users.

**Fixes**

- Added explicit organization integrity checks:
- Department belongs to faculty
- Class belongs to faculty and department
- User belongs to current faculty scope
- Standardized foreign-key violations into clear HTTP 400 messages.
- Added tests for faculty mismatch and cross-scope rejection cases.

### C. Attendance Report Logic Bug (Late vs Present)

**Issue**

- LATE records were not consistently counted in PRESENT totals in report aggregates.

**Root cause**

- Aggregation logic treated PRESENT and LATE inconsistently across report endpoints.

**Fixes**

- Introduced unified present-status evaluation where PRESENT and LATE are jointly considered for present totals.
- Applied consistent logic to summary, range, student, and session reports.
- Added dedicated tests validating late-count behavior in each report mode.

## 7. System Architecture Updates

### Backend changes

- FastAPI app modularization across routers, schemas, services, and core infrastructure.
- Better lifecycle management and startup checks.
- Role-scoped and tenant-aware DB access patterns.
- Rate limiting on sensitive routes (auth and attendance frame).

### AI / Face recognition module updates

- Quality and anti-spoof gates stabilized with temporal windows.
- Detection and recognition runtime behavior improved.
- Better handling for unknown, timeout, low-confidence, and invalid-frame scenarios.

### API improvements

- Expanded endpoint surface for institutional CRUD, scheduling, attendance, and reporting.
- Stronger boundary checks for multi-role and multi-faculty usage.
- Improved API error consistency and operational health visibility.

## 8. Database Changes

### Core schema additions and refinements

- Initial schema includes faculties, roles, users, classes, students, teachers, courses, assignments, schedules, enrollments, sessions, and attendance records.
- Department entity introduced and linked into academic structure.
- Tenant metadata fields added to faculties.

### Constraint and relationship improvements

- Department uniqueness per faculty (name/code).
- Class-batch uniqueness moved to department scope.
- Existing duplicate and foreign-key conflict handling made explicit in API layer.

### Migration chain highlights

- Initial schema creation.
- Weekday storage conversion to day codes.
- Department introduction and backfill.
- Faculty tenant metadata introduction.
- Legacy uniqueness scope correction for class batches.

## 9. Testing and Validation

### Automated validation

- Extensive pytest suites cover:
- Attendance logic (duplicate prevention, present/late cutoff, absent auto-marking)
- Attendance performance timeout behavior
- Permission and scheduler behavior
- Tenant-first writes and scope restrictions
- Tenant scheduler execution and readiness alerting
- Tenant sync and provisioning behavior
- Reporting logic correctness, including late-as-present policy

### Swagger/OpenAPI validation

- API is documented and testable through interactive docs, supporting endpoint-level validation in development and staging.

### Base64 image testing for attendance

- Attendance frame endpoint supports base64 image payloads and has been exercised through authenticated request workflows (including real token and sample image submission flow).

### Edge cases handled

- Duplicate creation attempts
- Cross-faculty assignment attempts
- Overlapping schedules
- Out-of-window attendance frames
- Recognition timeout and no-match cases
- Stale/failed tenant scheduler conditions
- Missing tenant metadata/provisioning conditions

## 10. Current System Status

### What is working well

- End-to-end backend API surface is broad and structured.
- Attendance session automation and absent-marking logic are implemented.
- Role-based and faculty-scoped access controls are in place.
- Conflict handling around duplicates and foreign keys is significantly improved.
- Reporting logic now aligns with late/present policy expectations.
- Tenant-aware architecture and scheduler observability are implemented.
- Deployment pathway exists via Docker and production env profile.

### What is partially complete or still stabilizing

- Commit history in this exact reporting window is less granular than the actual code progress, indicating in-progress local work that still needs clean commit segmentation.
- Full tenant operational rollout is controlled via feature flags and requires disciplined environment/migration sequencing.
- Real-world load, multi-camera concurrency, and long-duration production soak validation should continue.
- Final hardening for strict production governance (monitoring depth, incident playbooks, backup/restore automation) can be expanded.

## 11. Challenges Faced

### Technical and integration challenges

- Balancing central metadata ownership with tenant operational routing without unsafe fallback behavior.
- Migration safety for legacy databases with old uniqueness constraints.
- Maintaining backward compatibility during weekday format transition.
- Ensuring deterministic duplicate detection across DB engines and input normalization differences.
- Integrating computationally expensive face-recognition operations with API timeout expectations.
- Preventing scheduler drift and enabling actionable readiness signals in tenant mode.
- Managing environment/tooling constraints in local setup while preserving repeatable development flows.

## 12. Future Improvements / Next Steps

### Recommended next actions

- Finalize and commit outstanding migration and backend changes in smaller, auditable commit units.
- Execute full migration upgrade verification on staging replicas of legacy databases.
- Expand production observability:
- Metrics for recognition latency, match rates, duplicate-block rates, scheduler health
- Alerting for tenant drift and readiness degradation
- Add CI pipelines to run full backend test matrix automatically on every PR.
- Introduce stronger API contract tests for frontend integration.
- Extend report exports (CSV/PDF-ready endpoints) and dashboard analytics views.
- Implement resilience enhancements:
- Queue-based frame processing option for peak load
- Retry/backoff and circuit-breaker patterns around external AI dependencies
- Formalize security hardening checklist:
- Token rotation strategy
- Secret management integration
- Audit trails for role-sensitive operations

## Executive Summary

From March 4 to March 19, 2026, the project progressed from a developing hybrid attendance system into a substantially production-oriented platform with:

- A strong FastAPI backend
- AI-assisted attendance automation
- Robust role/faculty access control
- Maturing tenant-first architecture
- Improved schema integrity and reporting correctness
- Broad automated test coverage

The core platform is functionally strong and structurally mature, with the primary remaining work centered on production hardening, rollout discipline, and continued operational validation.
