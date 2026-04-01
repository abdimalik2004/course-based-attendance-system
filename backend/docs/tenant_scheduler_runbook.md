# Tenant Scheduler Runbook

## Purpose

Use this runbook when tenant scheduler mode is enabled and any tenant starts failing or stops ticking.

## Primary Signals

- `GET /health/ready`
  - Expected: `200 OK`
  - Failure: `503` when DB, model loading, or scheduler health is degraded.
- `GET /health/scheduler-tenants`
  - Inspect `unhealthy_tenant_count`
  - Inspect `unhealthy_tenants`
  - Inspect each tenant's `alert_reasons`, `last_error`, `last_tick_completed_at`, and `consecutive_failures`

## Alert Thresholds

- `SCHEDULER_TENANT_FAILURE_THRESHOLD`
  - Default: `3`
  - Meaning: a tenant becomes unhealthy after this many consecutive scheduler failures.
- `SCHEDULER_TENANT_STALE_SECONDS`
  - Default: `180`
  - Meaning: a tenant becomes unhealthy if its last completed tick is older than this threshold while the scheduler is running.

## Triage Flow

1. Check overall readiness.
   - `curl -i http://<host>/health/ready`
2. Check tenant scheduler detail.
   - `curl -s http://<host>/health/scheduler-tenants`
3. Identify affected tenant codes from `unhealthy_tenants`.
4. For each affected tenant, confirm central metadata is still valid.
   - `tenant_db_name` is present
   - `tenant_db_provisioned_at` is present
5. Verify the tenant database is reachable with the configured MySQL credentials.
6. Review backend logs for the tenant's `last_error` and nearby stack traces.

## Common Failure Modes

- Missing tenant DB metadata on the faculty row
- Tenant DB exists but schema drift prevents scheduler queries
- MySQL connectivity or permission failures for tenant databases
- Scheduler loop running, but tenant tick stale because the process is hung or blocked elsewhere

## Recovery Actions

1. If metadata is missing, reprovision tenant metadata:
   - `python -m app.db.provision_tenants --include-provisioned`
2. If schema drift exists, run migrations against the affected database set.
3. If tenant DB access fails, fix MySQL grants or connection settings and restart the backend.
4. If only one tenant is degraded, do not run legacy operational sync unless this is a declared emergency.
5. If emergency backfill is required, enable it explicitly and turn it back off immediately after use:
   - `TENANT_DB_ALLOW_LEGACY_OPERATIONAL_SYNC=true`
   - `python -m app.db.sync_tenants --faculty-code <CODE> --include-operational-tables`

## Exit Criteria

- `GET /health/ready` returns `200`
- `unhealthy_tenant_count` is `0`
- Affected tenants show a fresh `last_success_at`
- No repeated scheduler exceptions in application logs
