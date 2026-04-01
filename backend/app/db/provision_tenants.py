from __future__ import annotations

import argparse
from datetime import datetime, timezone

from app.db.models import Faculty
from app.db.session import SessionLocal
from app.services.tenant_provisioning import build_tenant_db_name, provision_faculty_tenant_database


def provision_all_faculty_tenants(*, include_provisioned: bool = False) -> dict[str, int]:
    db = SessionLocal()
    summary = {
        "processed": 0,
        "provisioned": 0,
        "skipped": 0,
        "failed": 0,
    }
    try:
        faculties = db.query(Faculty).order_by(Faculty.id).all()
        for faculty in faculties:
            if not include_provisioned and faculty.tenant_db_provisioned_at is not None:
                continue

            tenant_db_name = faculty.tenant_db_name or build_tenant_db_name(faculty.code)
            faculty.tenant_db_name = tenant_db_name

            result = provision_faculty_tenant_database(tenant_db_name)
            summary["processed"] += 1
            if result.provisioned:
                faculty.tenant_db_provisioned_at = datetime.now(timezone.utc)
                summary["provisioned"] += 1
                continue

            if result.skipped:
                summary["skipped"] += 1
            else:
                summary["failed"] += 1

        db.commit()
        return summary
    finally:
        db.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Provision tenant databases for all faculties")
    parser.add_argument(
        "--include-provisioned",
        action="store_true",
        help="Also re-check faculties already marked as tenant-provisioned",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = provision_all_faculty_tenants(include_provisioned=args.include_provisioned)
    print(
        "Tenant provisioning summary:",
        f"processed={summary['processed']}",
        f"provisioned={summary['provisioned']}",
        f"skipped={summary['skipped']}",
        f"failed={summary['failed']}",
    )


if __name__ == "__main__":
    main()
