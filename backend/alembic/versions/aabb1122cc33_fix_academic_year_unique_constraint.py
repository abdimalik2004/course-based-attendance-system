"""fix academic_year unique constraint to allow multiple terms per year

Revision ID: aabb1122cc33
Revises: merge_e5f6a7b8c9d0_f9e0a1b2c3d4
Create Date: 2026-05-24

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "aabb1122cc33"
down_revision: Union[str, Sequence[str], None] = "merge_e5f6a7b8c9d0_f9e0a1b2c3d4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop the old unique constraint on academic_year alone
    op.drop_constraint("uq_academic_years_academic_year", "academic_years", type_="unique")
    # Add new unique constraint on the combination (academic_year, term_name)
    op.create_unique_constraint(
        "uq_academic_years_year_term",
        "academic_years",
        ["academic_year", "term_name"],
    )


def downgrade() -> None:
    op.drop_constraint("uq_academic_years_year_term", "academic_years", type_="unique")
    op.create_unique_constraint(
        "uq_academic_years_academic_year",
        "academic_years",
        ["academic_year"],
    )
