ALTER TABLE faculties
ADD COLUMN years INT NOT NULL DEFAULT 4;

ALTER TABLE faculties
ADD COLUMN semesters INT NOT NULL DEFAULT 8;

UPDATE faculties
SET years = COALESCE(years, 4),
    semesters = COALESCE(semesters, COALESCE(years, 4) * 2)
WHERE years IS NULL OR semesters IS NULL;

-- Enforce years >= 3 and semesters = years * 2 through the ORM/Alembic migration layer.