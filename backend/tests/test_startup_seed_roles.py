from __future__ import annotations

import app.main as main_module
from app.db.models import Role


class _FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _FakeSession:
    def __init__(self):
        self.query_args = []
        self.added = []
        self.committed = False
        self.closed = False

    def query(self, arg):
        self.query_args.append(arg)
        return _FakeQuery([("SUPER_ADMIN",), ("ACADEMIA",)])

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        self.committed = True

    def close(self):
        self.closed = True


def test_seed_roles_uses_role_names_only(monkeypatch):
    session = _FakeSession()
    monkeypatch.setattr(main_module, "SessionLocal", lambda: session)

    main_module.seed_roles()

    assert session.query_args == [Role.name]
    assert [role.name for role in session.added] == ["FACULTY", "FACULTY_ADMIN", "HR", "ADMISSIONS", "TEACHER"]
    assert session.committed is True
    assert session.closed is True