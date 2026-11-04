"""Offline tests (no Docker, no LLM): replaying a trajectory and mining repair candidates."""

from traj2envs.families.repair import Repair
from traj2envs.trajectory import Trajectory, VirtualFilesystem


def tool_step(step_id, tool, arguments, output, is_error=False):
    result = {"source_call_id": f"call{step_id}", "content": output}
    if is_error:
        result["extra"] = {"is_error": True}
    return {"step_id": step_id, "source": "agent", "message": "",
            "tool_calls": [{"tool_call_id": f"call{step_id}", "function_name": tool, "arguments": arguments}],
            "observation": {"results": [result]}}


WRITE_BUGGY_CODE = tool_step(2, "Write", {"file_path": "/app/calc.py", "content": "def add(a, b):\n    return a - b\n"}, "ok")
WRITE_TEST = tool_step(3, "Write", {"file_path": "/app/test_calc.py",
                                    "content": "from calc import add\n\ndef test_add():\n    assert add(1, 2) == 3\n"}, "ok")
RUN_TESTS_FAILING = tool_step(4, "Bash", {"command": "cd /app && python -m pytest -q"},
                              "Exit code 1\nFAILED test_calc.py::test_add - assert -1 == 3", is_error=True)
FIX_CODE = tool_step(5, "Edit", {"file_path": "/app/calc.py", "old_string": "a - b", "new_string": "a + b"}, "ok")
RUN_TESTS_PASSING = tool_step(6, "Bash", {"command": "cd /app && python -m pytest -q"}, "1 passed")
USER_REQUEST = {"step_id": 1, "source": "user", "message": "<environment_context>cwd</environment_context>Write add()"}


def trajectory(*steps):
    return Trajectory({"schema_version": "ATIF-v1.7", "session_id": "demo", "steps": [USER_REQUEST, *steps]})


def test_replay_gives_files_before_each_command():
    runs, filesystem = trajectory(WRITE_BUGGY_CODE, WRITE_TEST, RUN_TESTS_FAILING, FIX_CODE, RUN_TESTS_PASSING).replay()
    assert [run.exit_code for run in runs] == [1, 0]
    assert runs[0].files_before["/app/calc.py"].endswith("a - b\n")
    assert filesystem.files["/app/calc.py"].endswith("a + b\n")


def test_user_messages_drop_injected_context():
    assert trajectory().user_messages() == [(1, "Write add()")]


def test_apply_patch_resolves_relative_paths():
    filesystem = VirtualFilesystem({"/work/a.py": "x = 1\ny = 2\n"})
    filesystem.apply_patch("*** Begin Patch\n*** Update File: a.py\n@@\n-x = 1\n+x = 10\n y = 2\n*** End Patch", "/work")
    assert filesystem.files["/work/a.py"] == "x = 10\ny = 2\n"


def test_repair_finds_the_failing_then_passing_check():
    (candidate,) = Repair().find_candidates(trajectory(WRITE_BUGGY_CODE, WRITE_TEST, RUN_TESTS_FAILING, FIX_CODE, RUN_TESTS_PASSING))
    assert candidate.project_root == "/app" and candidate.changed_files == ["calc.py"]
    assert "a + b" in candidate.files_at_success["calc.py"]


def test_repair_ignores_fixes_that_only_change_tests():
    weaken_test = tool_step(5, "Edit", {"file_path": "/app/test_calc.py", "old_string": "== 3", "new_string": "== -1"}, "ok")
    assert Repair().find_candidates(trajectory(WRITE_BUGGY_CODE, WRITE_TEST, RUN_TESTS_FAILING, weaken_test, RUN_TESTS_PASSING)) == []
