import pytest
from unittest.mock import AsyncMock, MagicMock

from project.task_estimator import EstimatorConfig, Estimate, TaskEstimator


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def mock_repo():
    repo = MagicMock()
    repo.get_task = MagicMock(return_value=None)
    repo.get_task_dir = MagicMock(return_value=None)
    return repo


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.call = AsyncMock()
    return llm


@pytest.fixture
def mock_config():
    return MagicMock()


@pytest.fixture
def est_config():
    return EstimatorConfig()


@pytest.fixture
def estimator(mock_repo, mock_llm, mock_config, est_config):
    return TaskEstimator(mock_repo, mock_llm, mock_config, est_config)


def make_task(task_type="coding", instruction="", check="exact_str"):
    return {"task_type": task_type, "instruction": instruction, "check": check}


def make_llm_response(content):
    resp = MagicMock()
    resp.content = content
    return resp


def wire_repo(estimator, task, task_dir=None):
    estimator.task_repo.get_task.return_value = task
    estimator.task_repo.get_task_dir.return_value = task_dir


# ----------------------------------------------------------------------
# Heuristic layer
# ----------------------------------------------------------------------

class TestEstimateHeuristic:

    def test_missing_task_returns_defaults(self, estimator):
        estimator.task_repo.get_task.return_value = None
        est = estimator._heuristic_estimate("missing")
        assert isinstance(est, Estimate)
        assert est.source == "heuristic_missing"
        assert est.tokens == estimator.est_cfg.default_base
        assert est.p_solvable == estimator.est_cfg.default_p_solvable
        assert est.confidence == 0.1

    def test_basic_estimate(self, estimator):
        wire_repo(estimator, make_task("coding", "x" * 400, "exact_int"))
        est = estimator._heuristic_estimate("t1")
        # base=3000, instr=100, workspace=0, check=200 => 3300 * 1.2 = 3960
        assert est.tokens == int((3000 + 100 + 0 + 200) * 1.2)
        assert est.p_solvable == 0.65
        assert est.source == "heuristic"
        assert est.confidence == estimator.est_cfg.heuristic_confidence

    def test_workspace_cost_included(self, estimator, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()
        (ws / "a.txt").write_text("x" * 1000)
        (ws / "b.txt").write_text("x" * 2000)
        wire_repo(estimator, make_task("search", "", "exact_str"), tmp_path)
        est = estimator._heuristic_estimate("t1")
        # base=800, instr=0, ws=750, check=100 => 1650 * 1.2 = 1980
        assert est.tokens == int((800 + 0 + 750 + 100) * 1.2)

    def test_workspace_cost_capped(self, estimator, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()
        (ws / "big.bin").write_bytes(b"x" * 100_000)
        assert estimator._workspace_cost(tmp_path) == estimator.est_cfg.workspace_read_cap

    def test_workspace_cost_none_dir(self, estimator):
        assert estimator._workspace_cost(None) == 0

    def test_workspace_cost_missing_workspace(self, estimator, tmp_path):
        assert estimator._workspace_cost(tmp_path) == 0

    def test_minimum_floor(self, estimator):
        estimator._base_costs["coding"] = 1
        estimator.est_cfg.heuristic_safety = 0.1
        wire_repo(estimator, make_task("coding", "", "exact_str"))
        est = estimator._heuristic_estimate("t1")
        assert est.tokens == 100

    def test_unknown_task_type_uses_default_base(self, estimator):
        wire_repo(estimator, make_task("weird_type", "", "exact_str"))
        est = estimator._heuristic_estimate("t1")
        assert est.tokens == int((estimator.est_cfg.default_base + 0 + 0 + 100) * 1.2)

    def test_heuristic_p_solvable_known(self, estimator):
        assert estimator._heuristic_p_solvable("coding") == 0.65
        assert estimator._heuristic_p_solvable("math") == 0.75

    def test_heuristic_p_solvable_unknown(self, estimator):
        assert (
            estimator._heuristic_p_solvable("unknown")
            == estimator.est_cfg.default_p_solvable
        )


class TestCheckBonus:

    @pytest.mark.parametrize("check,expected", [
        ("unittest", 1000),
        ("exact_int", 200),
        ("exact_str", 100),
        ("contains", 100),
        ("anything", 100),
    ])
    def test_check_bonus(self, estimator, check, expected):
        assert estimator._check_bonus({"check": check}) == expected

    def test_check_bonus_missing_field(self, estimator):
        assert estimator._check_bonus({}) == 100


class TestInstructionCost:

    def test_normal(self, estimator):
        assert estimator._instruction_cost({"instruction": "x" * 400}) == 100

    def test_empty(self, estimator):
        assert estimator._instruction_cost({}) == 0

    def test_short(self, estimator):
        assert estimator._instruction_cost({"instruction": "abc"}) == 0


# ----------------------------------------------------------------------
# LLM layer
# ----------------------------------------------------------------------

class TestEstimateLLM:

    @pytest.mark.asyncio
    async def test_llm_success(self, estimator):
        wire_repo(estimator, make_task("coding", "do it"))
        estimator.llm_client.call.return_value = make_llm_response("medium")
        est = await estimator.estimate("t1", depth="llm")
        assert est.source == "llm:medium"
        assert est.tokens == estimator.est_cfg.category_map["medium"][0]
        assert est.p_solvable == estimator.est_cfg.category_map["medium"][1]
        assert est.confidence == estimator.est_cfg.llm_confidence

    @pytest.mark.asyncio
    @pytest.mark.parametrize("cat", ["tiny", "small", "medium", "large", "huge"])
    async def test_llm_all_categories(self, estimator, cat):
        wire_repo(estimator, make_task("coding", "do it"))
        estimator.llm_client.call.return_value = make_llm_response(cat)
        est = await estimator.estimate("t1", depth="llm")
        assert est.source == f"llm:{cat}"
        assert est.tokens == estimator.est_cfg.category_map[cat][0]

    @pytest.mark.asyncio
    async def test_llm_fallback_on_exception(self, estimator):
        wire_repo(estimator, make_task("coding", "do it"))
        estimator.llm_client.call.side_effect = RuntimeError("boom")
        est = await estimator.estimate("t1", depth="llm")
        assert est.source == "heuristic"

    @pytest.mark.asyncio
    async def test_llm_fallback_on_unparseable(self, estimator):
        wire_repo(estimator, make_task("coding", "do it"))
        estimator.llm_client.call.return_value = make_llm_response("nonsense")
        est = await estimator.estimate("t1", depth="llm")
        assert est.source == "heuristic"

    @pytest.mark.asyncio
    async def test_llm_missing_task(self, estimator):
        estimator.task_repo.get_task.return_value = None
        est = await estimator.estimate("missing", depth="llm")
        assert est.source == "heuristic_missing"

    @pytest.mark.asyncio
    async def test_unknown_depth_raises(self, estimator):
        with pytest.raises(ValueError):
            await estimator.estimate("t1", depth="strict")

    @pytest.mark.asyncio
    async def test_default_depth_is_heuristic(self, estimator):
        wire_repo(estimator, make_task("coding", ""))
        est = await estimator.estimate("t1")
        assert est.source == "heuristic"
        estimator.llm_client.call.assert_not_called()


# ----------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------

class TestCalibrate:

    def test_calibrates_base_cost(self, estimator):
        wire_repo(estimator, make_task("coding"))
        estimator.calibrate("t1", real_tokens=10000, success=True, alpha=0.5)
        # 0.5*3000 + 0.5*10000 = 6500
        assert estimator._base_costs["coding"] == 6500

    def test_calibrates_per_type_p_up(self, estimator):
        wire_repo(estimator, make_task("coding"))
        estimator.calibrate("t1", real_tokens=1000, success=True, alpha=0.5)
        # 0.5*0.65 + 0.5*1.0 = 0.825
        assert estimator.est_cfg.heuristic_p_solvable["coding"] == pytest.approx(0.825)

    def test_calibrates_per_type_p_down(self, estimator):
        wire_repo(estimator, make_task("coding"))
        estimator.calibrate("t1", real_tokens=1000, success=False, alpha=0.5)
        # 0.5*0.65 + 0.5*0.0 = 0.325
        assert estimator.est_cfg.heuristic_p_solvable["coding"] == pytest.approx(0.325)

    def test_calibrates_per_type_p_new_type(self, estimator):
        wire_repo(estimator, make_task("brand_new"))
        estimator.calibrate("t1", real_tokens=1000, success=True, alpha=0.5)
        # starts from default_p_solvable=0.6: 0.5*0.6 + 0.5*1.0 = 0.8
        assert estimator.est_cfg.heuristic_p_solvable["brand_new"] == pytest.approx(0.8)

    def test_calibrates_default_p_when_type_empty(self, estimator):
        wire_repo(estimator, make_task(""))
        estimator.calibrate("t1", real_tokens=1000, success=True, alpha=0.5)
        # 0.5*0.6 + 0.5*1.0 = 0.8
        assert estimator.est_cfg.default_p_solvable == pytest.approx(0.8)

    def test_calibrates_safety_up(self, estimator):
        wire_repo(estimator, make_task("coding"))
        # prior tokens = (3000+0+0+100)*1.2 = 3720, ratio ~2.69
        estimator.calibrate("t1", real_tokens=10000, success=True, alpha=0.5)
        s = estimator.est_cfg.heuristic_safety
        assert 1.2 < s <= 3.0

    def test_calibrates_safety_down_clamped_to_1(self, estimator):
        wire_repo(estimator, make_task("coding"))
        # ratio ~0.027 -> geometric mean would push below 1, clamps to 1
        estimator.calibrate("t1", real_tokens=100, success=True, alpha=0.5)
        assert estimator.est_cfg.heuristic_safety == 1.0

    def test_calibrates_safety_clamped_to_3(self, estimator):
        wire_repo(estimator, make_task("coding"))
        estimator.calibrate("t1", real_tokens=1_000_000, success=True, alpha=0.9)
        assert estimator.est_cfg.heuristic_safety == 3.0

    def test_missing_task_is_noop(self, estimator):
        estimator.task_repo.get_task.return_value = None
        base0 = estimator._base_costs["coding"]
        p0 = estimator.est_cfg.heuristic_p_solvable["coding"]
        s0 = estimator.est_cfg.heuristic_safety
        estimator.calibrate("missing", real_tokens=1000, success=True)
        assert estimator._base_costs["coding"] == base0
        assert estimator.est_cfg.heuristic_p_solvable["coding"] == p0
        assert estimator.est_cfg.heuristic_safety == s0

    def test_zero_tokens_skips_base_and_safety(self, estimator):
        wire_repo(estimator, make_task("coding"))
        base0 = estimator._base_costs["coding"]
        s0 = estimator.est_cfg.heuristic_safety
        estimator.calibrate("t1", real_tokens=0, success=True, alpha=0.5)
        assert estimator._base_costs["coding"] == base0
        assert estimator.est_cfg.heuristic_safety == s0
        # p_solvable is still updated
        assert estimator.est_cfg.heuristic_p_solvable["coding"] == pytest.approx(0.825)

    def test_uses_config_alpha_by_default(self, estimator):
        wire_repo(estimator, make_task("coding"))
        estimator.est_cfg.calibration_alpha = 0.1
        estimator.calibrate("t1", real_tokens=10000, success=True)
        # 0.9*3000 + 0.1*10000 = 3700
        assert estimator._base_costs["coding"] == 3700

    def test_p_solvable_clamped(self, estimator):
        wire_repo(estimator, make_task("coding"))
        estimator.est_cfg.heuristic_p_solvable["coding"] = 0.99
        estimator.calibrate("t1", real_tokens=1000, success=True, alpha=0.9)
        # would exceed 1.0 without clamping
        assert 0.0 <= estimator.est_cfg.heuristic_p_solvable["coding"] <= 1.0


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

class TestParseCategory:

    @pytest.mark.parametrize("text,expected", [
        ("tiny", "tiny"),
        ("Tiny", "tiny"),
        ("  large  ", "large"),
        ("medium.", "medium"),
        ("I'd say huge", "huge"),
        ("nonsense", None),
        ("", None),
        ("largeish", None),
        ("smaller", None),
    ])
    def test_parse(self, estimator, text, expected):
        assert estimator._parse_category(text) == expected


class TestTruncate:

    def test_short_text_unchanged(self, estimator):
        assert estimator._truncate("hello", 10) == "hello"

    def test_long_text_truncated(self, estimator):
        out = estimator._truncate("hello world", 5)
        assert out.startswith("hello")
        assert "[truncated]" in out


class TestListWorkspaceFiles:

    def test_none_dir(self, estimator):
        assert "no task directory" in estimator._list_workspace_files(None)

    def test_missing_workspace(self, estimator, tmp_path):
        assert "not yet created" in estimator._list_workspace_files(tmp_path)

    def test_empty_workspace(self, estimator, tmp_path):
        (tmp_path / "workspace").mkdir()
        assert "empty" in estimator._list_workspace_files(tmp_path)

    def test_lists_files_with_sizes(self, estimator, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()
        (ws / "a.txt").write_text("x" * 42)
        out = estimator._list_workspace_files(tmp_path)
        assert "a.txt" in out
        assert "42 bytes" in out

    def test_cap_on_file_count(self, estimator, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()
        for i in range(50):
            (ws / f"f{i}.txt").write_text("x")
        estimator.est_cfg.llm_prompt_files_cap = 5
        out = estimator._list_workspace_files(tmp_path)
        assert "more files omitted" in out