import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from project.config import Config
from project.llm_client import LLMClient
from project.task_repo import TaskRepo


@dataclass
class Estimate:
    """Result of estimating a single task."""

    tokens: int          # normalized input-equivalent tokens
    p_solvable: float    # rough probability of successful solve [0..1]
    source: str          # "heuristic" | "llm:<cat>"
    confidence: float    # rough confidence in [0..1]


@dataclass
class EstimatorConfig:
    """Tunable parameters for ``TaskEstimator``."""

    # Per-task-type base costs; recalibrated at runtime via ``calibrate``.
    # Values include the fixed per-run overhead (system prompt + tool
    # schemas ≈ 1500 normalized tokens for one LLM call).
    base_costs: dict[str, int] = field(
        default_factory=lambda: {
            "coding": 4000,
            "math": 2500,
            "data_analysis": 3000,
            "search": 2000,
        }
    )
    default_base: int = 2500

    # Category -> (normalized tokens, p_solvable).
    # Tokens are total budgets, including the fixed per-task overhead.
    category_map: dict[str, tuple[int, float]] = field(
        default_factory=lambda: {
            "tiny": (2500, 0.95),
            "small": (4000, 0.90),
            "medium": (8000, 0.75),
            "large": (16000, 0.55),
            "huge": (30000, 0.40),
        }
    )
    category_order: tuple[str, ...] = ("tiny", "small", "medium", "large", "huge")

    # Heuristic p_solvable per task type.
    heuristic_p_solvable: dict[str, float] = field(
        default_factory=lambda: {
            "coding": 0.65,
            "math": 0.75,
            "data_analysis": 0.70,
            "search": 0.85,
        }
    )
    default_p_solvable: float = 0.60

    # Safety factors.
    heuristic_safety: float = 1.2
    workspace_read_cap: int = 8000

    # LLM classification layer settings.
    llm_prompt_instruction_cap: int = 2000
    llm_prompt_files_cap: int = 30
    llm_confidence: float = 0.7
    heuristic_confidence: float = 0.4

    # Calibration.
    calibration_alpha: float = 0.3


class TaskEstimator:
    """Estimates token cost and solvability of tasks in two layers.

    Layers (cheapest to most expensive):
      * heuristic - no LLM calls, deterministic, uses local metadata
      * llm       - one short classification call

    All token values are expressed in normalized "input-equivalent" tokens
    (the same scale used by ``Config.normalize_tokens``).
    """

    def __init__(
        self,
        task_repo: TaskRepo,
        llm_client: LLMClient,
        config: Config,
        estimator_config: Optional[EstimatorConfig] = None,
    ) -> None:
        self.task_repo = task_repo
        self.llm_client = llm_client
        self.config = config
        self.est_cfg = estimator_config or EstimatorConfig()
        self._base_costs: dict[str, int] = dict(self.est_cfg.base_costs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def estimate(self, task_id: str, depth: str = "heuristic") -> Estimate:
        """Estimate a task at the requested depth.

        Args:
            task_id: target task.
            depth: one of ``"heuristic"`` or ``"llm"``.

        Returns:
            An ``Estimate`` with normalized tokens, p_solvable, source and
            confidence. Falls back to the heuristic layer if the LLM call
            fails.
        """
        if depth == "heuristic":
            return self._heuristic_estimate(task_id)

        if depth == "llm":
            llm_est = await self._llm_estimate(task_id)
            return llm_est if llm_est is not None else self._heuristic_estimate(task_id)

        raise ValueError(f"Unknown estimation depth: {depth!r}")

    def calibrate(
        self,
        task_id: str,
        real_tokens: int,
        success: bool,
        alpha: Optional[float] = None,
     ) -> None:
        """Update the estimator's parameters from an observed solve attempt.

        Updates:
        * per-type base cost            (EMA on observed real token count)
        * per-type heuristic p_solvable (EMA toward 1.0 on success, 0.0 on failure)
        * global default p_solvable     (EMA on success for unknown types)
        * global heuristic safety factor(geometric EMA on real/prior ratio)

        The prior estimate is snapshotted before any mutation so the safety
        factor can react to systematic under- or over-estimation.
        """
        task = self.task_repo.get_task(task_id)
        if task is None:
            return

        task_type = task.get("task_type", "")
        a = self.est_cfg.calibration_alpha if alpha is None else alpha

        # Snapshot the current estimate BEFORE mutating any config fields.
        prior_tokens = self._heuristic_estimate(task_id).tokens

        # 1. Per-type base cost (linear EMA on the observed real token count).
        if real_tokens > 0:
            current_base = self._base_costs.get(task_type, self.est_cfg.default_base)
            updated_base = (1.0 - a) * current_base + a * real_tokens
            self._base_costs[task_type] = int(updated_base)

        # 2. Per-type solvability (EMA toward 1.0 on success, 0.0 on failure).
        target_p = 1.0 if success else 0.0
        if task_type:
            current_p = self.est_cfg.heuristic_p_solvable.get(
                task_type, self.est_cfg.default_p_solvable
            )
            updated_p = (1.0 - a) * current_p + a * target_p
            self.est_cfg.heuristic_p_solvable[task_type] = max(0.0, min(1.0, updated_p))
        else:
            # Unknown type: adapt the global fallback instead.
            updated_p = (1.0 - a) * self.est_cfg.default_p_solvable + a * target_p
            self.est_cfg.default_p_solvable = max(0.0, min(1.0, updated_p))

        # 3. Global safety factor (geometric EMA on real/prior ratio, clamped).
        #    Using geometric mean keeps multiplicative corrections stable and
        #    prevents the factor from blowing up or collapsing.
        if real_tokens > 0 and prior_tokens > 0:
            ratio = max(real_tokens / prior_tokens, 1e-3)
            current_s = self.est_cfg.heuristic_safety
            updated_s = (current_s ** (1.0 - a)) * (ratio ** a)
            self.est_cfg.heuristic_safety = max(1.0, min(3.0, updated_s))

    @property
    def base_costs(self) -> dict[str, int]:
        return dict(self._base_costs)

    # ------------------------------------------------------------------
    # Layer 0: heuristic
    # ------------------------------------------------------------------

    def _heuristic_estimate(self, task_id: str) -> Estimate:
        """Cheap, deterministic estimate based on local metadata only."""
        task = self.task_repo.get_task(task_id)
        task_dir = self.task_repo.get_task_dir(task_id)

        if task is None:
            return Estimate(
                tokens=self.est_cfg.default_base,
                p_solvable=self.est_cfg.default_p_solvable,
                source="heuristic_missing",
                confidence=0.1,
            )

        task_type = task.get("task_type", "")
        base = self._base_costs.get(task_type, self.est_cfg.default_base)
        total = (
            base
            + self._instruction_cost(task)
            + self._workspace_cost(task_dir)
            + self._check_bonus(task)
        )
        total = max(int(total * self.est_cfg.heuristic_safety), 100)

        return Estimate(
            tokens=total,
            p_solvable=self._heuristic_p_solvable(task_type),
            source="heuristic",
            confidence=self.est_cfg.heuristic_confidence,
        )

    def _instruction_cost(self, task: dict) -> int:
        """Rough token cost of consuming the instruction text."""
        instruction = task.get("instruction", "")
        return len(instruction) // 4

    def _workspace_cost(self, task_dir: Optional[Path]) -> int:
        """Rough token cost of reading all files in the workspace."""
        if task_dir is None:
            return 0
        workspace = task_dir / "workspace"
        if not workspace.is_dir():
            return 0
        total_bytes = 0
        try:
            for f in workspace.rglob("*"):
                if f.is_file():
                    total_bytes += f.stat().st_size
        except OSError:
            return 0
        return min(total_bytes // 4, self.est_cfg.workspace_read_cap)

    def _check_bonus(self, task: dict) -> int:
        """Additional cost implied by the expected answer format."""
        check = task.get("check", "exact_str")
        if check == "unittest":
            return 1000
        if check == "exact_int":
            return 200
        return 100

    def _heuristic_p_solvable(self, task_type: str) -> float:
        return self.est_cfg.heuristic_p_solvable.get(
            task_type, self.est_cfg.default_p_solvable
        )

    # ------------------------------------------------------------------
    # Layer 1: single LLM classification
    # ------------------------------------------------------------------

    async def _llm_estimate(self, task_id: str) -> Optional[Estimate]:
        """One short LLM call returning a difficulty category."""
        task = self.task_repo.get_task(task_id)
        if task is None:
            return None
        task_dir = self.task_repo.get_task_dir(task_id)

        prompt = self._build_classification_prompt(task, task_dir)
        messages = [
            {"role": "system", "content": "You classify task difficulty. Reply with exactly one word."},
            {"role": "user", "content": prompt},
        ]
        content = await self._safe_llm_call(messages)
        if content is None:
            return None

        category = self._parse_category(content)
        if category is None:
            return None

        tokens, p_solvable = self.est_cfg.category_map[category]
        return Estimate(
            tokens=tokens,
            p_solvable=p_solvable,
            source=f"llm:{category}",
            confidence=self.est_cfg.llm_confidence,
        )

    def _build_classification_prompt(self, task: dict, task_dir: Optional[Path]) -> str:
        instruction = self._truncate(
            task.get("instruction", ""), self.est_cfg.llm_prompt_instruction_cap
        )
        files_info = self._list_workspace_files(task_dir)
        return (
            f"Task type: {task.get('task_type', '')}\n"
            f"Instruction:\n{instruction}\n\n"
            f"Workspace files:\n{files_info}\n\n"
            "Estimate the LLM agent's cost (tool calls, reading, reasoning).\n"
            "Pick one: tiny | small | medium | large | huge\n"
            "Reply with one word."
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    async def _safe_llm_call(self, messages: list[dict]) -> Optional[str]:
        """Invoke the LLM without tools and return the raw content, or None."""
        try:
            response = await self.llm_client.call(messages, tools=[])
        except Exception:
            return None
        return getattr(response, "content", "") or ""

    def _list_workspace_files(self, task_dir: Optional[Path]) -> str:
        """List file names and sizes in the workspace (names only, no contents)."""
        if task_dir is None:
            return "(no task directory)"
        workspace = task_dir / "workspace"
        if not workspace.is_dir():
            return "(workspace not yet created)"
        lines: list[str] = []
        try:
            for f in sorted(workspace.rglob("*")):
                if f.is_file():
                    rel = f.relative_to(workspace)
                    lines.append(f"  {rel} ({f.stat().st_size} bytes)")
                    if len(lines) >= self.est_cfg.llm_prompt_files_cap:
                        lines.append("  ... (more files omitted)")
                        break
        except OSError:
            return "(error listing workspace)"
        return "\n".join(lines) if lines else "(empty workspace)"

    def _parse_category(self, text: str) -> Optional[str]:
        """Extract a difficulty category from free-form model output."""
        lowered = text.strip().lower()
        for category in self.est_cfg.category_order:
            if re.search(rf"\b{category}\b", lowered):
                return category
        return None

    def _truncate(self, text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[:limit] + "\n...[truncated]"