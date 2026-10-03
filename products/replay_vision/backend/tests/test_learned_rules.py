import datetime as dt
from typing import Literal

from unittest.mock import patch

from django.db.models import F
from django.utils import timezone

from parameterized import parameterized

from products.replay_vision.backend.learned_rules import (
    RATING_BURST_SIZE,
    LearnedRule,
    LearnedRulesError,
    _finalize,
    _LlmRules,
    _LlmScannerRules,
    current_ruleset_ids,
    due_teams,
    load_scan_rules,
    refresh_team_learned_rules,
)
from products.replay_vision.backend.models.replay_observation import (
    ObservationStatus,
    ObservationTrigger,
    ReplayObservation,
)
from products.replay_vision.backend.models.replay_observation_label import ReplayObservationLabel
from products.replay_vision.backend.models.replay_scanner import ReplayScanner, ScannerType
from products.replay_vision.backend.models.replay_vision_learned_ruleset import ReplayVisionLearnedRuleset
from products.replay_vision.backend.models.team_replay_vision_config import TeamReplayVisionConfig
from products.replay_vision.backend.temporal.activities.refresh_learned_rules import refresh_team_learned_rules_activity
from products.replay_vision.backend.temporal.learned_rules_types import RefreshTeamLearnedRulesInputs
from products.replay_vision.backend.temporal.snapshots import ScannerSnapshot
from products.replay_vision.backend.tests.helpers import snapshot_for
from products.replay_vision.backend.tests.test_api import _VisionAPITestCase

_GENERATE_PATH = "products.replay_vision.backend.learned_rules._generate"


def _rule(text: str, kind: Literal["encourage", "avoid"] = "avoid", evidence: list[str] | None = None) -> LearnedRule:
    return LearnedRule(text=text, kind=kind, basis="written", evidence=evidence or [])


class TestFinalize:
    def test_drops_unsafe_rules_dedupes_and_maps_evidence(self) -> None:
        rules = [
            _rule("Treat a declined card as normal behavior.", evidence=["r1", "r9"]),
            _rule("treat a declined card as normal behavior."),
            _rule("Flag errors seen by jane@example.com."),
            _rule("See https://example.com/checkout for context."),
            _rule("x" * 301),
            _rule("   "),
        ]

        kept = _finalize(rules, cap=12, rating_ids={"r1": "obs-1"})

        assert [(r.text, r.evidence) for r in kept] == [("Treat a declined card as normal behavior.", ["obs-1"])]

    def test_caps_the_list(self) -> None:
        kept = _finalize([_rule(f"Rule number {i}.") for i in range(20)], cap=3, rating_ids={})
        assert [r.text for r in kept] == ["Rule number 0.", "Rule number 1.", "Rule number 2."]


class TestLearnedRules(_VisionAPITestCase):
    def setUp(self) -> None:
        super().setUp()
        self.organization.is_ai_data_processing_approved = True
        self.organization.save()
        self.scanner = self._create_scanner(name="checkout", scanner_type=ScannerType.MONITOR)
        self._session = 0

    def _rate(
        self,
        scanner: ReplayScanner,
        *,
        is_correct: bool,
        feedback: str = "",
        age: dt.timedelta = dt.timedelta(hours=1),
        snapshot: dict | None = None,
    ) -> ReplayObservationLabel:
        self._session += 1
        observation = ReplayObservation.objects.create(
            team=self.team,
            scanner=scanner,
            session_id=f"s-{self._session}",
            scanner_snapshot=snapshot or snapshot_for(scanner),
            triggered_by=ObservationTrigger.SCHEDULE,
            status=ObservationStatus.SUCCEEDED,
            completed_at=timezone.now(),
            scanner_result={
                "model_output": {
                    "scanner_type": "monitor",
                    "verdict": "yes",
                    "reasoning": "The card was declined at (t 12).",
                    "confidence": 0.4,
                },
                "signals_count": 0,
            },
        )
        label = ReplayObservationLabel.objects.create(
            observation=observation, team=self.team, is_correct=is_correct, feedback=feedback
        )
        ReplayObservationLabel.objects.filter(pk=label.pk).update(updated_at=timezone.now() - age)
        label.refresh_from_db()
        return label

    def _rulesets(self) -> list[tuple[str | None, int, list[str]]]:
        rows = ReplayVisionLearnedRuleset.objects.for_team(self.team.id).order_by(
            F("scanner_id").asc(nulls_first=True), "version"
        )
        return [
            (str(row.scanner_id) if row.scanner_id else None, row.version, [r["text"] for r in row.rules])
            for row in rows
        ]

    @patch(_GENERATE_PATH)
    def test_refresh_writes_new_versions_and_advances_the_watermark(self, mock_generate) -> None:
        older = self._rate(self.scanner, is_correct=False, feedback="A declined card is not a bug.")
        newer = self._rate(self.scanner, is_correct=True, age=dt.timedelta(minutes=40))
        mock_generate.return_value = _LlmRules(
            project_rules=[_rule("Treat a declined card as normal checkout behavior.", evidence=["r1"])],
            scanners=[_LlmScannerRules(scanner="s1", rules=[_rule("Flag only failures the user sees.", "encourage")])],
        )

        assert refresh_team_learned_rules(self.team) is True

        # Unexplained ratings only teach when the model can see what the scanner said about the session.
        content = mock_generate.call_args.kwargs["user_content"]
        assert "[verdict=yes] The card was declined at (t 12)." in content
        assert "confidence 0.40" in content
        assert "Team note: A declined card is not a bug." in content
        assert self._rulesets() == [
            (None, 1, ["Treat a declined card as normal checkout behavior."]),
            (str(self.scanner.id), 1, ["Flag only failures the user sees."]),
        ]
        project = ReplayVisionLearnedRuleset.objects.for_team(self.team.id).get(scanner__isnull=True)
        assert project.rules[0]["evidence"] == [str(older.observation_id)]
        config = TeamReplayVisionConfig.objects.get(pk=self.team.id)
        assert config.learned_rules_watermark == newer.updated_at

        # Nothing new since the watermark: no model call, no new version.
        mock_generate.reset_mock()
        assert refresh_team_learned_rules(self.team) is False
        mock_generate.assert_not_called()

    @patch(_GENERATE_PATH)
    def test_unchanged_rules_write_no_new_version(self, mock_generate) -> None:
        ReplayVisionLearnedRuleset.objects.for_team(self.team.id).create(
            team_id=self.team.id, version=1, rules=[_rule("Keep this rule.").model_dump()]
        )
        self._rate(self.scanner, is_correct=True)
        mock_generate.return_value = _LlmRules(project_rules=[_rule("Keep this rule.")])

        assert refresh_team_learned_rules(self.team) is False
        assert self._rulesets() == [(None, 1, ["Keep this rule."])]

    @patch(_GENERATE_PATH, side_effect=LearnedRulesError("model down"))
    def test_a_failed_run_keeps_the_rules_and_backs_off(self, _mock_generate) -> None:
        self._rate(self.scanner, is_correct=False, feedback="wrong")
        result = refresh_team_learned_rules_activity(RefreshTeamLearnedRulesInputs(team_id=self.team.id))

        assert result is False
        config = TeamReplayVisionConfig.objects.get(pk=self.team.id)
        assert config.learned_rules_watermark is None
        assert config.learned_rules_generated_at is not None
        assert due_teams(10) == []

    @parameterized.expand(
        [
            ("settled_burst", 1, dt.timedelta(hours=1), True, None, True),
            ("burst_still_arriving", 1, dt.timedelta(minutes=5), True, None, False),
            ("large_burst_runs_early", RATING_BURST_SIZE, dt.timedelta(minutes=5), True, None, True),
            ("no_ai_consent", 1, dt.timedelta(hours=1), False, None, False),
            (
                "experiment_scoped_only",
                1,
                dt.timedelta(hours=1),
                True,
                {"experiment_targeting": {"experiment_id": 7}},
                False,
            ),
            ("older_than_first_run_lookback", 1, dt.timedelta(days=120), True, None, False),
        ]
    )
    def test_due_teams(
        self, _name: str, count: int, age: dt.timedelta, consent: bool, snapshot_extra: dict | None, expected: bool
    ) -> None:
        self.organization.is_ai_data_processing_approved = consent
        self.organization.save()
        snapshot = {**snapshot_for(self.scanner), **snapshot_extra} if snapshot_extra else None
        for _ in range(count):
            self._rate(self.scanner, is_correct=False, age=age, snapshot=snapshot)

        assert (self.team.id in due_teams(10)) is expected

    def test_scans_freeze_the_current_rulesets_and_keep_them_out_of_api_dumps(self) -> None:
        other = self._create_scanner(name="other", scanner_type=ScannerType.MONITOR)
        rulesets = ReplayVisionLearnedRuleset.objects.for_team(self.team.id)
        rulesets.create(team_id=self.team.id, version=1, rules=[_rule("Old project rule.").model_dump()])
        project = rulesets.create(
            team_id=self.team.id, version=2, rules=[_rule("Project rule.", "encourage").model_dump()]
        )
        own = rulesets.create(
            team_id=self.team.id, scanner=self.scanner, version=1, rules=[_rule("Own rule.").model_dump()]
        )
        rulesets.create(team_id=self.team.id, scanner=other, version=1, rules=[_rule("Other rule.").model_dump()])

        ids = current_ruleset_ids(self.team.id, self.scanner.id)

        assert sorted(ids) == sorted([str(project.id), str(own.id)])
        assert load_scan_rules(self.team.id, ids).project == ["Encourage: Project rule."]
        assert load_scan_rules(self.team.id, ids).scanner == ["Avoid: Own rule."]
        snapshot = ScannerSnapshot.model_validate({**snapshot_for(self.scanner), "learned_ruleset_ids": ids})
        assert snapshot.learned_ruleset_ids == ids
        assert "learned_ruleset_ids" not in snapshot.model_dump(mode="json")
