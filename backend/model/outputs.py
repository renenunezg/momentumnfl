from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite


def _validate_identity(abbr: str, name: str, label: str) -> None:
    if not abbr.strip():
        raise ValueError(f"{label}_abbr must not be empty")
    if not name.strip():
        raise ValueError(f"{label} must not be empty")


def _validate_as_of(as_of: datetime) -> None:
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("as_of must be timezone-aware")


def _validate_finite(**values: float) -> None:
    for name, value in values.items():
        if not isfinite(value):
            raise ValueError(f"{name} must be finite")


@dataclass(frozen=True, slots=True)
class TeamRating:
    """Weekly team strength on a neutral field against an average team.

    Offense is points scored above average.
    Defense is points prevented above average, so higher is better.
    """

    season: int
    week: int
    as_of: datetime
    model_version: str
    team_abbr: str
    team: str
    offense_points: float
    defense_points: float
    expected_drives: float
    power_rating_sd: float

    def __post_init__(self) -> None:
        _validate_identity(self.team_abbr, self.team, "team")
        _validate_as_of(self.as_of)
        _validate_finite(
            offense_points=self.offense_points,
            defense_points=self.defense_points,
            expected_drives=self.expected_drives,
            power_rating_sd=self.power_rating_sd,
        )
        if self.season < 1920:
            raise ValueError("season is invalid")
        if self.week < 0:
            raise ValueError("week must not be negative")
        if not self.model_version.strip():
            raise ValueError("model_version must not be empty")
        if self.expected_drives <= 0:
            raise ValueError("expected_drives must be positive")
        if self.power_rating_sd < 0:
            raise ValueError("power_rating_sd must not be negative")

    @property
    def power_rating(self) -> float:
        return self.offense_points + self.defense_points

    @property
    def scoring_environment(self) -> float:
        return self.offense_points - self.defense_points

    def to_record(self) -> dict[str, object]:
        return {
            "season": self.season,
            "week": self.week,
            "as_of": self.as_of.astimezone(UTC).isoformat(),
            "model_version": self.model_version,
            "team_abbr": self.team_abbr,
            "team": self.team,
            "offense_points": self.offense_points,
            "defense_points": self.defense_points,
            "power_rating": self.power_rating,
            "scoring_environment": self.scoring_environment,
            "expected_drives": self.expected_drives,
            "power_rating_sd": self.power_rating_sd,
        }


@dataclass(frozen=True, slots=True)
class GameProjection:
    """Pregame score distribution and derived spread and total outputs.

    expected points already include the QB, rest, and market-blend layers, so
    home_margin is the published (blended) number; pure_home_margin preserves
    the model's own opinion (QB and rest applied, market not).
    home_margin is positive when the home team is favored.
    home_spread uses sportsbook notation and therefore has the opposite sign.
    """

    season: int
    week: int
    as_of: datetime
    model_version: str
    game_id: str
    home_team_abbr: str
    home_team: str
    away_team_abbr: str
    away_team: str
    neutral_site: bool
    home_field_points: float
    expected_home_points: float
    expected_away_points: float
    margin_sd: float
    total_sd: float
    margin_total_correlation: float
    degrees_of_freedom: float
    start_date: datetime | None = None
    div_game: bool | None = None
    home_qb_adjustment: float = 0.0
    away_qb_adjustment: float = 0.0
    rest_adjustment: float = 0.0
    pure_home_margin: float | None = None
    market_home_spread: float | None = None
    market_weight: float = 0.0

    def __post_init__(self) -> None:
        if not self.game_id.strip():
            raise ValueError("game_id must not be empty")
        _validate_identity(self.home_team_abbr, self.home_team, "home_team")
        _validate_identity(self.away_team_abbr, self.away_team, "away_team")
        _validate_as_of(self.as_of)
        _validate_finite(
            home_field_points=self.home_field_points,
            expected_home_points=self.expected_home_points,
            expected_away_points=self.expected_away_points,
            margin_sd=self.margin_sd,
            total_sd=self.total_sd,
            margin_total_correlation=self.margin_total_correlation,
            degrees_of_freedom=self.degrees_of_freedom,
        )
        if self.season < 1920:
            raise ValueError("season is invalid")
        if self.week < 0:
            raise ValueError("week must not be negative")
        if not self.model_version.strip():
            raise ValueError("model_version must not be empty")
        if self.home_team_abbr == self.away_team_abbr:
            raise ValueError("home and away teams must differ")
        if self.expected_home_points < 0 or self.expected_away_points < 0:
            raise ValueError("expected points must not be negative")
        if self.margin_sd <= 0 or self.total_sd <= 0:
            raise ValueError("distribution scales must be positive")
        if not -1 < self.margin_total_correlation < 1:
            raise ValueError("margin_total_correlation must be between -1 and 1")
        if self.degrees_of_freedom <= 2:
            raise ValueError("degrees_of_freedom must exceed 2")
        if self.neutral_site and abs(self.home_field_points) > 1e-9:
            raise ValueError(
                "neutral-site projections cannot include home-field points"
            )
        if not 0 <= self.market_weight <= 0.5:
            raise ValueError("market_weight must be in [0, 0.5]")

    @property
    def home_margin(self) -> float:
        return self.expected_home_points - self.expected_away_points

    @property
    def home_spread(self) -> float:
        return -self.home_margin

    @property
    def model_total(self) -> float:
        return self.expected_home_points + self.expected_away_points

    @property
    def pure_home_spread(self) -> float | None:
        if self.pure_home_margin is None:
            return None
        return -self.pure_home_margin

    def to_record(self) -> dict[str, object]:
        return {
            "season": self.season,
            "week": self.week,
            "as_of": self.as_of.astimezone(UTC).isoformat(),
            "model_version": self.model_version,
            "game_id": self.game_id,
            "start_date": (
                None
                if self.start_date is None
                else self.start_date.astimezone(UTC).isoformat()
            ),
            "home_team_abbr": self.home_team_abbr,
            "home_team": self.home_team,
            "away_team_abbr": self.away_team_abbr,
            "away_team": self.away_team,
            "neutral_site": self.neutral_site,
            "div_game": self.div_game,
            "home_field_points": self.home_field_points,
            "expected_home_points": self.expected_home_points,
            "expected_away_points": self.expected_away_points,
            "home_qb_adjustment": self.home_qb_adjustment,
            "away_qb_adjustment": self.away_qb_adjustment,
            "rest_adjustment": self.rest_adjustment,
            "pure_home_margin": self.pure_home_margin,
            "pure_home_spread": self.pure_home_spread,
            "market_home_spread": self.market_home_spread,
            "market_weight": self.market_weight,
            "home_margin": self.home_margin,
            "home_spread": self.home_spread,
            "model_total": self.model_total,
            "margin_sd": self.margin_sd,
            "total_sd": self.total_sd,
            "margin_total_correlation": self.margin_total_correlation,
            "distribution": "bivariate_student_t",
            "degrees_of_freedom": self.degrees_of_freedom,
        }
