"""Per-position player trait ratings and the 20 to 80 grade.

``TRAITS`` is keyed by trait name. A trait that several position groups share
(a havoc rate across DL, LB and DB, say) is one spec listing every group in
``position_groups``. The position modules fill the registry as they are
imported, so importing ``backend.players.traits`` completes it.
"""

from dataclasses import dataclass

MODEL_VERSION = "nfl_position_traits_v1"
AGGREGATES = frozenset({"rate", "quantile20", "share_of_games", "latest"})
EPA_TRAIT = "epa.credit"
UNIT_PREFIX = "unit:"
# Roster and box-score positions to the groups the grade rates.
POSITION_GROUPS = {
    "QB": "QB",
    "RB": "RB",
    "HB": "RB",
    "FB": "RB",
    "WR": "WR",
    "TE": "TE",
    "T": "OL",
    "OT": "OL",
    "G": "OL",
    "C": "OL",
    "OL": "OL",
    "DE": "DL",
    "DT": "DL",
    "NT": "DL",
    "DL": "DL",
    "LB": "LB",
    "OLB": "LB",
    "ILB": "LB",
    "MLB": "LB",
    "CB": "DB",
    "S": "DB",
    "FS": "DB",
    "SS": "DB",
    "SAF": "DB",
    "DB": "DB",
    "K": "K",
    "P": "P",
}
GROUP_ORDER = ("QB", "RB", "WR", "TE", "OL", "DL", "LB", "DB", "K", "P")
CREDITED_GROUPS = ("QB", "RB", "WR", "TE", "DL", "LB", "DB")


@dataclass(frozen=True)
class TraitSpec:
    name: str
    position_groups: tuple[str, ...]
    kind: str  # "counted" or "inferred"
    aggregate: str = "rate"  # "rate" | "quantile20" | "share_of_games" | "latest"
    higher_is_better: bool | None = True

    def __post_init__(self) -> None:
        if self.aggregate not in AGGREGATES:
            raise ValueError(
                f"trait {self.name!r} has unknown aggregate {self.aggregate!r}"
            )
        if self.kind not in ("counted", "inferred"):
            raise ValueError(f"trait {self.name!r} has unknown kind {self.kind!r}")


TRAITS: dict[str, TraitSpec] = {}


def register(spec: TraitSpec) -> TraitSpec:
    if spec.name in TRAITS:
        raise ValueError(f"trait {spec.name!r} is already registered")
    TRAITS[spec.name] = spec
    return spec


# The player's opponent-adjusted competitive EPA credit per credited game. It is
# built by ``backend.players.credit`` from tagged plays, so unlike the other
# traits it is not owned by a position module; every credited group reads it.
register(TraitSpec(EPA_TRAIT, CREDITED_GROUPS, "counted"))
