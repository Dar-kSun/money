"""The data panel: aligned (dates x instruments) matrices per field."""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Iterable, Mapping, Sequence

import numpy as np

GROUP_FIELDS = ("market", "sector", "industry", "subindustry")


@dataclass
class Panel:
    """Every field is a float array of shape ``(len(dates), len(instruments))``.

    Rows are trading days in ascending order, columns are instruments in a
    fixed order. ``universe`` marks which instruments are tradable on each day
    (Brain's TOP3000 changes membership over time, and an alpha must not take
    positions outside it). ``groups`` holds integer codes used by the group
    operators; -1 means unknown.
    """

    dates: np.ndarray                      # datetime64[D], shape (T,)
    instruments: list[str]
    fields: dict[str, np.ndarray] = dc_field(default_factory=dict)
    groups: dict[str, np.ndarray] = dc_field(default_factory=dict)
    universe: np.ndarray | None = None     # bool, shape (T, N)

    # ------------------------------------------------------------- geometry
    @property
    def shape(self) -> tuple[int, int]:
        return len(self.dates), len(self.instruments)

    @property
    def n_dates(self) -> int:
        return len(self.dates)

    @property
    def n_instruments(self) -> int:
        return len(self.instruments)

    def __post_init__(self) -> None:
        self.dates = np.asarray(self.dates)
        if self.universe is None:
            self.universe = np.ones(self.shape, dtype=bool)
        self._validate()

    def _validate(self) -> None:
        expected = self.shape
        for name, array in list(self.fields.items()):
            arr = np.asarray(array, dtype=float)
            if arr.shape != expected:
                raise ValueError(
                    f"field {name!r} has shape {arr.shape}, expected {expected}"
                )
            self.fields[name] = arr
        for name, array in list(self.groups.items()):
            arr = np.asarray(array)
            if arr.shape != expected:
                raise ValueError(
                    f"group {name!r} has shape {arr.shape}, expected {expected}"
                )
            self.groups[name] = arr.astype(np.int64)
        if self.universe.shape != expected:
            raise ValueError(
                f"universe has shape {self.universe.shape}, expected {expected}"
            )

    # ---------------------------------------------------------------- access
    def has(self, name: str) -> bool:
        return name in self.fields or name in self.groups

    def get(self, name: str) -> np.ndarray:
        if name in self.fields:
            return self.fields[name]
        if name in self.groups:
            return self.groups[name].astype(float)
        raise KeyError(
            f"field {name!r} is not in the panel; available: "
            f"{', '.join(sorted(self.field_names))}"
        )

    @property
    def field_names(self) -> list[str]:
        return sorted(set(self.fields) | set(self.groups))

    def add_field(self, name: str, values: np.ndarray) -> None:
        arr = np.asarray(values, dtype=float)
        if arr.shape != self.shape:
            raise ValueError(f"{name!r} has shape {arr.shape}, expected {self.shape}")
        self.fields[name] = arr

    # ------------------------------------------------------------- universe
    def top_by(self, field: str, count: int) -> np.ndarray:
        """A universe mask of the ``count`` largest instruments by ``field``.

        Brain's TOP3000 is liquidity-ranked and re-selected over time, so the
        mask is computed per day rather than once.
        """
        values = self.get(field)
        mask = np.zeros(self.shape, dtype=bool)
        for t in range(self.n_dates):
            row = values[t]
            valid = np.flatnonzero(~np.isnan(row))
            if valid.size == 0:
                continue
            take = valid[np.argsort(-row[valid], kind="stable")[:count]]
            mask[t, take] = True
        return mask

    def restrict_universe(self, mask: np.ndarray) -> None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != self.shape:
            raise ValueError(f"mask has shape {mask.shape}, expected {self.shape}")
        self.universe = mask

    def slice_dates(self, start: int = 0, stop: int | None = None) -> "Panel":
        stop = self.n_dates if stop is None else stop
        sl = slice(start, stop)
        return Panel(
            dates=self.dates[sl],
            instruments=list(self.instruments),
            fields={k: v[sl] for k, v in self.fields.items()},
            groups={k: v[sl] for k, v in self.groups.items()},
            universe=self.universe[sl],
        )

    # --------------------------------------------------------------- returns
    def forward_returns(self, price_field: str = "close") -> np.ndarray:
        """Close-to-close return realised *from* day t to day t+1.

        Row ``t`` is the return earned by a position held at the close of day
        ``t``, so pairing weights at ``t`` with this row introduces no
        look-ahead. The last row is NaN.
        """
        price = self.get(price_field)
        out = np.full(self.shape, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[:-1] = price[1:] / price[:-1] - 1.0
        out[~np.isfinite(out)] = np.nan
        return out

    @classmethod
    def from_mapping(
        cls,
        dates: Sequence,
        instruments: Sequence[str],
        fields: Mapping[str, np.ndarray],
        groups: Mapping[str, np.ndarray] | None = None,
        universe: np.ndarray | None = None,
    ) -> "Panel":
        return cls(
            dates=np.asarray(dates),
            instruments=list(instruments),
            fields={k: np.asarray(v, dtype=float) for k, v in fields.items()},
            groups={k: np.asarray(v) for k, v in (groups or {}).items()},
            universe=universe,
        )
