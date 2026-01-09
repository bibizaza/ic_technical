"""
Historical data tracker for Market Compass.

Stores weekly snapshots per asset for:
1. Subtitle generation context
2. Streamlit evolution charts
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class WeeklySnapshot:
    """Single week's data for an asset."""
    date: str  # "2025-12-28"
    dmas: int
    technical_score: int
    momentum_score: int
    price_vs_50ma_pct: float
    price_vs_100ma_pct: float
    price_vs_200ma_pct: float
    rating: str


class HistoryTracker:
    """
    Tracks historical data for Market Compass assets.
    Stores last 52 weeks (1 year) per asset.

    Storage path priority:
    1. Explicit storage_path parameter
    2. IC_HISTORY_PATH environment variable
    3. Dropbox folder (auto-detected)
    4. Local project folder (fallback)
    """

    def __init__(self, storage_path: str = None):
        if storage_path is None:
            # Check environment variable first
            storage_path = os.environ.get("IC_HISTORY_PATH")

        if storage_path is None:
            # Try to find Dropbox folder (Tools_In_Construction/ic/score_history)
            dropbox_path = self._find_dropbox_path()
            if dropbox_path:
                storage_path = os.path.join(dropbox_path, "history.json")
                print(f"[HistoryTracker] Using Dropbox: {storage_path}")

        if storage_path is None:
            # Fall back to local project directory
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            storage_path = os.path.join(base_dir, "data", "history.json")
            os.makedirs(os.path.dirname(storage_path), exist_ok=True)

        self.storage_path = storage_path
        self.data: Dict[str, List[dict]] = self._load()
        print(f"[HistoryTracker] Initialized with path: {self.storage_path}")
        print(f"[HistoryTracker] Loaded {len(self.data)} assets")

    def _find_dropbox_path(self) -> Optional[str]:
        """Auto-detect Dropbox folder and return IC score_history subfolder."""
        import sys

        # Common Dropbox root locations
        if sys.platform == "win32":
            # Windows
            dropbox_roots = [
                os.path.expandvars(r"%USERPROFILE%\Dropbox"),
                os.path.expandvars(r"%USERPROFILE%\Dropbox (Personal)"),
            ]
        else:
            # macOS / Linux
            # macOS CloudStorage location (newer Dropbox installations)
            dropbox_roots = [
                os.path.expanduser("~/Library/CloudStorage/Dropbox"),
                os.path.expanduser("~/Library/CloudStorage/Dropbox-Personal"),
                # Traditional Dropbox locations (fallback)
                os.path.expanduser("~/Dropbox"),
                os.path.expanduser("~/Dropbox (Personal)"),
            ]

        # IC Technical subfolder path
        ic_subfolder = os.path.join("Tools_In_Construction", "ic", "score_history")

        for dropbox_root in dropbox_roots:
            if os.path.isdir(dropbox_root):
                # Check if IC subfolder exists
                ic_path = os.path.join(dropbox_root, ic_subfolder)
                if os.path.isdir(ic_path):
                    return ic_path
                # If not, create it
                try:
                    os.makedirs(ic_path, exist_ok=True)
                    print(f"[HistoryTracker] Created Dropbox folder: {ic_path}")
                    return ic_path
                except OSError:
                    # Can't create, fall back to Dropbox root
                    return dropbox_root

        return None

    def _load(self) -> Dict[str, List[dict]]:
        """Load from JSON."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save(self):
        """Save to JSON."""
        with open(self.storage_path, 'w') as f:
            json.dump(self.data, f, indent=2, default=str)

    def record(
        self,
        asset_name: str,
        dmas: int,
        technical_score: int,
        momentum_score: int,
        price_vs_50ma_pct: float,
        price_vs_100ma_pct: float,
        price_vs_200ma_pct: float,
        rating: str,
        date: str = None
    ):
        """
        Record snapshot. Same date = overwrite.
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        snapshot = {
            "date": date,
            "dmas": dmas,
            "technical_score": technical_score,
            "momentum_score": momentum_score,
            "price_vs_50ma_pct": round(price_vs_50ma_pct, 2),
            "price_vs_100ma_pct": round(price_vs_100ma_pct, 2),
            "price_vs_200ma_pct": round(price_vs_200ma_pct, 2),
            "rating": rating
        }

        if asset_name not in self.data:
            self.data[asset_name] = []

        # Find and replace if same date exists
        existing_idx = None
        for i, s in enumerate(self.data[asset_name]):
            if s["date"] == date:
                existing_idx = i
                break

        if existing_idx is not None:
            self.data[asset_name][existing_idx] = snapshot
        else:
            self.data[asset_name].append(snapshot)

        # Sort by date and keep last 52 weeks
        self.data[asset_name] = sorted(
            self.data[asset_name],
            key=lambda x: x["date"]
        )[-52:]

        self._save()

    def record_batch(self, assets_data: List[dict], date: str = None):
        """Record multiple assets at once (more efficient)."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        print(f"[HistoryTracker] Recording batch of {len(assets_data)} assets for date {date}")
        for asset in assets_data:
            self.record(
                asset_name=asset["asset_name"],
                dmas=asset["dmas"],
                technical_score=asset["technical_score"],
                momentum_score=asset["momentum_score"],
                price_vs_50ma_pct=asset.get("price_vs_50ma_pct", 0),
                price_vs_100ma_pct=asset.get("price_vs_100ma_pct", 0),
                price_vs_200ma_pct=asset.get("price_vs_200ma_pct", 0),
                rating=asset.get("rating", "Neutral"),
                date=date
            )
        print(f"[HistoryTracker] Saved to: {self.storage_path}")

    def get_history(self, asset_name: str) -> List[dict]:
        """Get all historical snapshots for an asset."""
        return self.data.get(asset_name, [])

    def get_last_week(self, asset_name: str) -> Optional[dict]:
        """Get previous week's data.

        Returns the most recent entry that is NOT from the current week.
        - If latest entry is from a previous week, return it
        - If latest entry is from this week, return second-to-last
        """
        history = self.get_history(asset_name)
        if not history:
            return None

        today = datetime.now().date()
        # Get the Monday of the current week
        current_week_start = today - timedelta(days=today.weekday())

        # Check the most recent entry
        last_entry = history[-1]
        last_date = datetime.strptime(last_entry["date"], "%Y-%m-%d").date()

        # If the last entry is from a previous week, return it
        if last_date < current_week_start:
            return last_entry

        # Last entry is from this week, so return second-to-last if available
        if len(history) >= 2:
            return history[-2]

        return None

    def get_context_for_subtitle(self, asset_name: str) -> Optional[str]:
        """
        Generate context string for subtitle prompt.
        Returns None if insufficient history.
        """
        history = self.get_history(asset_name)

        if len(history) < 3:
            return None

        context_parts = []

        # 1. Weeks below/above 50d MA
        current_below = history[-1]["price_vs_50ma_pct"] < -1
        weeks_same_side = 0
        for h in reversed(history):
            if (h["price_vs_50ma_pct"] < -1) == current_below:
                weeks_same_side += 1
            else:
                break

        if weeks_same_side >= 3:
            side = "below" if current_below else "above"
            context_parts.append(f"{side.capitalize()} 50d MA for {weeks_same_side} consecutive weeks")

        # 2. DMAS trend (last 4 weeks)
        if len(history) >= 4:
            recent_dmas = [h["dmas"] for h in history[-4:]]
            dmas_change = recent_dmas[-1] - recent_dmas[0]

            if dmas_change >= 10:
                context_parts.append(f"DMAS improved +{dmas_change} over past month")
            elif dmas_change <= -10:
                context_parts.append(f"DMAS declined {dmas_change} over past month")

        # 3. Rating persistence
        current_rating = history[-1]["rating"]
        weeks_at_rating = 0
        for h in reversed(history):
            if h["rating"] == current_rating:
                weeks_at_rating += 1
            else:
                break

        if weeks_at_rating >= 4:
            context_parts.append(f"{current_rating} for {weeks_at_rating} consecutive weeks")

        # 4. Divergence persistence
        if len(history) >= 3:
            divergence_weeks = 0
            mom_leads = history[-1]["momentum_score"] > history[-1]["technical_score"]

            for h in reversed(history):
                current_mom_leads = h["momentum_score"] > h["technical_score"]
                diff = abs(h["momentum_score"] - h["technical_score"])
                if current_mom_leads == mom_leads and diff >= 15:
                    divergence_weeks += 1
                else:
                    break

            if divergence_weeks >= 3:
                leader = "Momentum" if mom_leads else "Technical"
                context_parts.append(f"{leader} leading for {divergence_weeks} weeks")

        if context_parts:
            return "; ".join(context_parts)
        return None

    def get_dataframe(self, asset_name: str):
        """Get history as pandas DataFrame for Streamlit charts."""
        import pandas as pd

        history = self.get_history(asset_name)
        if not history:
            return pd.DataFrame()

        df = pd.DataFrame(history)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        return df

    def get_all_assets(self) -> List[str]:
        """Get list of all tracked assets."""
        return list(self.data.keys())


# Singleton
_tracker = None


def get_tracker(storage_path: str = None) -> HistoryTracker:
    global _tracker
    if _tracker is None:
        _tracker = HistoryTracker(storage_path)
    return _tracker
