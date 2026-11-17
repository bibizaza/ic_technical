"""S1 reproduction stop and offline safety contracts."""
import socket
import pytest
from research.us_gauge import run_s1 as s1


def test_reproduction_requires_published_rounding_and_sample():
    assert s1.reproduces({'n': 247, 'ic': .0621, 'ls_ann': .0631})
    assert not s1.reproduces({'n': 247, 'ic': .0614, 'ls_ann': .0631})
    assert not s1.reproduces({'n': 247, 'ic': .0621, 'ls_ann': .0624})
    assert not s1.reproduces({'n': 248, 'ic': .0621, 'ls_ann': .0631})


def test_offline_guard_blocks_network_and_external_writes(tmp_path):
    with s1.read_only_external():
        with pytest.raises(RuntimeError, match='network'):
            socket.create_connection(('127.0.0.1', 9))
        with pytest.raises(RuntimeError, match='write'):
            (s1.MACRO / 's1_must_not_write.txt').write_text('no')


def test_failed_reproduction_stops_before_tilt(tmp_path, monkeypatch):
    monkeypatch.setattr(s1, 'HERE', tmp_path)
    monkeypatch.setattr(s1, 'reproduce', lambda: ({'n': 247, 'ic': .01, 'ls_ann': .02}, None, None, None))
    monkeypatch.setattr(s1, 'run_tilt', lambda *a: pytest.fail('Must stop before tilt'))
    s1.main()
    assert 'STOPPED' in (tmp_path / 'RESULTS_s1.md').read_text()
    assert not (tmp_path / 'out/s1_metrics.csv').exists()


def test_tilt_uses_top_three_next_month_returns_and_ten_bp_turnover():
    import numpy as np
    import pandas as pd
    idx = pd.date_range('2000-01-31', periods=4, freq='ME')
    cols = list('ABCDEF')
    pred = pd.DataFrame([[6, 5, 4, 3, 2, 1], [1, 2, 3, 4, 5, 6],
                         [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]], index=idx, columns=cols)
    industry = pd.DataFrame(0., index=idx, columns=cols)
    industry.loc[idx[1], 'A'] = .1
    panel = pd.DataFrame({'r_eq': [.5, .02, .03, .04], 'r_rf': .001}, index=idx)
    got = s1.tilt_returns(panel, pred, industry)
    # Initial allocation charged on both portfolios, 10 bp on total weight 1.
    assert got.S1.iloc[0] == pytest.approx(.7 * .02 + .1 * .1 - .001)
    # Rotate all three 10% industry positions: 60% turnover = 6 bp.
    assert got.S1.iloc[1] == pytest.approx(.7 * .03 - .0006)
    assert got.S1.iloc[2] == pytest.approx(.7 * .04)
    assert got.market.iloc[0] == pytest.approx(.02 - .001)
    assert len(got) == 3
    assert got.cash.eq(.001).all()
    changed = industry.copy()
    changed.loc[idx[2]:] = 1.
    assert s1.tilt_returns(panel, pred, changed).S1.iloc[0] == got.S1.iloc[0]


def test_monthly_metrics_annualize_at_twelve():
    import numpy as np
    import pandas as pd
    r = pd.Series([.01, -.005, .02] * 4)
    cash = pd.Series(.001, index=r.index)
    got = s1.monthly_metrics(r, cash)
    assert got['cagr'] == pytest.approx((1 + r).prod() - 1)
    assert got['sharpe'] == pytest.approx((r - cash).mean() / (r - cash).std() * np.sqrt(12))
