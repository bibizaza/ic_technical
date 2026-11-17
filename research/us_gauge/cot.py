"""Pre-registered leveraged-fund positioning, selected by actual release date."""
import pandas as pd


def cot_signal(lev_long, lev_short, oi, release_lag_days, decision_dates) -> pd.DataFrame:
    net = ((lev_long - lev_short) / oi).sort_index().rename('net')
    history = net.shift(1).rolling(156, min_periods=104)
    reports = net.to_frame()
    reports['C_z'] = (net - history.mean()) / history.std()
    reports['C_in'] = reports.C_z.ge(0).astype('boolean').mask(reports.C_z.isna())
    reports['report_date_used'] = reports.index
    release = reports.index + pd.to_timedelta(release_lag_days.reindex(reports.index), unit='D')
    # Select by report date, not arrival order: an old delayed report must never
    # displace a newer report that is already available.
    selected = [reports.index[release <= d].max() for d in decision_dates]
    result = reports.reindex(pd.DatetimeIndex(selected))
    result.index = pd.DatetimeIndex(decision_dates)
    return result
