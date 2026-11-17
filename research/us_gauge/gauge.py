"""Four-factor production DMAS and the pre-registered two-key gauge."""
import numpy as np
import pandas as pd
from pipeline.momentum import _compute_technical_score, _compute_mars_momentum

BANDS = ['G>=1', '0.5-1', '0-0.5', '-0.5-0', 'G<-0.5']


def dmas_on(px: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    scores = []
    for d in dates:
        past = px.loc[:d]
        technical = _compute_technical_score(past, d)
        momentum = _compute_mars_momentum(past, d, benchmark_close=None)
        if momentum is None:
            momentum = technical
        scores.append(int((technical + momentum) / 2))
    return pd.Series(scores, index=dates, dtype=int, name='dmas')


def gauge_frame(dmas: pd.Series, rev_z: pd.Series) -> pd.DataFrame:
    gf = pd.DataFrame({'dmas': dmas, 'rev_z': rev_z.reindex(dmas.index)})
    gf['T_in'] = gf.dmas >= 45
    gf['R_in'] = gf.rev_z >= 0
    gf['T_n'] = (gf.dmas - 45) / 15
    gf['G'] = gf[['T_n', 'rev_z']].max(axis=1)
    gf['cash_R1'] = ~gf.T_in & ~gf.R_in
    gf['cash_R0'] = ~gf.T_in
    gf['deep'] = gf.dmas < 30
    gf['band'] = np.select([gf.G >= 1, gf.G >= .5, gf.G >= 0, gf.G >= -.5], BANDS[:4], default=BANDS[4])
    return gf
