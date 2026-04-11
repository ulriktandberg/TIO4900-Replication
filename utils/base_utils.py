import os
import pandas as pd
from typing import Tuple
import numpy as np
import statsmodels.api as sm
from scipy.stats import t as tstat

# Repo root: one level up from utils/
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_fred_md_data(filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load FRED-MD dataset with proper handling of transformation codes
    
    Parameters:
    -----------
    filepath : str
        Path to FRED-MD CSV file
        
    Returns:
    --------
    tuple
        Raw data and transformation codes
    """
    # Read the full file
    full_data = pd.read_csv(filepath)
    
    # Extract transformation codes (second row)
    transform_codes = full_data.iloc[0, 1:].astype(int)
    transform_codes.name = 'transform_codes'
    
    # Extract data (third row onwards)
    data = full_data.iloc[1:].copy()
    data = data.reset_index(drop=True)
    
    # Convert date column
    data['date'] = pd.to_datetime(data['date'], format='%m/%d/%Y')
    data = data.set_index('date')

    # align dates to last date in previous month
    data.index = pd.to_datetime(data.index)  # ensure datetime index
    def _prev_month_end(dt):
        if pd.isna(dt):
            return dt
        return (dt.replace(day=1) - pd.Timedelta(days=1)).normalize()
    data.index = pd.DatetimeIndex([_prev_month_end(d) for d in data.index])
    data.index.name = 'date'

    # Convert all other columns to numeric
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    return data, transform_codes


def apply_fred_transformations(data: pd.DataFrame, transform_codes: pd.Series) -> pd.DataFrame:
    """
    Apply FRED-MD transformation codes to data
    
    Transform codes:
    1 = no transformation (levels)
    2 = first difference
    3 = second difference  
    4 = log
    5 = log first difference
    6 = log second difference
    7 = delta(x_t/x_{t-1} - 1)
    
    Parameters:
    -----------
    data : pd.DataFrame
        Raw data with variables in columns
    transform_codes : pd.Series
        Transformation codes for each variable
        
    Returns:
    --------
    pd.DataFrame
        Transformed data
    """
    transformed = data.copy()
    
    for col in data.columns:
        if col not in transform_codes.index:
            continue
            
        code = transform_codes[col]
        series = data[col].copy()
        
        # Handle missing values
        if series.isna().all():
            continue
            
        try:
            if code == 1:  # Levels
                transformed[col] = series
            elif code == 2:  # First difference
                transformed[col] = series.diff()
            elif code == 3:  # Second difference
                transformed[col] = series.diff().diff()
            elif code == 4:  # Log
                # Only take log of positive values
                series_pos = series[series > 0]
                if len(series_pos) > 0:
                    transformed[col] = np.log(series)
                else:
                    transformed[col] = np.nan
            elif code == 5:  # Log first difference
                series_pos = series[series > 0]
                if len(series_pos) > 0:
                    transformed[col] = np.log(series).diff()
                else:
                    transformed[col] = np.nan
            elif code == 6:  # Log second difference
                series_pos = series[series > 0]
                if len(series_pos) > 0:
                    transformed[col] = np.log(series).diff().diff()
                else:
                    transformed[col] = np.nan
            elif code == 7:  # Delta(x_t/x_{t-1} - 1)
                transformed[col] = (series / series.shift(1) - 1).diff()
            else:
                print(f"Unknown transformation code {code} for variable {col}")
                transformed[col] = series
                
        except Exception as e:
            print(f"Error transforming {col} with code {code}: {e}")
            transformed[col] = np.nan
            
    return transformed


def get_fred_data(filepath: str, start: str, end: str) -> pd.DataFrame:
    """Convenience function to load and transform FRED-MD data in one step. """
    # Resolve relative paths against repo root
    if not os.path.isabs(filepath):
        filepath = os.path.join(_REPO_ROOT, filepath)

    # For sparse UMCSENTx history, fill gaps first so first-difference
    # transformation does not create long initial NaN stretches.
    fred_raw, transform_codes = load_fred_md_data(filepath)
    if 'UMCSENTx' in fred_raw.columns:
        fred_raw['UMCSENTx'] = fred_raw['UMCSENTx'].ffill()

    fred_md = apply_fred_transformations(fred_raw, transform_codes)
    return fred_md[start:end]


def get_yields(type: str, start: str, end: str, maturities: list) -> pd.DataFrame:
    """Load and preprocess KR yields data."""
    if type == 'kr':
        yields = pd.read_csv(os.path.join(_REPO_ROOT, 'data', 'kr_yields.csv'), index_col=0, parse_dates=True)
        # Snap business-day month-ends to calendar month-ends (e.g. Sep 29 -> Sep 30)
        yields.index = yields.index + pd.offsets.MonthEnd(0)
        yields.index.name = 'date'

    if type == 'lw':
        yields = pd.read_csv(os.path.join(_REPO_ROOT, 'data', 'lw_yields.csv'), index_col=0, comment='%')
        yields.index.name = 'date'
        # parse date column – snap to calendar month-end
        yields.index = pd.to_datetime(yields.index, format='%Y%m') + pd.offsets.MonthEnd(0)

        yields = yields/100  # Convert from percentage points to decimals
        # Rename columns to '1', '12', '24', ..., '120' (remove 'm' suffix and convert to string)
        yields.columns = [str(int(col.strip().rstrip('m'))) for col in yields.columns]

    if type == 'gsw':
        yields = pd.read_csv(os.path.join(_REPO_ROOT, 'data', 'gsw_yields.csv'), index_col=0, parse_dates=True, skiprows=9)
        # select only columns starting with SVENY and rename to '1', '12', '24', ..., '120' (remove 'SVENY' prefix and convert to string)
        yields = yields[[col for col in yields.columns if col.startswith('SVENY')]]
        # strip 0-padding and 'SVENY' prefix, convert to string
        yields.columns = [col.strip().lstrip('SVENY').lstrip('0') for col in yields.columns]
        # resample to month-end frequency, taking the last available observation in each month
        yields = yields.resample('ME').last()
        # snap to calendar month-end
        yields.index = yields.index + pd.offsets.MonthEnd(0)
        # rename columns to '1', '12', '24', ..., '120' instead of years 1, 2, 3, ..., 10
        yields.columns = [str(int(col) * 12) for col in yields.columns]
        # include day (end of month)
        yields.index.name = 'date'
        yields = yields/100  # Convert from percentage points to decimals

    if type not in ['kr', 'lw', 'gsw']:
        raise ValueError(f"Excess return calculation got unknown yield type: {type}")

    yields = yields[maturities]

    return yields.loc[start:end]


def get_forward_rates(yields: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate forward rates from zero-coupon yields.
    
    f_t(n) = log P_t(n-1) - log P_t(n)
    
    With monthly maturities m (in months), n = m/12 in years:
        log P_t(m) = -(m/12) * y_t(m)
        f_t(m) = -(m-1)/12 * y_t(m-1) + m/12 * y_t(m)
    
    Forward rates are computed for maturities > 1 month, i.e. 12, 24, ..., 120.
    For these maturities, m-1 means one year shorter (m-12), matching the
    yearly loan interpretation: forward rate for a loan from t+n-1 to t+n years.
    
    Parameters:
    -----------
    yields : pd.DataFrame
        Zero-coupon yields with monthly maturity columns (as strings: '1','12','24',...,'120')
        
    Returns:
    --------
    pd.DataFrame
        Forward rates for maturities 12, 24, ..., 120
    """
    # Maturities for which we compute forward rates (yearly maturities in months)
    forward_maturities = [str(i) for i in range(12, 121) if i % 12 == 0]
    
    forward_rates = pd.DataFrame(index=yields.index)
    
    for m_str in forward_maturities:
        m = int(m_str)
        # n = m/12 in years, n-1 = (m-12)/12 in years
        m_prev = m - 12  # maturity one year shorter (in months)
        
        # log P_t(m) = -(m/12) * y_t(m)
        log_p_m = -(m / 12) * yields[m_str]
        
        if m_prev == 0:
            # log P_t(0) = 0 (price of a matured bond is 1, log(1) = 0)
            log_p_m_prev = 0.0
        else:
            log_p_m_prev = -(m_prev / 12) * yields[str(m_prev)]
        
        # f_t(n) = log P_t(n-1) - log P_t(n)
        forward_rates[m_str] = log_p_m_prev - log_p_m
    
    return forward_rates


def get_fredmd_grouping():
    """
    Return (ordered_series, series_to_group) for the earlier FRED-MD style
    grouping used previously in LN.py (uppercase FRED-MD mnemonics).
    """
    ordered_series = [
        # Output & Income
        'RPI','W875RX1','INDPRO','IPFPNSS','IPFINAL','IPCONGD','IPDCONGD','IPNCONGD','IPBUSEQ','IPMAT','IPDMAT','IPNMAT','IPMANSICS','IPB51222s','IPFUELS','NAPMPI','CUMFNS',
        # Labor Market
        'HWI','HWIURATIO','CLF16OV','CE16OV','UNRATE','UEMPMEAN','UEMPLT5','UEMP5TO14','UEMP15OV','UEMP15T26','UEMP27OV','CLAIMSx','PAYEMS','USGOOD','CES1021000001','USCONS','MANEMP','DMANEMP','NDMANEMP','SRVPRD','USTPU','USWTRADE','USTRADE','USFIRE','USGOVT','CES0600000007','AWOTMAN','AWHMAN','NAPMEI','CES0600000008','CES2000000008','CES3000000008',
        # Consumption & Housing
        'HOUST','HOUSTNE','HOUSTMW','HOUSTS','HOUSTW','PERMIT','PERMITNE','PERMITMW','PERMITS','PERMITW',
        # Orders & Inventories
        'DPCERA3M086SBEA','CMRMTSPLx','RETAILx','NAPM','NAPMNOI','NAPMSDI','NAPMII','ACOGNO','AMDMNOx','ANDENOx','AMDMUOx','BUSINVx','ISRATIOx','UMCSENTx',
        # Money & Credit
        'M1SL','M2SL','M2REAL','AMBSL','TOTRESNS','NONBORRES','BUSLOANS','REALLN','NONREVSL','CONSPL','MZMSL','DTCOLNVHFNM','DTCTHFNM','INVEST',
        # Rates & FX
        'FEDFUNDS','CP3Mx','TB3MS','TB6MS','GS1','GS5','GS10','AAA','BAA','COMPAPFFx','TB3SMFFM','TB6SMFFM','T1YFFM','T5YFFM','T10YFFM','AAAFFM','BAAFFM','TWEXMMTH','EXSZUSx','EXJPUSx','EXUSUKx','EXCAUSx',
        # Prices
        'PPIFGS','PPIFCG','PPIITM','PPICRM','OILPRICEx','PPICMM','NAPMPRI','CPIAUCSL','CPIAPPSL','CPITRNSL','CPIMEDSL','CUSR0000SAC','CUUR0000SAD','CUSR0000SAS','CPIULFSL','CUUR0000SA0L2','CUSR0000SA0L5','PCEPI','DDURRG3M086SBEA','DNDGRG3M086SBEA','DSERRG3M086SBEA',
        # Stock Market
        'S&P 500','S&P: indust','S&P div yield','S&P PE ratio'
    ]
    def label_group(name: str) -> str:
        if name in ['RPI','W875RX1','INDPRO','IPFPNSS','IPFINAL','IPCONGD','IPDCONGD','IPNCONGD','IPBUSEQ','IPMAT','IPDMAT','IPNMAT','IPMANSICS','IPB51222s','IPFUELS','NAPMPI','CUMFNS']:
            return 'Output & Income'
        if name in ['HWI','HWIURATIO','CLF16OV','CE16OV','UNRATE','UEMPMEAN','UEMPLT5','UEMP5TO14','UEMP15OV','UEMP15T26','UEMP27OV','CLAIMSx','PAYEMS','USGOOD','CES1021000001','USCONS','MANEMP','DMANEMP','NDMANEMP','SRVPRD','USTPU','USWTRADE','USTRADE','USFIRE','USGOVT','CES0600000007','AWOTMAN','AWHMAN','NAPMEI','CES0600000008','CES2000000008','CES3000000008']:
            return 'Labor Market'
        if name in ['HOUST','HOUSTNE','HOUSTMW','HOUSTS','HOUSTW','PERMIT','PERMITNE','PERMITMW','PERMITS','PERMITW']:
            return 'Consumption & Housing'
        if name in ['DPCERA3M086SBEA','CMRMTSPLx','RETAILx','NAPM','NAPMNOI','NAPMSDI','NAPMII','ACOGNO','AMDMNOx','ANDENOx','AMDMUOx','BUSINVx','ISRATIOx','UMCSENTx']:
            return 'Orders & Inventories'
        if name in ['M1SL','M2SL','M2REAL','AMBSL','TOTRESNS','NONBORRES','BUSLOANS','REALLN','NONREVSL','CONSPL','MZMSL','DTCOLNVHFNM','DTCTHFNM','INVEST']:
            return 'Money & Credit'
        if name in ['FEDFUNDS','CP3Mx','TB3MS','TB6MS','GS1','GS5','GS10','AAA','BAA','COMPAPFFx','TB3SMFFM','TB6SMFFM','T1YFFM','T5YFFM','T10YFFM','AAAFFM','BAAFFM','TWEXMMTH','EXSZUSx','EXJPUSx','EXUSUKx','EXCAUSx']:
            return 'Rates & FX'
        if name in ['PPIFGS','PPIFCG','PPIITM','PPICRM','OILPRICEx','PPICMM','NAPMPRI','CPIAUCSL','CPIAPPSL','CPITRNSL','CPIMEDSL','CUSR0000SAC','CUUR0000SAD','CUSR0000SAS','CPIULFSL','CUUR0000SA0L2','CUSR0000SA0L5','PCEPI','DDURRG3M086SBEA','DNDGRG3M086SBEA','DSERRG3M086SBEA']:
            return 'Prices'
        if name in ['S&P 500','S&P: indust','S&P div yield','S&P PE ratio']:
            return 'Stock Market'
        return 'Other'
    series_to_group = {s: label_group(s) for s in ordered_series}
    return ordered_series, series_to_group


def get_excess_returns(yields: pd.DataFrame, horizon: int = 12) -> pd.DataFrame:
    """
    Calculate excess holding-period returns from zero-coupon yields.
    
    For horizon h (in months):
        r_{t+h}(m) = log P_{t+h}(m-h) - log P_t(m)
        rx_{t+h}(m) = r_{t+h}(m) - (h/12) * y_t(h)
    
    where log P_t(m) = -(m/12) * y_t(m).
    
    Parameters
    ----------
    yields : pd.DataFrame
        Zero-coupon yields. Columns are maturity strings in months
        ('1', '12', '23', '24', ..., '120').
    horizon : int
        Holding period in months.
        horizon=12 : annual (original Bianchi). Needs maturities m and m-12.
        horizon=1  : monthly (corrigendum). Needs maturities m and m-1.
    
    Returns
    -------
    pd.DataFrame
        Excess returns indexed at time t (realized at t+h).
        Columns are the original maturity m (in months, as string).
    """
    h = horizon
    available = set(yields.columns)
    
    # Require the risk-free maturity
    if str(h) not in available:
        raise ValueError(
            f"Yields must include the {h}-month maturity for the risk-free rate. "
            f"Available: {sorted(available, key=lambda x: int(x))}"
        )
    
    # Find maturities for which we can compute excess returns:
    # need m > h  AND  str(m - h) in available columns
    all_mats = sorted([int(c) for c in available])
    valid_mats = [m for m in all_mats if m > h and str(m - h) in available]
    
    if not valid_mats:
        raise ValueError(
            f"No valid maturities for horizon={h}. Need maturity m > {h} "
            f"with maturity m-{h} also available. "
            f"Available maturities: {sorted(all_mats)}"
        )
    
    # Risk-free return over h months: (h/12) * y_t(h)
    rf = (h / 12) * yields[str(h)]
    
    # Build all columns at once to avoid fragmentation
    columns = {}
    for m in valid_mats:
        m_after = m - h  # residual maturity after holding h months
        
        # log P_t(m) = -(m/12) * y_t(m)
        log_p_t = -(m / 12) * yields[str(m)]
        
        # log P_{t+h}(m-h) = -((m-h)/12) * y_{t+h}(m-h)
        log_p_th = -(m_after / 12) * yields[str(m_after)].shift(-h)
        
        # Holding-period return
        hpr = log_p_th - log_p_t
        
        columns[str(m)] = hpr - rf
    
    excess_returns = pd.DataFrame(columns, index=yields.index)
    
    # Return only yearly maturities
    yearly_cols = [str(m) for m in range(12, 121) if m % 12 == 0 and str(m) in excess_returns.columns]
    return excess_returns[yearly_cols]


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_cssed(y_true, y_forecast, dates, oos_start, secondary_start=None, model_name="Model"):
    """
    Constructs a Welch-Goyal style CSSED plot.
    
    Parameters:
    -----------
    y_true : np.array
        The actual excess returns.
    y_forecast : np.array
        The OOS forecasts from your model.
    dates : pd.Series / pd.DatetimeIndex
        The dates corresponding to the returns.
    oos_start : pd.Timestamp
        The date your OOS period actually begins.
    secondary_start : pd.Timestamp, optional
        The date for the right-hand zero-point (e.g., 2000-01-31).
    """
    # 1. Align data and filter for OOS period
    df = pd.DataFrame({
        'realized': y_true,
        'forecast': y_forecast
    }, index=dates)
    
    oos_df = df.loc[oos_start:].copy()
    
    # 2. Generate Historical Mean Benchmark (Expanding Window)
    # We use the same logic as the paper: mean of all returns available up to t-1
    full_series = pd.Series(y_true, index=dates)
    oos_df['hist_mean_bench'] = [full_series.loc[:d].iloc[:-1].mean() for d in oos_df.index]
    
    # 3. Calculate Squared Errors
    oos_df['error_model'] = (oos_df['realized'] - oos_df['forecast'])**2
    oos_df['error_bench'] = (oos_df['realized'] - oos_df['hist_mean_bench'])**2
    
    # 4. Calculate CSSED: Cumulative (SE_benchmark - SE_model)
    # An increase means the model is beating the benchmark
    oos_df['cssed'] = (oos_df['error_bench'] - oos_df['error_model']).cumsum()
    
    # 5. Plotting
    sns.set_style("whitegrid")
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Primary Line (Solid black/blue is standard for OOS)
    sns.lineplot(data=oos_df, x=oos_df.index, y='cssed', ax=ax1, color='black', linewidth=2)
    
    ax1.axhline(0, color='red', linestyle='--', alpha=0.6) # The "Null" Benchmark line
    ax1.set_ylabel(f'Cumulative SSE Difference (Zero at {oos_start.year})', fontsize=12)
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_title(f'OOS Performance: {model_name} vs. Historical Mean', fontsize=14)

    # 6. Secondary Axis (The Welch-Goyal Vertical Shift)
    if secondary_start and secondary_start in oos_df.index:
        val_at_secondary = oos_df.loc[secondary_start, 'cssed']
        
        # We create a twin axis
        ax2 = ax1.twinx()
        
        # To make the right axis 0 at the secondary_start, we align the limits
        # by shifting the primary limits by the value at the secondary start
        y1_min, y1_max = ax1.get_ylim()
        ax2.set_ylim(y1_min - val_at_secondary, y1_max - val_at_secondary)
        
        ax2.set_ylabel(f'CSSED (Zero at {secondary_start.year})', fontsize=12)
        ax2.axhline(0, color='gray', linestyle=':', alpha=0.5) # Zero line for the right axis
        
        # Mark the secondary zero point on the x-axis
        ax1.axvline(secondary_start, color='blue', linestyle='--', alpha=0.3)
        ax1.text(secondary_start, y1_max, f' Start {secondary_start.year}', 
                 verticalalignment='top', color='blue', fontsize=10)

    plt.tight_layout()
    plt.show()

def RSZ_Signif(y_true, y_forecast):
    # Copied from the replication code of Bianchi et al. (2021):
    # Compute conidtional mean forecast
    y_condmean = np.divide(y_true.cumsum(), (np.arange(y_true.size)+1))

    # lag by one period
    y_condmean = np.insert(y_condmean, 0, np.nan)
    y_condmean = y_condmean[:-1]
    y_condmean[np.isnan(y_forecast)] = np.nan

    # Compute f-measure
    f = np.square(y_true-y_condmean)-np.square(y_true-y_forecast)  \
        + np.square(y_condmean-y_forecast)

    # Regress f on a constant
    x = np.ones(np.shape(f))
    model = sm.OLS(f, x, missing='drop', hasconst=True)
    results = model.fit(cov_type='HAC', cov_kwds={'maxlags': 12})

    return 1-tstat.cdf(results.tvalues[0], results.nobs-1)
