# IC Technical App - CLI Refactoring Analysis

## Executive Summary

This document analyzes the IC Technical presentation generation app for headless CLI execution. The app is a Streamlit-based tool that generates PowerPoint presentations with technical analysis charts, performance heatmaps, and AI-generated subtitles.

**Key Findings:**
- ~597 Streamlit-specific calls throughout `app.py`
- Core business logic is already modular and largely Streamlit-independent
- Main refactoring needed: decouple UI state management from business logic
- Estimated effort: Medium (2-3 days for core functionality)

---

## 1. Execution Flow When User Clicks "Generate"

### Phase 1: Data Loading (Lines 3590-3610)
```
User clicks "Generate updated PPTX" button
    │
    ├── Load PPTX template from st.session_state["pptx_file"]
    │   └── Write to temp file, load with python-pptx
    │
    ├── Load Excel data from st.session_state["excel_file"]
    │   └── Write to temp file for multiple sheet reads
    │
    └── Clear Excel cache (technical_analysis.common_helpers.clear_excel_cache)
```

### Phase 2: Date & Config Processing (Lines 3665-3758)
```
    ├── Determine current date (Europe/Zurich timezone)
    │   └── Format for slide date text and filename stamp
    │
    ├── Replace [DataIC] placeholder with formatted date
    │
    └── Retrieve from session_state:
        ├── anchor_date for each instrument (spx_anchor, csi_anchor, etc.)
        ├── price_mode ("Last Price" or "PX_Mid")
        └── All DMAS scores, assessments, subtitles
```

### Phase 3: Technical Analysis Charts (Lines 3783-5283)
```
For each of 20 instruments (9 equity + 6 commodity + 5 crypto):
    │
    ├── Read session_state for:
    │   ├── {ticker}_dmas (current DMAS score)
    │   ├── {ticker}_last_week_avg (previous week DMAS)
    │   ├── {ticker}_last_week_tech/mom/rsi (previous scores)
    │   ├── {ticker}_selected_view (assessment text)
    │   └── {ticker}_subtitle (generated subtitle)
    │
    ├── Compute scores:
    │   ├── _get_{ticker}_technical_score(excel_path)
    │   └── _get_{ticker}_momentum_score(excel_path)
    │
    ├── Generate chart:
    │   └── create_technical_analysis_v2_chart(
    │         excel_path, ticker, price_mode, dmas_score, ...
    │       ) → Returns PNG bytes
    │
    └── Insert into presentation:
        └── insert_technical_analysis_v2_slide(prs, bytes, ...)
```

### Phase 4: Performance Charts (Lines 5292-5555)
```
Generate and insert performance slides:
    ├── Equity: weekly bar, historical heatmap, YTD evolution, FX impact (EUR/CHF)
    ├── FX: weekly HTML chart, historical HTML heatmap
    ├── Crypto: weekly/historical HTML charts, YTD evolution
    ├── Rates: weekly bar, historical heatmap
    ├── Credit: weekly bar, historical heatmap
    └── Commodities: weekly/historical HTML charts, YTD evolution
```

### Phase 5: Summary Slides (Lines 5557-5700)
```
    ├── Technical Analysis "In A Nutshell" slide
    │   └── Aggregates all DMAS scores into summary table
    │
    ├── Market Breadth slide (if breadth page data exists)
    │
    └── Fundamental slide (if fundamental data exists)
```

### Phase 6: Output Generation (Lines 5756-5793)
```
    ├── Force Calibri font on all tables
    ├── Disable image compression
    ├── Save to BytesIO stream
    ├── Generate filename: {DDMMYYYY}_Herculis_Partners_Technical_Update.pptx
    └── Display download button via st.sidebar.download_button()
```

---

## 2. Streamlit Dependencies Analysis

### 2.1 Session State Keys Used

**Input Files:**
| Key | Type | Description |
|-----|------|-------------|
| `excel_file` | UploadedFile | Consolidated Excel with price data |
| `pptx_file` | UploadedFile | PowerPoint template |

**Date Configuration:**
| Key | Type | Description |
|-----|------|-------------|
| `data_as_of` | date | Calendar date for data filtering |
| `price_mode` | str | "Last Price" or "PX_Mid" |

**Per-Instrument Keys (20 instruments):**
| Pattern | Type | Description |
|---------|------|-------------|
| `{ticker}_dmas` | float | Current DMAS score |
| `{ticker}_last_week_avg` | float | Previous week DMAS |
| `{ticker}_last_week_tech` | float | Previous technical score |
| `{ticker}_last_week_mom` | float | Previous momentum score |
| `{ticker}_last_week_rsi` | float | Previous RSI |
| `{ticker}_tech_score` | float | Current technical score |
| `{ticker}_mom_score` | float | Current momentum score |
| `{ticker}_selected_view` | str | Assessment text (Bullish/Neutral/etc.) |
| `{ticker}_subtitle` | str | Generated subtitle |
| `{ticker}_anchor` | datetime | Regression channel start date |
| `{ticker}_enable_channel` | bool | Whether to show regression channel |
| `{ticker}_prev_days_gap` | int | Days since previous entry |
| `{ticker}_prev_date` | date | Previous entry date |

**YTD Subtitles:**
| Key | Type | Description |
|-----|------|-------------|
| `eq_subtitle` | str | Equity YTD recap subtitle |
| `co_subtitle` | str | Commodity YTD recap subtitle |
| `cr_subtitle` | str | Crypto YTD recap subtitle |

**Market Caps (optional):**
| Key | Type | Description |
|-----|------|-------------|
| `crypto_market_caps` | dict | Live crypto market caps from CoinMarketCap |
| `commodity_market_caps` | dict | Calculated commodity market caps |

### 2.2 Streamlit UI Components (Categorized)

**Must Replace for CLI:**
- `st.sidebar.button()` - Triggers generation
- `st.sidebar.download_button()` - Output delivery
- `st.sidebar.file_uploader()` - File input
- `st.sidebar.date_input()` - Date selection
- `st.sidebar.radio()` - Navigation
- `st.sidebar.selectbox()` - Model selection
- `st.sidebar.checkbox()` - Instrument selection
- `st.progress()` - Progress display
- `st.spinner()` - Loading indicator
- `st.stop()` - Flow control

**Informational (Can be print/logging):**
- `st.sidebar.success()` / `st.sidebar.error()` / `st.sidebar.warning()` / `st.sidebar.info()`
- `st.write()` / `st.text()`
- `st.empty()` - Status placeholders

**Secrets Access:**
- `st.secrets["anthropic"]["api_key"]` - Claude API key
- `st.secrets["coinmarketcap"]["api_key"]` - CoinMarketCap API key

---

## 3. Input Identification

### 3.1 Required Files

| File | Format | Purpose | Excel Sheets Used |
|------|--------|---------|-------------------|
| Excel Data | .xlsx/.xlsm | Price data, scores | `data_prices`, `mars_score`, `transition` |
| PPTX Template | .pptx/.pptm | Slide template | N/A |

### 3.2 Environment/Secrets

| Secret | Environment Variable Alternative | Purpose |
|--------|----------------------------------|---------|
| `anthropic.api_key` | `ANTHROPIC_API_KEY` | Claude subtitle generation |
| `coinmarketcap.api_key` | `CMC_API_KEY` | Live crypto market caps |

### 3.3 Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_as_of` | date | Max date in Excel | Filter data up to this date |
| `price_mode` | str | "Last Price" | Price column to use |
| `selected_instruments` | list | All 20 | Which instruments to include |
| `output_format` | str | "pptx" | Could add PDF option |
| `output_path` | str | N/A | Where to save output |

---

## 4. Output Identification

### 4.1 Primary Outputs

| Output | Format | Size | Description |
|--------|--------|------|-------------|
| PowerPoint | .pptx | ~15-30 MB | Full presentation with charts |
| Filename Format | - | - | `{DDMMYYYY}_Herculis_Partners_Technical_Update.pptx` |

### 4.2 Side Effects / Persistent Data

| Output | Location | Description |
|--------|----------|-------------|
| `history.json` | Dropbox or local | Historical DMAS scores, subtitles |
| Console logs | stdout | Debug/progress information |

### 4.3 Export PNG (Optional Feature)

| Output | Format | Resolution | Description |
|--------|--------|------------|-------------|
| PNG slides | .png | Up to 3840x2400 | Individual technical slides |

---

## 5. Date Selection Logic

### 5.1 Current Implementation (Lines 1335-1380)

```python
# 1. Read max date from Excel data_prices sheet
df_dates = pd.read_excel(excel_path, sheet_name="data_prices")
df_dates["Date"] = pd.to_datetime(df_dates[df_dates.columns[0]], errors="coerce")
max_date = df_dates["Date"].max().date()

# 2. Allow user to select earlier date (for historical generation)
data_as_of = st.sidebar.date_input(
    "Data As Of",
    value=st.session_state.get("data_as_of", max_date),
    max_value=max_date
)
st.session_state["data_as_of"] = data_as_of

# 3. Filter all data reads by this date
df_prices = df_prices[df_prices["Date"] <= pd.Timestamp(st.session_state["data_as_of"])]
```

### 5.2 CLI Equivalent

```python
def determine_data_date(excel_path: Path, data_as_of: date = None) -> date:
    """
    Determine the effective data date.

    Args:
        excel_path: Path to Excel file
        data_as_of: Optional override date (CLI argument)

    Returns:
        date: The effective date to use for data filtering
    """
    df_dates = pd.read_excel(excel_path, sheet_name="data_prices")
    df_dates["Date"] = pd.to_datetime(df_dates[df_dates.columns[0]], errors="coerce")
    max_date = df_dates["Date"].max().date()

    if data_as_of is None:
        return max_date

    if data_as_of > max_date:
        raise ValueError(f"data_as_of ({data_as_of}) exceeds max date in Excel ({max_date})")

    return data_as_of
```

---

## 6. Proposed CLI Entry Point

### 6.1 Architecture

```
cli_generate.py          # CLI entry point
    │
    ├── config.py         # Configuration dataclass
    ├── state.py          # State manager (replaces session_state)
    └── generator.py      # Core generation logic (extracted from app.py)

Reuses existing modules:
    ├── technical_analysis/*
    ├── performance/*
    ├── market_compass/*
    └── assessment_integration.py
```

### 6.2 CLI Interface Design

```bash
# Basic usage
python cli_generate.py \
    --excel /path/to/ic_data.xlsx \
    --template /path/to/template.pptx \
    --output /path/to/output.pptx

# With options
python cli_generate.py \
    --excel /path/to/ic_data.xlsx \
    --template /path/to/template.pptx \
    --output /path/to/output.pptx \
    --data-as-of 2025-02-21 \
    --price-mode "Last Price" \
    --instruments spx,gold,bitcoin \
    --skip-subtitles \
    --verbose

# Environment variables for secrets
export ANTHROPIC_API_KEY="sk-..."
export CMC_API_KEY="..."
```

### 6.3 Configuration Dataclass

```python
@dataclass
class GenerationConfig:
    """Configuration for headless generation."""

    # Required inputs
    excel_path: Path
    template_path: Path
    output_path: Path

    # Date configuration
    data_as_of: Optional[date] = None  # None = use max date
    price_mode: str = "Last Price"

    # Instrument selection
    instruments: Optional[List[str]] = None  # None = all

    # Feature flags
    generate_subtitles: bool = True
    fetch_crypto_caps: bool = True
    save_to_history: bool = True

    # API keys (from env if not provided)
    anthropic_api_key: Optional[str] = None
    cmc_api_key: Optional[str] = None

    # Output options
    verbose: bool = False
    export_png: bool = False
    png_scale: int = 4
```

### 6.4 State Manager (Replaces session_state)

```python
class GenerationState:
    """
    Manages state during generation, replacing Streamlit session_state.

    Can be initialized from:
    - Transition sheet in Excel
    - Previous generation state (JSON)
    - History tracker
    """

    def __init__(self):
        self._state: Dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._state[key] = value

    def load_from_transition_sheet(self, excel_path: Path) -> None:
        """Load initial state from transition sheet."""
        from transition_loader import read_transition_sheet, apply_transition_data_to_session_state
        transition_data = read_transition_sheet(excel_path)
        # Adapt apply_transition_data_to_session_state to use self
        ...

    def load_from_history(self, data_as_of: date) -> None:
        """Load previous week's values from history tracker."""
        from market_compass.subtitle_generator.history_tracker import get_tracker
        tracker = get_tracker()
        ...

    def to_dict(self) -> Dict[str, Any]:
        """Export state for debugging/caching."""
        return self._state.copy()
```

### 6.5 Core Generator (Extracted from app.py)

```python
def generate_presentation(
    config: GenerationConfig,
    state: GenerationState,
    progress_callback: Callable[[str, float], None] = None
) -> bytes:
    """
    Generate presentation headlessly.

    Args:
        config: Generation configuration
        state: Pre-populated state with scores, subtitles, etc.
        progress_callback: Optional callback for progress updates

    Returns:
        bytes: PowerPoint file content
    """
    # 1. Load template
    prs = Presentation(config.template_path)

    # 2. Update date placeholder
    update_date_placeholder(prs)

    # 3. Generate technical analysis slides
    for ticker_key in get_instruments(config):
        chart_bytes = generate_technical_chart(config, state, ticker_key)
        insert_technical_slide(prs, chart_bytes, state, ticker_key)
        if progress_callback:
            progress_callback(f"Generated {ticker_key}", ...)

    # 4. Generate performance slides
    generate_performance_slides(prs, config, state)

    # 5. Generate summary slides
    generate_summary_slides(prs, config, state)

    # 6. Finalize
    finalize_presentation(prs)

    # 7. Return bytes
    output = BytesIO()
    prs.save(output)
    return output.getvalue()
```

---

## 7. Risk Assessment for Headless Execution

### 7.1 High Risk

| Risk | Description | Mitigation |
|------|-------------|------------|
| **Playwright Browser** | Chart generation requires headless browser | Ensure Playwright is installed with `playwright install chromium` |
| **Memory Usage** | Generating 20+ high-res charts uses significant memory | Process instruments in batches, explicit garbage collection |
| **Timeout** | Subtitle generation via Claude API can be slow | Implement timeouts, fallback to pattern-based subtitles |

### 7.2 Medium Risk

| Risk | Description | Mitigation |
|------|-------------|------------|
| **Excel File Locking** | Multiple reads can lock file | Use tempfile copies as current code does |
| **Missing Data** | Some instruments may lack data | Graceful skip with warning |
| **API Rate Limits** | Claude API batch calls | Already batched, add retry logic |
| **History File Conflicts** | Multiple processes writing | File locking or separate history per run |

### 7.3 Low Risk

| Risk | Description | Mitigation |
|------|-------------|------------|
| **Font Availability** | Calibri font required | Document requirement, already works on Mac/Windows |
| **Timezone** | Date formatting uses Europe/Zurich | Explicitly set timezone in CLI |
| **Path Handling** | Windows vs Unix paths | Use pathlib throughout (already done) |

---

## 8. Implementation Roadmap

### Phase 1: Core Extraction (Day 1)
1. Create `cli/config.py` with GenerationConfig dataclass
2. Create `cli/state.py` with GenerationState class
3. Extract generation logic from `app.py` lines 3546-5793 into `cli/generator.py`
4. Replace `st.session_state` references with `state` parameter

### Phase 2: CLI Interface (Day 2)
1. Create `cli_generate.py` with argparse
2. Add environment variable handling for secrets
3. Add progress output (tqdm or simple print)
4. Test with minimal inputs

### Phase 3: Testing & Polish (Day 3)
1. Test all 20 instruments
2. Add error handling and validation
3. Add --dry-run option
4. Document usage

### Optional Enhancements
- JSON config file support
- Multiple output formats (PPTX, PDF, PNG)
- Scheduling integration (cron/Task Scheduler)
- Webhook notifications on completion

---

## 9. Files Requiring Modification

### Direct Modification Required

| File | Changes |
|------|---------|
| `app.py` | Extract generation logic to reusable functions |
| `transition_loader.py` | Make `apply_transition_data_to_session_state` work with dict-like interface |

### No Modification Required (Already Streamlit-Independent)

| Module | Description |
|--------|-------------|
| `technical_analysis/*` | Chart generation logic |
| `performance/*` | Performance chart generation |
| `market_compass/*` | Slide generators, subtitle generators |
| `assessment_integration.py` | Assessment/subtitle logic |
| `technical_score_wrapper.py` | Score computation |
| `utils.py` | Price adjustment utilities |

---

## 10. Testing Strategy

### Unit Tests
- Config validation
- State loading from transition sheet
- State loading from history

### Integration Tests
- Full generation with minimal Excel (1 instrument)
- Full generation with all instruments
- Generation with missing data
- Generation without API keys (fallback modes)

### End-to-End Tests
- Compare CLI output to Streamlit output
- Verify all slides generated
- Verify chart quality

---

## Appendix A: Instrument List

| Category | Ticker Key | Bloomberg Ticker | Display Name |
|----------|------------|------------------|--------------|
| Equity | spx | SPX Index | S&P 500 |
| Equity | csi | SHSZ300 Index | CSI 300 |
| Equity | nikkei | NKY Index | Nikkei 225 |
| Equity | tasi | SASEIDX Index | TASI |
| Equity | sensex | SENSEX Index | Sensex |
| Equity | dax | DAX Index | DAX |
| Equity | smi | SMI Index | SMI |
| Equity | ibov | IBOV Index | Ibovespa |
| Equity | mexbol | MEXBOL Index | MEXBOL |
| Commodity | gold | GCA COMDTY | Gold |
| Commodity | silver | SIA COMDTY | Silver |
| Commodity | platinum | XPT COMDTY | Platinum |
| Commodity | palladium | XPD CURNCY | Palladium |
| Commodity | oil | CL1 COMDTY | Oil |
| Commodity | copper | LP1 COMDTY | Copper |
| Crypto | bitcoin | XBTUSD CURNCY | Bitcoin |
| Crypto | ethereum | XETUSD CURNCY | Ethereum |
| Crypto | ripple | XRPUSD CURNCY | Ripple |
| Crypto | solana | XSOUSD CURNCY | Solana |
| Crypto | binance | XBIUSD CURNCY | Binance |

---

## Appendix B: Key Module Dependencies

```
cli_generate.py
    ├── cli/config.py
    ├── cli/state.py
    ├── cli/generator.py
    │   ├── technical_analysis/
    │   │   ├── equity/{spx,csi,nikkei,...}.py
    │   │   ├── commodity/{gold,silver,...}.py
    │   │   ├── crypto/{bitcoin,ethereum,...}.py
    │   │   ├── templates/technical_analysis_v2.py
    │   │   └── templates/full_slide_renderer.py
    │   ├── performance/
    │   │   ├── equity_perf.py
    │   │   ├── fx_perf.py
    │   │   ├── crypto_perf.py
    │   │   ├── rates_perf.py
    │   │   ├── corp_bonds_perf.py
    │   │   └── commodity_perf.py
    │   └── market_compass/
    │       ├── subtitle_generator/
    │       │   ├── claude_generator.py
    │       │   └── history_tracker.py
    │       ├── technical_slide/
    │       ├── breadth_slide/
    │       └── fundamental_slide/
    ├── assessment_integration.py
    ├── technical_score_wrapper.py
    ├── transition_loader.py
    └── utils.py
```
