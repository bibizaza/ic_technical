# Transition Sheet Format

The **transition** sheet in the Excel file is used to automatically pre-populate app fields when you upload the file.

## Sheet Structure

| Column | Name | Description | Example |
|--------|------|-------------|---------|
| A | Ticker | Bloomberg ticker (uppercase) | `SPX INDEX`, `GCA COMDTY`, `XBTUSD CURNCY` |
| B | Last Week DMAS | Previous week's DMAS value (0-100) | `65.5` |
| C | Anchor Date | Regression channel start date | `2024-01-15` |
| D | Assessment | Technical assessment | `Bullish`, `Neutral`, `Bearish` |
| E | Subtitle | Custom subtitle text | `Breaking above key resistance` |

**Important:**
- Row 1 is the header row (can be any text or blank)
- **Data starts at row 2**
- If a field is empty for a ticker, the app will use its default value
- Ticker names must match exactly (case-insensitive)

## Example

```
Row 1: | Ticker        | Last Week | Anchor Date | Assessment          | Subtitle                    |
Row 2: | SPX INDEX     | 70        | 2024-01-01  | Bullish             | Strong momentum continuing  |
Row 3: | GCA COMDTY    | 55        | 2024-02-15  | Neutral             | Consolidating near highs    |
Row 4: | XBTUSD CURNCY | 85        |             | Moderately Bullish  |                             |
```

## Supported Tickers

### Equity
- `SPX INDEX` - S&P 500
- `SHSZ300 INDEX` - CSI 300
- `NKY INDEX` - Nikkei 225
- `SASEIDX INDEX` - TASI
- `SENSEX INDEX` - Sensex
- `DAX INDEX` - DAX
- `SMI INDEX` - SMI
- `IBOV INDEX` - Ibov
- `MEXBOL INDEX` - Mexbol

### Commodity
- `GCA COMDTY` - Gold
- `SIA COMDTY` - Silver
- `XPT COMDTY` - Platinum
- `XPD CURNCY` - Palladium
- `CL1 COMDTY` - Oil (WTI)
- `LP1 COMDTY` - Copper

### Crypto
- `XBTUSD CURNCY` - Bitcoin
- `XETUSD CURNCY` - Ethereum
- `XRPUSD CURNCY` - Ripple
- `XSOUSD CURNCY` - Solana
- `XBIUSD CURNCY` - Binance

## Assessment Values

Valid assessment values:
- `Bullish`
- `Moderately Bullish`
- `Neutral`
- `Moderately Bearish`
- `Bearish`

## How It Works

1. Upload your Excel file with a **transition** sheet
2. The app automatically reads this sheet when the file is uploaded
3. All fields are pre-populated for the specified tickers
4. Navigate to Technical Analysis to see the pre-filled values

## Notes

- The transition sheet is **optional** - if it doesn't exist, the app works normally
- Empty cells are ignored - the app uses default values for those fields
- Invalid data (wrong dates, non-numeric DMAS, etc.) is ignored with a warning
- The anchor date automatically enables the regression channel for that instrument
