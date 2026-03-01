#!/usr/bin/env python3
"""
Append YTD prices from ic_file.xlsx data_prices sheet to master_prices.csv.

Reads new dates from Excel and appends them to the master CSV, matching
columns by ticker+field (e.g., "SPX Index" + "#price").
"""

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd


def parse_date_from_csv(date_str: str) -> Optional[datetime]:
    """Parse date from CSV format (DD/MM/YYYY)."""
    if pd.isna(date_str) or str(date_str).strip() == "":
        return None
    try:
        return datetime.strptime(str(date_str).strip(), "%d/%m/%Y")
    except ValueError:
        return None


def format_date_for_csv(dt: datetime) -> str:
    """Format date for CSV output (DD/MM/YYYY)."""
    return dt.strftime("%d/%m/%Y")


def read_master_csv(csv_path: Path) -> Tuple[List[str], List[str], pd.DataFrame]:
    """
    Read master_prices.csv with 2-row header.

    Returns:
        header_row0: List of ticker names (repeated 3x per ticker)
        header_row1: List of field types (#price, #low, #high)
        data_df: DataFrame with date column and data columns
    """
    # Read raw to preserve exact header format
    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if len(lines) < 2:
        raise ValueError("CSV must have at least 2 header rows")

    # Parse headers (semicolon-separated)
    header_row0 = lines[0].rstrip('\n').split(';')
    header_row1 = lines[1].rstrip('\n').split(';')

    # Read data rows
    data_rows = []
    for line in lines[2:]:
        if line.strip():
            data_rows.append(line.rstrip('\n').split(';'))

    # Create DataFrame
    num_cols = len(header_row0)
    data_df = pd.DataFrame(data_rows, columns=range(num_cols))

    return header_row0, header_row1, data_df


def read_excel_data_prices(excel_path: Path) -> Tuple[List[str], List[str], pd.DataFrame]:
    """
    Read data_prices sheet from Excel.

    Returns:
        header_row0: List of ticker names
        header_row1: List of field types
        data_df: DataFrame with date column and data columns
    """
    # Read without header to get raw data
    df = pd.read_excel(excel_path, sheet_name="data_prices", header=None)

    # Drop trailing all-NaN rows
    df = df.dropna(how='all')

    if len(df) < 3:
        raise ValueError("data_prices sheet must have at least 2 header rows + 1 data row")

    # Extract headers
    header_row0 = [str(x) if pd.notna(x) else "" for x in df.iloc[0].tolist()]
    header_row1 = [str(x) if pd.notna(x) else "" for x in df.iloc[1].tolist()]

    # Data starts from row 2
    data_df = df.iloc[2:].reset_index(drop=True)
    data_df.columns = range(len(header_row0))

    return header_row0, header_row1, data_df


def build_column_map(
    header_row0: List[str],
    header_row1: List[str]
) -> Dict[Tuple[str, str], int]:
    """
    Build mapping from (ticker, field) -> column index.

    Args:
        header_row0: Ticker names
        header_row1: Field types (#price, #low, #high)

    Returns:
        Dict mapping (ticker, field) tuples to column indices
    """
    col_map = {}
    for i, (ticker, field) in enumerate(zip(header_row0, header_row1)):
        ticker = str(ticker).strip()
        field = str(field).strip()
        if ticker and field:
            col_map[(ticker, field)] = i
    return col_map


def get_existing_dates(data_df: pd.DataFrame) -> Set[datetime]:
    """Get set of existing dates from CSV data (column 0)."""
    dates = set()
    for val in data_df[0]:
        dt = parse_date_from_csv(val)
        if dt:
            dates.add(dt)
    return dates


def normalize_excel_date(val) -> Optional[datetime]:
    """Convert Excel date value to datetime."""
    if pd.isna(val):
        return None
    if isinstance(val, datetime):
        return val
    if isinstance(val, pd.Timestamp):
        return val.to_pydatetime()
    # Try parsing as string
    try:
        return pd.to_datetime(val).to_pydatetime()
    except Exception:
        return None


def append_ytd_prices(
    excel_path: Path,
    master_path: Path,
    dry_run: bool = False
) -> None:
    """
    Append new dates from Excel data_prices to master_prices.csv.

    Args:
        excel_path: Path to ic_file.xlsx
        master_path: Path to master_prices.csv
        dry_run: If True, preview changes without writing
    """
    print(f"Reading master CSV: {master_path}")
    csv_header0, csv_header1, csv_data = read_master_csv(master_path)
    csv_col_map = build_column_map(csv_header0, csv_header1)

    print(f"Reading Excel data_prices: {excel_path}")
    xl_header0, xl_header1, xl_data = read_excel_data_prices(excel_path)
    xl_col_map = build_column_map(xl_header0, xl_header1)

    # Get existing dates
    existing_dates = get_existing_dates(csv_data)
    print(f"Existing dates in CSV: {len(existing_dates)}")

    # Find new dates in Excel
    new_rows = []
    for idx in range(len(xl_data)):
        row = xl_data.iloc[idx]
        date_val = normalize_excel_date(row[0])
        if date_val is None:
            continue
        # Normalize to date only (no time)
        date_val = datetime(date_val.year, date_val.month, date_val.day)
        if date_val not in existing_dates:
            new_rows.append((date_val, row))

    if not new_rows:
        print("No new dates to add. Master CSV is up to date.")
        return

    # Sort new rows by date
    new_rows.sort(key=lambda x: x[0])

    print(f"Found {len(new_rows)} new dates to add")
    print(f"  From: {new_rows[0][0].strftime('%Y-%m-%d')}")
    print(f"  To:   {new_rows[-1][0].strftime('%Y-%m-%d')}")

    # Check for ticker mismatches
    xl_tickers = set(t for t, f in xl_col_map.keys())
    csv_tickers = set(t for t, f in csv_col_map.keys())

    missing_in_csv = xl_tickers - csv_tickers
    if missing_in_csv:
        print(f"Warning: {len(missing_in_csv)} tickers in Excel not found in CSV (will be ignored):")
        for t in sorted(missing_in_csv)[:5]:
            print(f"  - {t}")
        if len(missing_in_csv) > 5:
            print(f"  ... and {len(missing_in_csv) - 5} more")

    # Build new CSV rows
    num_cols = len(csv_header0)
    new_csv_rows = []

    for date_val, xl_row in new_rows:
        # Start with empty row
        csv_row = [""] * num_cols
        # Set date in column 0
        csv_row[0] = format_date_for_csv(date_val)

        # Copy values for matching ticker+field combinations
        for (ticker, field), xl_idx in xl_col_map.items():
            if (ticker, field) in csv_col_map:
                csv_idx = csv_col_map[(ticker, field)]
                val = xl_row.iloc[xl_idx]
                if pd.notna(val):
                    # Format numbers without excessive decimals
                    if isinstance(val, float):
                        csv_row[csv_idx] = f"{val:.6g}"
                    else:
                        csv_row[csv_idx] = str(val)

        new_csv_rows.append(csv_row)

    if dry_run:
        print("\n=== DRY RUN - No changes will be written ===")
        print(f"Would add {len(new_csv_rows)} rows")
        print("First new row sample:")
        sample = new_csv_rows[0]
        non_empty = [(i, v) for i, v in enumerate(sample) if v][:10]
        for i, v in non_empty:
            ticker = csv_header0[i] if i < len(csv_header0) else "?"
            field = csv_header1[i] if i < len(csv_header1) else "?"
            print(f"  [{i}] {ticker}/{field}: {v}")
        return

    # Create backup
    backup_path = master_path.with_suffix('.csv.bak')
    print(f"Creating backup: {backup_path}")
    shutil.copy2(master_path, backup_path)

    # Combine existing and new data
    all_rows = []

    # Existing rows
    for idx in range(len(csv_data)):
        row = csv_data.iloc[idx].tolist()
        # Ensure row has correct number of columns
        while len(row) < num_cols:
            row.append("")
        all_rows.append(row)

    # New rows
    all_rows.extend(new_csv_rows)

    # Sort all rows by date
    def row_date(row):
        dt = parse_date_from_csv(row[0])
        return dt if dt else datetime.min

    all_rows.sort(key=row_date)

    # Write updated CSV
    print(f"Writing updated CSV: {master_path}")
    with open(master_path, 'w', encoding='utf-8') as f:
        # Write headers exactly as they were
        f.write(';'.join(csv_header0) + '\n')
        f.write(';'.join(csv_header1) + '\n')
        # Write data rows
        for row in all_rows:
            f.write(';'.join(str(v) for v in row) + '\n')

    print(f"Added {len(new_csv_rows)} new dates "
          f"({new_rows[0][0].strftime('%Y-%m-%d')} to {new_rows[-1][0].strftime('%Y-%m-%d')}). "
          f"Total rows: {len(all_rows)}")


def main():
    parser = argparse.ArgumentParser(
        description="Append YTD prices from Excel to master_prices.csv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Append new dates from Excel to CSV
    python append_ytd_prices.py \\
        --excel /path/to/ic_file.xlsx \\
        --master /path/to/master_prices.csv

    # Preview changes without writing
    python append_ytd_prices.py \\
        --excel /path/to/ic_file.xlsx \\
        --master /path/to/master_prices.csv \\
        --dry-run
"""
    )
    parser.add_argument(
        "--excel", "-e",
        required=True,
        help="Path to ic_file.xlsx with data_prices sheet"
    )
    parser.add_argument(
        "--master", "-m",
        required=True,
        help="Path to master_prices.csv"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Preview changes without writing"
    )

    args = parser.parse_args()

    excel_path = Path(args.excel)
    master_path = Path(args.master)

    if not excel_path.exists():
        print(f"Error: Excel file not found: {excel_path}")
        sys.exit(1)

    if not master_path.exists():
        print(f"Error: Master CSV not found: {master_path}")
        sys.exit(1)

    append_ytd_prices(excel_path, master_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
