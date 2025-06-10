use crate::{errors::CsvParseError, Float};
use std::io::BufRead;

/// Parses a CSV file of asset returns given through a reader handle. This returns the asset names,
/// the dates under consideration, and the rows of return values.
///
/// For further details, see [`AlgorithmHandle::new`].
pub fn parse_csv(
    mut assets_csv: impl BufRead,
) -> Result<(Vec<String>, Vec<String>, Vec<Vec<Float>>), CsvParseError> {
    // Read the header line to get the asset names, skipping the first one (should be the date
    // column)
    let mut header_line = String::new();
    assets_csv.read_line(&mut header_line)?;
    let asset_names = header_line
        .split(',')
        .map(|s| s.trim().to_string())
        .skip(1)
        .collect::<Vec<_>>();

    // Stores whether or not we've seen the start of each asset (i.e. the first non-0.0 value)
    let mut seen_start: Vec<bool> = vec![false; asset_names.len()];
    // Stores the raw asset data, parsed as `Float`s
    let mut asset_rows: Vec<Vec<Float>> = Vec::new();
    // Stores the dates for which we have asset data
    let mut dates = Vec::new();

    // Read the CSV file line by line
    for (row_idx, line) in assets_csv.lines().enumerate() {
        let line = line?;
        // Split the line by commas, skipping the first column (the date)
        let mut parts = line.split(',');
        // Read the date (don't add it yet though, just in case we're in a dropping phase)
        let date = parts.next().unwrap().to_string();

        let mut row: Vec<Float> = Vec::new();
        let mut should_drop = true; // Starts true, false once we see non-zero
        for (col_idx, value_str) in parts.enumerate() {
            // Parse the value as an Float
            let value =
                value_str
                    .trim()
                    .parse::<Float>()
                    .map_err(|source| CsvParseError::NotFloat {
                        row: row_idx,
                        col: col_idx,
                        source,
                    })?;
            if value.is_nan() {
                return Err(CsvParseError::NaN {
                    row: row_idx,
                    col: col_idx,
                });
            }
            row.push(value);

            // If we've found a 0.0 and we still haven't seen the start of non-zero values, drop
            // this row. If the value is non-zero, we shouldn't drop this row on account of this
            // column. And if we see a 0.0 after seeing the start, it doesn't matter (but don't
            // change `should_drop` from `true` to `false`, if the row is all zeroes it's a holiday
            // and we should drop it).
            if value == 0.0 && !seen_start[col_idx] {
                should_drop = true;
                break;
            } else if value != 0.0 {
                should_drop = false;
                seen_start[col_idx] = true;
            }
        }

        if !should_drop {
            asset_rows.push(row);
            dates.push(date);
        }
    }

    Ok((asset_names, dates, asset_rows))
}
