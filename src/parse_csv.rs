use ordered_float::NotNan;
use std::{io::BufRead, num::ParseFloatError};
use thiserror::Error;

/// The maximum number of consecutive zeroes we'll allow in asset data before treating an asset as
/// no longer returning and dropping the block for all assets. See [`AlgorithmInstance::new`] for
/// details. To be clear, precisely this many zeroes *will* be allowed, and then one more will
/// trigger the drop.
const MAX_CONSECUTIVE_ZEROES: i8 = 3;

/// An error encountered while trying to parse a CSV file of assets.
#[derive(Error, Debug)]
pub enum CsvParseError {
    #[error("io error occurred while reading csv file")]
    Io(#[from] std::io::Error),
    #[error("failed to parse return at row #{row}, column #{col} as f64")]
    NotF64 {
        row: usize,
        col: usize,
        #[source]
        source: ParseFloatError,
    },
    #[error("found NaN at row #{row}, column #{col}")]
    NaN { row: usize, col: usize },
}

/// Parses a CSV file of asset returns given through a reader handle. This returns the asset names,
/// the dates under consideration, and the rows of return values.
///
/// For further details, see [`AlgorithmHandle::new`].
pub fn parse_csv(
    mut assets_csv: impl BufRead,
) -> Result<(Vec<String>, Vec<String>, Vec<Vec<NotNan<f64>>>), CsvParseError> {
    // Read the header line to get the asset names, skipping the first one (should be the date
    // column)
    let mut header_line = String::new();
    assets_csv.read_line(&mut header_line)?;
    let asset_names = header_line
        .split(',')
        .map(|s| s.trim().to_string())
        .skip(1)
        .collect::<Vec<_>>();

    // Stores the number of consecutive 0.0s we've seen for each asset. Once this passes the
    // threshold, we'll drop the entire section of consecutives retroactively across all
    // assets, and set this to `-1`, meaning we'll drop continually until we see a non-zero
    // value again.
    let mut num_zeroes: Vec<i8> = vec![0; asset_names.len()];
    // Stores the raw asset data, parsed as non-NaN `f64`s
    let mut asset_rows: Vec<Vec<NotNan<f64>>> = Vec::new();
    // Stores the dates for which we have asset data
    let mut dates = Vec::new();

    // Read the CSV file line by line
    for (row_idx, line) in assets_csv.lines().enumerate() {
        let line = line?;
        // Split the line by commas, skipping the first column (the date)
        let mut parts = line.split(',');
        // Read the date (don't add it yet though, just in case we're in a dropping phase)
        let date = parts.next().unwrap().to_string();

        let mut row: Vec<NotNan<f64>> = Vec::new();
        let mut should_drop = true; // Starts true, false once we see non-zero
        for (col_idx, value_str) in parts.enumerate() {
            // Parse the value as an f64
            let value =
                value_str
                    .trim()
                    .parse::<f64>()
                    .map_err(|source| CsvParseError::NotF64 {
                        row: row_idx,
                        col: col_idx,
                        source,
                    })?;
            let value = NotNan::new(value).map_err(|_| CsvParseError::NaN {
                row: row_idx,
                col: col_idx,
            })?;
            row.push(value);

            // If we've found a 0.0, record it for this column
            if value == 0.0 {
                // If we've passed the threshold, drop the last `num_zeroes[col_idx]` rows
                // entirely (this will be the number of previous rows already added where we
                // had zeroes in this column), and skip adding this one. Also set the zero
                // counters for all other columns back to 0 (as this is threshold-based and
                // we're the first column to pass the threshold, the others are all guaranteed
                // to be less, so any consecutive zeroes they'd been watching will now be
                // irrelevant).
                //
                // Alternately, if our tracker is `-1`, we've already seen too many consecutive
                // zeroes and dropped them, so if we're seeing another zero, we should just
                // drop this whole row immediately.
                if num_zeroes[col_idx] + 1 > MAX_CONSECUTIVE_ZEROES {
                    // Drop bad rows for the assets and dates
                    for _ in 0..num_zeroes[col_idx] {
                        asset_rows.pop();
                        dates.pop();
                    }

                    // Clear for the other columns
                    num_zeroes.iter_mut().for_each(|n| *n = 0);
                    // Set ours to `-1` so we progressively drop from now until the next
                    // non-zero
                    num_zeroes[col_idx] = -1;
                    // We'll now see no more columns, and immediately exit the loop for this
                    // row without recording it or the associated date
                    should_drop = true;
                    break;
                } else if num_zeroes[col_idx] == -1 {
                    should_drop = true;
                    break;
                } else {
                    // Otherwise we've seen a few, but not too many yet, keep recording
                    num_zeroes[col_idx] += 1;
                }
            } else {
                // Not a zero, record that the number of consecutive zeroes is definitely 0 for
                // this column
                should_drop = false;
                num_zeroes[col_idx] = 0;
            }
        }

        if !should_drop {
            asset_rows.push(row);
            dates.push(date);
        }
    }

    Ok((asset_names, dates, asset_rows))
}
