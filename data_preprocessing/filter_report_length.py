"""
Standalone Report-Length Preprocessing Script
=============================================
Tokenizes the `findings` column of a multiview CSV using the BioGPT tokenizer
(`microsoft/biogpt`) with `add_special_tokens=False` and filters out rows whose
content length exceeds `MAX_LENGTH - 2` (126 tokens).

Reserving 2 slots for `<bos>` and `<eos>` ensures that wrapping a report as
`[bos_id] + content_ids + [eos_id]` fits within the 128-token decoder budget
without truncating the `<eos>` token.

What this script does
----------------------
1. Loads the CSV.
2. Applies the identical fallback logic the training script uses for missing
   findings ("normal findings." if NaN/empty).
3. Tokenizes each report's content exactly as training does (no bos/eos).
4. Reports how many rows have content token length > 126.
5. Drops those rows.
6. Saves the filtered CSV to a new path — this filtered CSV is then safe to
   point TRAIN_CSV / VAL_CSV / TEST_CSV at in the training script, since every
   remaining report is guaranteed to fit within budget without truncation.
"""

import os
import pandas as pd
from transformers import BioGptTokenizer

INPUT_CSV_PATH  = "/kaggle/input/datasets/divyanshu134/final-multiview-2-encoders/Multimodal_val_multiview_final.csv"

KAGGLE_WORKING_DIR = "/kaggle/working"
OUTPUT_CSV_FILENAME = "Multimodal_train_multiview_final_lenfiltered.csv"
OUTPUT_CSV_PATH = os.path.join(KAGGLE_WORKING_DIR, OUTPUT_CSV_FILENAME)

# Must match the training script's ChestXrayDataset(max_length=...) value.
MAX_LENGTH = 128
# Content budget = MAX_LENGTH - 2 (slots reserved for <bos> and <eos>).
MAX_CONTENT_LEN = MAX_LENGTH - 2   # = 126
# ============================================================================


def main():
    # refuse to run if OUTPUT_CSV_PATH was edited to point somewhere
    if not os.path.abspath(OUTPUT_CSV_PATH).startswith(os.path.abspath(KAGGLE_WORKING_DIR)):
        raise ValueError(
            f"OUTPUT_CSV_PATH ('{OUTPUT_CSV_PATH}') is not inside "
            f"KAGGLE_WORKING_DIR ('{KAGGLE_WORKING_DIR}'). /kaggle/input is "
            f"read-only, so the filtered CSV must be saved under "
            f"/kaggle/working. Fix OUTPUT_CSV_FILENAME instead of the path."
        )
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)

    # --- Tokenizer setup: identical to the training script ---
    report_tokenizer = BioGptTokenizer.from_pretrained('microsoft/biogpt')
    report_tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    print(f"Report tokenizer (BioGPT) — vocab size: {len(report_tokenizer)}, "
          f"PAD id: {report_tokenizer.pad_token_id}")

    df = pd.read_csv(INPUT_CSV_PATH)
    print(f"Loaded {len(df)} rows from {INPUT_CSV_PATH}")

    if 'findings' not in df.columns:
        raise ValueError(
            f"Expected a 'findings' column in the CSV, found columns: {list(df.columns)}"
        )

    content_lengths = []
    for _, row in df.iterrows():
        raw_findings = row['findings']
        if pd.isna(raw_findings) or str(raw_findings).strip() == '':
            report = "normal findings."
        else:
            report = str(raw_findings).strip()

        encoded_report = report_tokenizer(
            report,
            add_special_tokens=False,
            truncation=False
        )['input_ids']

        content_lengths.append(len(encoded_report))

    df['_content_token_len'] = content_lengths

    over_limit_mask = df['_content_token_len'] > MAX_CONTENT_LEN
    num_over = int(over_limit_mask.sum())
    total = len(df)
    print(f"\nReports with content length > {MAX_CONTENT_LEN} tokens "
          f"(would have lost <eos> under the old bug): "
          f"{num_over} / {total} ({100 * num_over / total:.2f}%)")

    if num_over > 0:
        print("\nExample over-limit rows (first 5, by content length):")
        preview_cols = [c for c in ['uid', 'findings', '_content_token_len'] if c in df.columns]
        print(
            df.loc[over_limit_mask, preview_cols]
              .sort_values('_content_token_len', ascending=False)
              .head(5)
              .to_string(index=False)
        )

    df_filtered = (
        df.loc[~over_limit_mask]
          .drop(columns=['_content_token_len'])
          .reset_index(drop=True)
    )
    print(f"\nRows remaining after filtering: {len(df_filtered)} / {total}")

    df_filtered.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Saved filtered CSV to {OUTPUT_CSV_PATH}")


if __name__ == '__main__':
    main()