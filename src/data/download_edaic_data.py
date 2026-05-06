from pathlib import Path
import subprocess
import re


PROJECT_DIR = Path("/content/drive/MyDrive/master/courses/AI_in_medicine/AI_MD_Project")
BASE_URL = "https://dcapswoz.ict.usc.edu/wwwedaic"

RAW_DIR = PROJECT_DIR / "data" / "raw" / "edaic"
MANUAL_DIR = RAW_DIR / "manual"
METADATA_DIR = RAW_DIR / "metadata"
LABELS_DIR = RAW_DIR / "labels"
TRANSCRIPTS_DIR = RAW_DIR / "transcripts"
OUTPUTS_DIR = PROJECT_DIR / "outputs"


def run_cmd(cmd: str, check: bool = True):
    print(f"\nRunning:\n{cmd}")
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True)

    if result.stdout:
        print(result.stdout[-2000:])
    if result.stderr:
        print(result.stderr[-2000:])

    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed with return code {result.returncode}")

    return result


def create_dirs():
    for d in [MANUAL_DIR, METADATA_DIR, LABELS_DIR, TRANSCRIPTS_DIR, OUTPUTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def download_small_files():
    downloads = [
        (f"{BASE_URL}/E-DAIC%20Manual.pdf", MANUAL_DIR),
        (f"{BASE_URL}/metadata_mapped.csv", METADATA_DIR),
        (f"{BASE_URL}/labels2019.tar.gz", LABELS_DIR),
    ]

    for url, out_dir in downloads:
        cmd = f'wget -nc "{url}" -P "{out_dir}"'
        run_cmd(cmd, check=True)


def extract_labels():
    labels_tar = LABELS_DIR / "labels2019.tar.gz"

    if not labels_tar.exists():
        raise FileNotFoundError(f"Missing labels archive: {labels_tar}")

    cmd = f'tar -xzf "{labels_tar}" -C "{LABELS_DIR}"'
    run_cmd(cmd, check=True)

    label_files = sorted([p for p in LABELS_DIR.rglob("*") if p.is_file()])
    print("\nLabel files:")
    for p in label_files:
        print(p.relative_to(PROJECT_DIR))


def create_tar_url_file():
    index_path = OUTPUTS_DIR / "edaic_data_index.html"
    urls_file = OUTPUTS_DIR / "edaic_tar_urls.txt"

    cmd = f'wget -q -O "{index_path}" "{BASE_URL}/data/"'
    run_cmd(cmd, check=True)

    html = index_path.read_text(errors="ignore")
    tar_links = re.findall(r'href="([^"]+\.tar\.gz)"', html)

    if not tar_links:
        raise RuntimeError("No tar.gz links found in E-DAIC data index.")

    tar_urls = [f"{BASE_URL}/data/{link}" for link in tar_links]
    urls_file.write_text("\n".join(tar_urls), encoding="utf-8")

    print(f"\nNumber of tar.gz files found: {len(tar_urls)}")
    print(f"Saved URL file: {urls_file}")


def create_extraction_script():
    urls_file = OUTPUTS_DIR / "edaic_tar_urls.txt"
    script_path = OUTPUTS_DIR / "extract_transcripts_fast_safe.sh"
    log_dir = OUTPUTS_DIR / "download_logs"

    log_dir.mkdir(parents=True, exist_ok=True)

    script = f'''#!/usr/bin/env bash
set -u

URL_FILE="{urls_file}"
OUT_DIR="{TRANSCRIPTS_DIR}"
LOG_DIR="{log_dir}"
TMP_ROOT="/content/edaic_tmp_extract"

if [ ! -f "$URL_FILE" ]; then
  echo "ERROR: URL file does not exist: $URL_FILE"
  exit 1
fi

mkdir -p "$OUT_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$TMP_ROOT"

FAILED_FILE="$LOG_DIR/failed_urls.txt"
rm -f "$FAILED_FILE"

extract_one() {{
  url="$1"

  if [ -z "$url" ]; then
    return 0
  fi

  fname=$(basename "$url")
  pid="${{fname%%_P.tar.gz}}"

  out_file="$OUT_DIR/${{pid}}_Transcript.csv"
  tmp_dir="$TMP_ROOT/${{pid}}"
  archive="$tmp_dir/${{pid}}_P.tar.gz"
  inner_file="${{pid}}_P/${{pid}}_Transcript.csv"

  if [ -f "$out_file" ]; then
    echo "SKIP $pid"
    return 0
  fi

  rm -rf "$tmp_dir"
  mkdir -p "$tmp_dir"

  echo "START $pid"

  wget \\
    --tries=3 \\
    --timeout=60 \\
    --read-timeout=60 \\
    --continue \\
    -q \\
    -O "$archive" \\
    "$url"

  status=$?

  if [ $status -ne 0 ]; then
    echo "FAIL_DOWNLOAD $pid"
    echo "$url" >> "$FAILED_FILE"
    rm -rf "$tmp_dir"
    return 0
  fi

  tar -xzf "$archive" -C "$tmp_dir" "$inner_file" \\
    > "$LOG_DIR/${{pid}}.out.log" \\
    2> "$LOG_DIR/${{pid}}.err.log"

  status=$?

  if [ $status -ne 0 ]; then
    echo "FAIL_EXTRACT $pid"
    echo "$url" >> "$FAILED_FILE"
    rm -rf "$tmp_dir"
    return 0
  fi

  if [ -f "$tmp_dir/$inner_file" ]; then
    cp "$tmp_dir/$inner_file" "$out_file"
    echo "DONE $pid"
  else
    echo "FAIL_MISSING_FILE $pid"
    echo "$url" >> "$FAILED_FILE"
  fi

  rm -rf "$tmp_dir"
}}

export -f extract_one
export OUT_DIR
export LOG_DIR
export TMP_ROOT
export FAILED_FILE

cat "$URL_FILE" | xargs -n 1 -P 3 bash -c 'extract_one "$0"'
'''

    script_path.write_text(script, encoding="utf-8")
    print(f"\nExtraction script written to: {script_path}")
    return script_path


def run_extraction_script(script_path: Path):
    cmd = f'bash "{script_path}"'
    run_cmd(cmd, check=True)


def final_check():
    transcript_files = sorted(TRANSCRIPTS_DIR.glob("*_Transcript.csv"))
    print(f"\nNumber of transcript files: {len(transcript_files)}")

    print("\nFirst files:")
    for p in transcript_files[:10]:
        print(p.name)

    failed_file = OUTPUTS_DIR / "download_logs" / "failed_urls.txt"
    if failed_file.exists():
        failed_urls = failed_file.read_text().splitlines()
        print(f"\nFailed URLs: {len(failed_urls)}")
    else:
        print("\nNo failed URLs file found.")


def main():
    create_dirs()
    download_small_files()
    extract_labels()
    create_tar_url_file()
    script_path = create_extraction_script()
    run_extraction_script(script_path)
    final_check()


if __name__ == "__main__":
    main()
