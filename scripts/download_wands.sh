#!/usr/bin/env bash

set -euo pipefail

wands_commit="3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5"
wands_base_url="https://raw.githubusercontent.com/wayfair/WANDS/${wands_commit}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
wands_data_dir="${project_root}/data/raw/wands"
wands_checksum_file="${project_root}/data/WANDS_SHA256SUMS"
wands_tmp_dir="$(mktemp -d /tmp/commercerag-wands.XXXXXX)"

cleanup() {
  find "${wands_tmp_dir}" -mindepth 1 -delete 2>/dev/null || true
  rmdir "${wands_tmp_dir}" 2>/dev/null || true
}
trap cleanup EXIT

download_file() {
  local remote_path="$1"
  local output_name="$2"

  echo "下载 ${output_name}"
  curl \
    --retry 5 \
    --retry-all-errors \
    --retry-delay 1 \
    --connect-timeout 15 \
    --max-time 300 \
    --fail \
    --silent \
    --show-error \
    --location \
    "${wands_base_url}/${remote_path}" \
    --output "${wands_tmp_dir}/${output_name}"
}

assert_line_count() {
  local file_name="$1"
  local expected_count="$2"
  local actual_count

  actual_count="$(wc -l < "${wands_tmp_dir}/${file_name}" | tr -d ' ')"
  if [[ "${actual_count}" != "${expected_count}" ]]; then
    echo "${file_name} 行数校验失败：期望 ${expected_count}，实际 ${actual_count}" >&2
    exit 1
  fi
}

download_file "dataset/product.csv" "product.csv"
download_file "dataset/query.csv" "query.csv"
download_file "dataset/label.csv" "label.csv"
download_file "LICENSE" "LICENSE"
download_file "README.md" "README.md"
download_file \
  "Product%20Search%20Relevance%20Annotation%20Guidelines.pdf" \
  "Product Search Relevance Annotation Guidelines.pdf"

echo "执行 SHA-256 校验"
(
  cd "${wands_tmp_dir}"
  shasum -a 256 -c "${wands_checksum_file}"
)

assert_line_count "product.csv" "42995"
assert_line_count "query.csv" "481"
assert_line_count "label.csv" "233449"

expected_product_header=$'product_id\tproduct_name\tproduct_class\tcategory hierarchy\tproduct_description\tproduct_features\trating_count\taverage_rating\treview_count'
expected_query_header=$'query_id\tquery\tquery_class'
expected_label_header=$'id\tquery_id\tproduct_id\tlabel'

[[ "$(head -n 1 "${wands_tmp_dir}/product.csv")" == "${expected_product_header}" ]]
[[ "$(head -n 1 "${wands_tmp_dir}/query.csv")" == "${expected_query_header}" ]]
[[ "$(head -n 1 "${wands_tmp_dir}/label.csv")" == "${expected_label_header}" ]]

mkdir -p "${wands_data_dir}"
for file_name in \
  product.csv \
  query.csv \
  label.csv \
  LICENSE \
  README.md \
  "Product Search Relevance Annotation Guidelines.pdf"; do
  mv "${wands_tmp_dir}/${file_name}" "${wands_data_dir}/${file_name}"
done

echo "WANDS 数据下载并校验完成：${wands_data_dir}"
