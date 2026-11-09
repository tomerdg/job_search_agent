#!/usr/bin/env bash
# Encrypt / decrypt a file with a passphrase (AES-256, PBKDF2), so job history can be stored
# in Actions cache and artifacts of a public repo without anyone else being able to read it.
#
#   DB_PASSPHRASE=... scripts/dbcrypt.sh encrypt <in> <out>
#   DB_PASSPHRASE=... scripts/dbcrypt.sh decrypt <in> <out>
#
# Works with OpenSSL 1.1.1+ / 3.x and with the LibreSSL that ships with macOS.
set -euo pipefail

mode="${1:-}"; in="${2:-}"; out="${3:-}"
if [[ -z "$mode" || -z "$in" || -z "$out" ]]; then
  echo "usage: DB_PASSPHRASE=... $0 encrypt|decrypt <in> <out>" >&2; exit 2
fi
if [[ -z "${DB_PASSPHRASE:-}" ]]; then
  echo "DB_PASSPHRASE is not set" >&2; exit 2
fi

args=(enc -aes-256-cbc -pbkdf2 -iter 600000 -md sha256 -salt -pass env:DB_PASSPHRASE -in "$in" -out "$out")
case "$mode" in
  encrypt) openssl "${args[@]}" ;;
  decrypt) openssl "${args[@]}" -d ;;
  *) echo "mode must be encrypt or decrypt" >&2; exit 2 ;;
esac
