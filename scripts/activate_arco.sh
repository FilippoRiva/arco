#!/usr/bin/env bash
# Load ARCO's virtual environment and provider credentials into the current shell.
# This file must be sourced:
#   source ./activate_arco.sh

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Usage: source ./activate_arco.sh" >&2
  exit 1
fi

_ARCO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "> Loading the virtual environment..."
source "${_ARCO_ROOT}/.venv/bin/activate"
echo ""
source "${_ARCO_ROOT}/load_env_from_keyring.sh"

unset _ARCO_ROOT
