#!/bin/bash

echo "> Loading Secrets..."

load_secret() {
  local var_name=$1
  shift

  local value
  value=$(secret-tool lookup "$@")

  if [ -z "$value" ]; then
    echo "    $var_name : ❌"
  else
    export "$var_name=$value"
    echo "    $var_name : ✅"
  fi
}

# Define secrets (var_name + attributes)
load_secret OPENAI_API_KEY app arco provider openai
load_secret OPENROUTER_API_KEY app arco provider openrouter
