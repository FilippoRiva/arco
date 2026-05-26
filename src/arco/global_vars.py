"""Global variables and shared constants.

This module centralizes application-wide variables, constants, and shared
state used across the project. It provides a single source of truth for:
- Shared paths and resource locations

The module helps ensure consistency and avoids duplicated configuration
definitions across components.
"""

import os

# Path to parquet files
DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "Store_Sales_Price_Elasticity_Promotions_Data.parquet"
)

# codecarbon availability check
try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as _:
    EmissionsTracker = None
    CODECARBON_AVAILABLE = False

# Timeout for Ollama LLM HTTP requests (seconds).
# A 33B model at ~30 tok/s with 4000 max tokens needs ~130s; 600s is a 4× safety margin.
OLLAMA_REQUEST_TIMEOUT: int = 600
