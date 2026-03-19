"""Result caching system for intelligent reuse and selective re-execution.

This module provides the RunCache class which manages:
- Saving complete agent runs with all step results
- Finding similar past runs using prompt similarity
- Loading cached results for selective reuse
"""

import json
import os
import re
import hashlib
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path


class RunCache:
    """Manages caching and retrieval of agent run results.

    Saves all N results from best-of-n per step (not just the best),
    enabling re-selection with different evaluation criteria later.

    Storage structure:
        cache/
        ├── index.json           # Fast lookup: all runs with prompts and timestamps
        ├── <run_id_1>/
        │   ├── metadata.json    # Run info: prompt, config, final result
        │   ├── lookup_sales_data.json     # Array of N results
        │   ├── analyzing_data.json        # Array of N results
        │   └── create_visualization.json  # Array of N results
        └── <run_id_2>/
            └── ...

    Attributes:
        cache_dir: Path to the cache directory
        index: In-memory index of all cached runs
    """

    def __init__(self, cache_dir: str = "./cache/agent_runs"):
        """Initialize the cache manager.

        Args:
            cache_dir: Directory to store cached runs
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.cache_dir / "index.json"
        self.index = self._load_index()

    def _load_index(self) -> Dict:
        """Load the cache index from disk."""
        if self.index_path.exists():
            try:
                with open(self.index_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {"runs": []}
        return {"runs": []}

    def _save_index(self) -> None:
        """Save the cache index to disk."""
        with open(self.index_path, 'w') as f:
            json.dump(self.index, f, indent=2)

    def _compute_prompt_hash(self, prompt: str) -> str:
        """Compute hash of prompt for quick lookup."""
        return hashlib.md5(prompt.encode()).hexdigest()

    def _serialize_result(self, result: Any) -> Any:
        """Serialize a result for JSON storage.

        Handles pandas DataFrames and other non-serializable types.
        DataFrames are stored as {'__dataframe__': True, 'records': [...]}
        so they can be faithfully restored on load.
        """
        if result is None:
            return None
        if isinstance(result, dict):
            serialized = {}
            for k, v in result.items():
                if k == 'data_df':
                    if v is not None:
                        try:
                            serialized[k] = {
                                '__dataframe__': True,
                                'records': v.to_dict(orient='records'),
                            }
                        except Exception:
                            pass  # Skip if DataFrame can't be serialized
                    continue
                serialized[k] = self._serialize_result(v)
            return serialized
        if isinstance(result, list):
            return [self._serialize_result(item) for item in result]
        # Try to convert to basic types
        try:
            json.dumps(result)
            return result
        except (TypeError, ValueError):
            return str(result)

    def _deserialize_result(self, result: Any) -> Any:
        """Restore a result loaded from JSON, reconstructing DataFrames.

        Converts {'__dataframe__': True, 'records': [...]} back to
        a pandas DataFrame.
        """
        if result is None:
            return None
        if isinstance(result, dict):
            if result.get('__dataframe__') is True:
                try:
                    import pandas as pd
                    return pd.DataFrame(result.get('records', []))
                except Exception:
                    return None
            return {k: self._deserialize_result(v) for k, v in result.items()}
        if isinstance(result, list):
            return [self._deserialize_result(item) for item in result]
        return result

    def save_run(
        self,
        run_id: str,
        prompt: str,
        agent_config: Dict,
        step_results: Dict[str, List[Dict]],
        final_result: Dict,
        metadata: Optional[Dict] = None
    ) -> None:
        """Save a complete agent run with all step results.

        Args:
            run_id: Unique identifier for this run
            prompt: User prompt that initiated this run
            agent_config: Agent configuration used (as dict)
            step_results: Dict mapping step_name -> list of N results
            final_result: The final result returned by the agent
            metadata: Additional metadata (lookup_only, no_vis, etc.)
        """
        run_dir = self.cache_dir / run_id
        run_dir.mkdir(exist_ok=True)

        # Save step results (all N runs per step)
        for step_name, results_list in step_results.items():
            step_file = run_dir / f"{step_name}.json"
            serialized = [self._serialize_result(r) for r in results_list]
            with open(step_file, 'w') as f:
                json.dump(serialized, f, indent=2, default=str)

        # Save run metadata
        run_meta = {
            "run_id": run_id,
            "prompt": prompt,
            "prompt_hash": self._compute_prompt_hash(prompt),
            "timestamp": datetime.now().isoformat(),
            "agent_config": agent_config,
            "final_result": self._serialize_result(final_result),
            "metadata": metadata or {},
        }

        meta_file = run_dir / "metadata.json"
        with open(meta_file, 'w') as f:
            json.dump(run_meta, f, indent=2, default=str)

        # Update index (avoid duplicates)
        existing_ids = {r["run_id"] for r in self.index["runs"]}
        if run_id not in existing_ids:
            self.index["runs"].append({
                "run_id": run_id,
                "prompt": prompt,
                "prompt_hash": run_meta["prompt_hash"],
                "timestamp": run_meta["timestamp"],
            })
            self._save_index()

    def find_similar_runs(
        self,
        prompt: str,
        top_k: int = 5,
        similarity_threshold: float = 0.3
    ) -> List[str]:
        """Find runs with similar prompts using keyword matching.

        Uses Jaccard similarity on word sets. For production use,
        consider upgrading to embedding-based similarity.

        Args:
            prompt: Prompt to find similar runs for
            top_k: Maximum number of similar runs to return
            similarity_threshold: Minimum similarity score (0-1)

        Returns:
            List of run_ids sorted by similarity (most similar first)
        """
        def _tokenize(text: str):
            # Split on whitespace and underscores, strip punctuation from each token
            tokens = re.split(r'[\s_]+', text.lower())
            return set(t.strip('.,?!;:()[]"\'') for t in tokens if t.strip('.,?!;:()[]"\''))

        prompt_words = _tokenize(prompt)

        similarities = []
        for run_meta in self.index["runs"]:
            cached_prompt = run_meta.get("prompt", "")
            cached_words = _tokenize(cached_prompt)

            # Jaccard similarity
            intersection = len(prompt_words & cached_words)
            union = len(prompt_words | cached_words)
            similarity = intersection / union if union > 0 else 0.0

            similarities.append((run_meta["run_id"], similarity))

        # Sort by similarity descending and filter by threshold
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [
            run_id for run_id, sim in similarities[:top_k]
            if sim >= similarity_threshold
        ]

    def find_exact_match(self, prompt: str) -> Optional[str]:
        """Find a run with the exact same prompt.

        Args:
            prompt: Prompt to match

        Returns:
            run_id if exact match found, None otherwise
        """
        prompt_hash = self._compute_prompt_hash(prompt)
        for run_meta in self.index["runs"]:
            if run_meta.get("prompt_hash") == prompt_hash:
                # Verify actual prompt matches (hash collision protection)
                if run_meta.get("prompt") == prompt:
                    return run_meta["run_id"]
        return None

    def load_step_results(
        self,
        run_id: str,
        step_name: str
    ) -> Optional[List[Dict]]:
        """Load all N results for a specific step from a cached run.

        Args:
            run_id: ID of the cached run
            step_name: Name of the step to load results for

        Returns:
            List of result dictionaries, or None if not found
        """
        step_file = self.cache_dir / run_id / f"{step_name}.json"
        if step_file.exists():
            try:
                with open(step_file, 'r') as f:
                    raw = json.load(f)
                return [self._deserialize_result(r) for r in raw]
            except (json.JSONDecodeError, IOError):
                return None
        return None

    def load_all_step_results(self, run_id: str) -> Dict[str, List[Dict]]:
        """Load all step results from a cached run.

        Args:
            run_id: ID of the cached run

        Returns:
            Dict mapping step_name -> list of results
        """
        run_dir = self.cache_dir / run_id
        if not run_dir.exists():
            return {}

        results = {}
        step_names = [
            "decide_tool",
            "lookup_sales_data",
            "analyzing_data",
            "create_visualization"
        ]

        for step_name in step_names:
            step_results = self.load_step_results(run_id, step_name)
            if step_results:
                results[step_name] = step_results

        return results

    def load_run_metadata(self, run_id: str) -> Optional[Dict]:
        """Load metadata for a cached run.

        Args:
            run_id: ID of the cached run

        Returns:
            Metadata dict, or None if not found
        """
        meta_file = self.cache_dir / run_id / "metadata.json"
        if meta_file.exists():
            try:
                with open(meta_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return None
        return None

    def list_runs(self, limit: int = 20) -> List[Dict]:
        """List recent cached runs.

        Args:
            limit: Maximum number of runs to return

        Returns:
            List of run metadata dicts, most recent first
        """
        # Sort by timestamp descending
        runs = sorted(
            self.index["runs"],
            key=lambda r: r.get("timestamp", ""),
            reverse=True
        )
        return runs[:limit]

    def delete_run(self, run_id: str) -> bool:
        """Delete a cached run.

        Args:
            run_id: ID of the run to delete

        Returns:
            True if deleted, False if not found
        """
        import shutil

        run_dir = self.cache_dir / run_id
        if run_dir.exists():
            shutil.rmtree(run_dir)

            # Update index
            self.index["runs"] = [
                r for r in self.index["runs"]
                if r["run_id"] != run_id
            ]
            self._save_index()
            return True
        return False

    def clear_cache(self) -> int:
        """Clear all cached runs.

        Returns:
            Number of runs deleted
        """
        import shutil

        count = len(self.index["runs"])

        # Delete all run directories
        for run_meta in self.index["runs"]:
            run_dir = self.cache_dir / run_meta["run_id"]
            if run_dir.exists():
                shutil.rmtree(run_dir)

        # Reset index
        self.index = {"runs": []}
        self._save_index()

        return count

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache.

        Returns:
            Dict with cache statistics
        """
        total_runs = len(self.index["runs"])

        # Calculate total size
        total_size = 0
        for run_meta in self.index["runs"]:
            run_dir = self.cache_dir / run_meta["run_id"]
            if run_dir.exists():
                for file in run_dir.iterdir():
                    total_size += file.stat().st_size

        return {
            "total_runs": total_runs,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "cache_dir": str(self.cache_dir),
        }
