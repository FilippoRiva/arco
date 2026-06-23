"""Result caching system for intelligent reuse and selective re-execution.

This module provides the RunCache class which manages:
- Saving complete agent runs with all step results
- Finding similar past runs using prompt similarity
- Loading cached results for selective reuse
"""

import hashlib
import json
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, TYPE_CHECKING

from arco.core import State, AgentType

if TYPE_CHECKING:
    from arco.core import ArcoConfig, Answer


def _compute_prompt_hash(prompt: str) -> str:
    """Compute hash of prompt for quick lookup."""
    return hashlib.md5(prompt.encode()).hexdigest()


class RunCache:
    """Manages caching and retrieval of agent run results.

    Saves all N results from best-of-n per step (not just the best),
    enabling re-selection with different evaluation criteria later.

    Storage structure:
        cache/
        ├── index.json           # Fast lookup: all runs with prompts and timestamps
        ├── <run_id_1>/
        │   ├── metadata.json    # Run info: prompt, config, final result
        │   ├── Retriever.json     # Array of N results
        │   ├── Analyzer.json        # Array of N results
        │   └── Visualizer.json  # Array of N results
        └── <run_id_2>/
            └── ...

    Attributes:
        cache_dir: Path to the cache directory
        index: In-memory index of all cached runs
    """

    def __init__(self, save_dir: str = "./output"):
        """Initialize the cache manager.

        Args:
            save_dir: Directory to store cached runs
        """
        self.cache_dir = Path(save_dir + "/cache")
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

    def save_run(self, final_result: State, config: ArcoConfig, additional_metadata=None) -> None:
        """Save a complete agent run with all step results.

        Args:
            final_result: The final result returned by the agent
            config: The overall configuration loaded at the beginning
            additional_metadata: Additional metadata if needed
        """
        run_id = config.run_id
        run_dir = self.cache_dir / run_id
        run_dir.mkdir(exist_ok=True)

        # Save the last result for each agent
        for agent_type in AgentType:
            answer: Answer | None = final_result.get_last_answer(agent_type)
            if answer:
                agent_file = run_dir / f"{agent_type.value}.json"
                with open(agent_file, 'w') as f:
                    json.dump(answer.to_dict(), f, indent=2, default=str)

        # Save run metadata
        run_meta = {
            "config": asdict(config),
            "prompt_hash": _compute_prompt_hash(config.prompt),
            "timestamp": datetime.now().isoformat(),
            "final_result": final_result.to_dict(),
            "metadata": additional_metadata or {},
        }

        meta_file = run_dir / "metadata.json"
        with open(meta_file, 'w') as f:
            json.dump(run_meta, f, indent=2, default=str)

        # Update index (avoid duplicates)
        existing_ids = {r["run_id"] for r in self.index["runs"]}
        if run_id not in existing_ids:
            self.index["runs"].append({
                "run_id": run_id,
                "prompt": config.prompt,
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
        prompt_hash = _compute_prompt_hash(prompt)
        for run_meta in self.index["runs"]:
            if run_meta.get("prompt_hash") == prompt_hash:
                # Verify actual prompt matches (hash collision protection)
                if run_meta.get("prompt") == prompt:
                    return run_meta["run_id"]
        return None

    def load_agent_answer(
            self,
            run_id: str,
            agent_type: AgentType
    ) -> Answer | None:
        """Load the result for a specific agent from a cached run.

        Args:
            run_id: ID of the cached run
            agent_type: The type of agent to load results for

        Returns:
            The Answer
        """
        step_file = self.cache_dir / run_id / f"{agent_type.value}.json"
        if step_file.exists():
            with open(step_file, 'r') as f:
                raw = json.load(f)
            if "perplexity" in raw:
                from arco.core import EmpoweredAnswer
                return EmpoweredAnswer.from_dict(raw)
            from arco.core import Answer
            return Answer.from_dict(raw)
        return None

    def load_all_step_results(self, run_id: str) -> Dict[AgentType, Answer]:
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

        for agent_type in AgentType:
            agent_answer = self.load_agent_answer(run_id, agent_type)
            if agent_answer:
                results[agent_type] = agent_answer

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
                    metadata = json.load(f)
                    metadata['final_result'] = State.from_dict(metadata['final_result'])
                    return metadata
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
