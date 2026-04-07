
from flask import Flask, request, jsonify
import os
import tempfile
from Agent.data_agent import SalesDataAgent
from Agent.utils import get_evaluation_functions
from Agent.schema import DatabaseSchema, TableSchema, ColumnSchema


def _schema_from_dict(schema_dict: dict) -> DatabaseSchema:
    """Parse a schema definition from an API JSON payload into a DatabaseSchema.

    Expected format::

        {
          "compact_threshold": 5,
          "tables": [
            {
              "name": "sales",
              "description": "Daily sales transactions",
              "file_path": "/data/sales.parquet",
              "columns": [
                {"name": "date", "description": "Transaction date", "data_type": "DATE"},
                {"name": "revenue", "description": "Total revenue", "data_type": "FLOAT"}
              ]
            }
          ]
        }
    """
    tables = []
    for t in schema_dict.get("tables", []):
        columns = [
            ColumnSchema(
                name=c["name"],
                description=c.get("description", c["name"]),
                data_type=c.get("data_type", "VARCHAR"),
                example_values=c.get("example_values"),
                nullable=c.get("nullable", True),
            )
            for c in t.get("columns", [])
        ]
        tables.append(TableSchema(
            name=t["name"],
            description=t.get("description", t["name"]),
            file_path=t["file_path"],
            columns=columns,
        ))
    return DatabaseSchema(
        tables=tables,
        compact_threshold=schema_dict.get("compact_threshold", 5),
    )

app = Flask(__name__)

# Initialize a default agent (can be overridden per request)
agent = SalesDataAgent()

@app.route('/call-agent', methods=['POST'])
def call_agent():
    """
    API endpoint to call the SalesDataAgent.

    POST body (JSON):
    {
        "prompt": "Show me sales in Nov 2021",  # Required
        "model": "llama3.2:3b",  # Optional
        "data_path": "/path/to/data.parquet",  # Optional (single-table legacy)
        "schema": {                            # Optional (multi-table)
            "compact_threshold": 5,
            "tables": [
                {"name": "sales", "description": "...", "file_path": "/data/s.parquet",
                 "columns": [{"name": "date", "description": "...", "data_type": "DATE"}]}
            ]
        },
        "visualization_goal": "Create a bar chart",  # Optional
        "lookup_only": false,  # Optional
        "no_vis": false,  # Optional (run without visualization)
        "best_of_n": 1,  # Optional
        "temp": 0.1,  # Optional (base temperature)
        "temp_max": 0.5,  # Optional (max temperature for best-of-n)
        "save_dir": "/path/to/save",  # Optional
        "enable_codecarbon": false,  # Optional

        # Ground truth for evaluation
        "gt_csv": "/path/to/ground_truth.csv",  # Optional
        "gt_text": "/path/to/ground_truth.txt",  # Optional

        # Evaluation options
        "py_csv_eval": false,  # Use Python CSV evaluator
        "iou_type": "rows",  # "rows", "columns", or "table"

        "bleu_text_eval": false,  # Use BLEU for text evaluation
        "spice_text_eval": false,  # Use SPICE for text evaluation
        "llm_text_eval": false,  # Use LLM for text evaluation
        "bleu_nltk": false,  # Use NLTK BLEU instead of simple
        "spice_jar": "/path/to/spice.jar",  # SPICE jar path
        "spice_java_bin": "java",  # Java binary for SPICE

        # Phoenix tracing
        "enable_tracing": false,
        "phoenix_endpoint": "http://localhost:6006/v1/traces",
        "project_name": "evaluating-agent"
    }
    """
    payload = request.get_json(silent=True) or {}

    # Required parameter
    prompt = payload.get('prompt')
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400

    # Extract parameters with support for various naming conventions
    visualization_goal = payload.get("visualization_goal") or payload.get("goal")
    data_path = payload.get("data_path") or payload.get("data")
    model = payload.get("model")

    # Multi-table schema (optional)
    schema_dict = payload.get("schema")
    schema = None
    if schema_dict:
        try:
            schema = _schema_from_dict(schema_dict)
        except Exception as e:
            return jsonify({"error": f"Invalid schema definition: {str(e)}"}), 400

    # Agent behavior flags
    lookup_only = bool(payload.get("lookup_only") or payload.get("lookup-only") or payload.get("lookupOnly", False))
    no_vis = bool(payload.get("no_vis") or payload.get("no-vis") or payload.get("noVis", False))

    # Best-of-n parameters
    best_of_n = int(payload.get("best_of_n") or payload.get("best-of-n", 1))
    temp = payload.get("temp")
    if temp is not None:
        temp = float(temp)
    temp_max = payload.get("temp_max") or payload.get("temp-max")
    if temp_max is not None:
        temp_max = float(temp_max)

    # Save directory
    save_dir = payload.get("save_dir") or payload.get("save-dir")
    if not save_dir:
        save_dir = tempfile.mkdtemp(prefix="api_agent_runs_")

    # Ground truth paths
    gt_csv = payload.get("gt_csv") or payload.get("gt-csv")
    gt_text_path = payload.get("gt_text") or payload.get("gt-text")

    # CSV evaluation options
    py_csv_eval = bool(payload.get("py_csv_eval") or payload.get("py-csv-eval", False))
    iou_type = payload.get("iou_type", "rows")

    # Text evaluation options
    bleu_text_eval = bool(payload.get("bleu_text_eval") or payload.get("bleu-text-eval", False))
    spice_text_eval = bool(payload.get("spice_text_eval") or payload.get("spice-text-eval", False))
    llm_text_eval = bool(payload.get("llm_text_eval") or payload.get("llm-text-eval", False))
    bleu_nltk = bool(payload.get("bleu_nltk") or payload.get("bleu-nltk", False))
    spice_jar = payload.get("spice_jar") or payload.get("spice-jar")
    spice_java_bin = payload.get("spice_java_bin") or payload.get("spice-java-bin", "java")

    # Phoenix tracing options
    enable_tracing = bool(payload.get("enable_tracing") or payload.get("enable-tracing", False))
    phoenix_endpoint = payload.get("phoenix_endpoint") or payload.get("phoenix-endpoint", "http://localhost:6006/v1/traces")
    project_name = payload.get("project_name") or payload.get("project-name", "evaluating-agent")

    # CodeCarbon option
    enable_codecarbon = bool(payload.get("enable_codecarbon") or payload.get("enable-codecarbon", False))

    # Create agent (use global or create new one with custom params)
    req_agent = agent
    if model or data_path or schema or enable_tracing:
        try:
            req_agent = SalesDataAgent(
                model=model or "llama3.2:3b",
                temperature=temp or 0.1,
                data_path=data_path,
                schema=schema,
                enable_tracing=enable_tracing,
                phoenix_endpoint=phoenix_endpoint,
                project_name=project_name,
            )
        except Exception as e:
            return jsonify({"error": f"Failed to create agent: {str(e)}"}), 500

    # Get evaluation functions from utils
    try:
        csv_eval_fn, text_eval_fn = get_evaluation_functions(
            lookup_only=lookup_only,
            gt_csv_path=gt_csv,
            py_csv_eval=py_csv_eval,
            gt_text_path=gt_text_path,
            iou_type=iou_type,
            spice_text_eval=spice_text_eval,
            bleu_text_eval=bleu_text_eval,
            llm_text_eval=llm_text_eval,
            bleu_nltk=bleu_nltk,
            spice_jar=spice_jar,
            spice_java_bin=spice_java_bin,
        )
    except Exception as e:
        return jsonify({"error": f"Failed to create evaluation functions: {str(e)}"}), 400

    # Run agent
    try:
        result, score_variance = req_agent.run(
            prompt,
            visualization_goal=visualization_goal,
            lookup_only=lookup_only,
            no_vis=no_vis,
            best_of_n=best_of_n,
            temp=temp,
            temp_max=temp_max,
            csv_eval_fn=csv_eval_fn,
            text_eval_fn=text_eval_fn,
            save_dir=save_dir,
            enable_codecarbon=enable_codecarbon,
        )

        # Prepare response
        response = {
            "status": "success",
            "result": result,
            "save_dir": save_dir,
        }

        # Add score variance if best-of-n was used
        if best_of_n > 1:
            response["score_variance"] = score_variance

        # Add evaluation scores if available
        if "csv_score" in result:
            response["csv_score"] = result["csv_score"]
        if "text_score" in result:
            response["text_score"] = result["text_score"]

        # Extract and include key fields for convenience
        if "answer" in result:
            response["answer"] = result["answer"]
        if "data" in result:
            response["data_preview"] = result["data"][:500] + "..." if len(result["data"]) > 500 else result["data"]
        if "sql_query" in result:
            response["sql_query"] = result["sql_query"]
        if "error" in result:
            response["error"] = result["error"]

        return jsonify(response)

    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "type": type(e).__name__
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    try:
        # Check if agent can be initialized
        test_agent = SalesDataAgent()
        return jsonify({"status": "healthy", "model": test_agent.llm.model})
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 503

@app.route('/models', methods=['GET'])
def list_models():
    """List available Ollama models."""
    try:
        import requests
        ollama_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        response = requests.get(f"{ollama_url}/api/tags")
        models = response.json().get("models", [])
        return jsonify({"models": [m["name"] for m in models]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
