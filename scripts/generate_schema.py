"""Generate JSON schema for ReductionConfig."""

import json
from pathlib import Path

from usansred.model import ReductionConfig

schema = ReductionConfig.model_json_schema()

output = Path(__file__).parent.parent / "src" / "usansred" / "io" / "usansred.json"
with open(output, "w", encoding="utf-8", newline="\n") as f:
    json.dump(schema, f, indent=2)
