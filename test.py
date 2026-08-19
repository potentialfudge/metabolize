import pandas as pd
from core.baybe_integration import recommend_next_batch

config = {
    "parameters": [
        {"name": "ligand", "type": "categorical", "values": ["A", "B", "C"]},
        {"name": "temp", "type": "continuous", "min": 20, "max": 100},
    ],
    "target_name": "yield",
}

history = pd.DataFrame({
    "ligand": ["A", "B", "C", "A", "B"],
    "temp": [30, 50, 70, 40, 60],
    "yield": [0.3, 0.5, 0.7, 0.35, 0.55],
    "batch": [0, 0, 0, 0, 0],
})

out = recommend_next_batch(config, history, batch_size=3, total_batches=10)
print(out["recommendations"])
print("xi:", out["xi"])
print(out["blend"]["reason"])