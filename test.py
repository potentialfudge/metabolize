from core.baybe_integration import generate_sobol_init

config = {
    "parameters": [
        {"name": "ligand", "type": "categorical", "values": ["A", "B", "C"]},
        {"name": "temp", "type": "continuous", "min": 20, "max": 100},
    ],
    "init_size": 5,
}

df = generate_sobol_init(config, seed=42)
print(df)