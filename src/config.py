model_configs = [
    {
        "name": "ADAM",
        "layers": [
            {"units": 16, "activation_name": "relu"},
            {"units": 2, "activation_name": "softmax"},
        ],
        "params": {
            "epochs": 50,
            "batch_size": 16,
            "learning_rate": 0.001,
            "algorithm": "adam"
        }
    },
    {
        "name": "SOFTGRAD",
        "layers": [
            {"units": 32, "activation_name": "relu"},
            {"units": 16, "activation_name": "relu"},
            {"units": 2, "activation_name": "softmax"},
        ],
        "params": {
            "epochs": 100,
            "batch_size": 32,
            "learning_rate": 0.001,
            "algorithm": "gradient_descent"
        }
    },
]