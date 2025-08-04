MODEL_CONFIGS = [
    {
        "name": "gradient_descent",
        "layers": [
            {"units": 16, "activation_name": "relu"},
            {"units": 8, "activation_name": "relu"},
            {"units": 2, "activation_name": "softmax"},
        ],
        "optimizer": "gradient_descent",
        "optimizer_params": {
            "learning_rate": 0.001,
        },
        "fit_params": {
            "epochs": 100,
            "batch_size": 32,
            "epsilon": 1e-6
        }
    },
    {
        "name": "momentum",
        "layers": [
            {"units": 16, "activation_name": "relu"},
            {"units": 8, "activation_name": "relu"},
            {"units": 2, "activation_name": "softmax"},
        ],
        "optimizer": "momentum",
        "optimizer_params": {
            "learning_rate": 0.01,
        },
        "fit_params": {
            "epochs": 100,
            "batch_size": 32,
            "epsilon": 1e-6
        }
    },
    {
        "name": "rmsprop",
        "layers": [
            {"units": 16, "activation_name": "relu"},
            {"units": 8, "activation_name": "relu"},
            {"units": 2, "activation_name": "softmax"},
        ],
        "optimizer": "rmsprop",
        "optimizer_params": {
            "learning_rate": 0.01,
        },
        "fit_params": {
            "epochs": 100,
            "batch_size": 32,
            "epsilon": 1e-6
        }
    },
    {
        "name": "adam",
        "layers": [
            {"units": 16, "activation_name": "relu"},
            {"units": 8, "activation_name": "relu"},
            {"units": 2, "activation_name": "softmax"},
        ],
        "optimizer": "adam",
        "optimizer_params": {
            "learning_rate": 0.01,
        },
        "fit_params": {
            "epochs": 100,
            "batch_size": 32,
            "epsilon": 1e-6
        }
    },
]