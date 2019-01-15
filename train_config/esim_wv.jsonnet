{
    "dataset_reader": {
        "type": "snli",
        "tokenizer": {
            "type": "word",
            "word_splitter": {
                "type": "just_spaces"
            }
        },
        "token_indexers": {
            "tokens": {
                "type": "single_id",
            }
        }
    },
  "train_data_path": "/home/6/18M31289/home/entasum/data/dbs31_1-25000_part1_01140725_snli_format_train.jsonl",
  "validation_data_path": "/home/6/18M31289/home/entasum/data/dbs31_1-25000_part1_01140725_snli_format_test.jsonl",
    "model": {
        "type": "esim",
        "dropout": 0.5,
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "pretrained_file": "/gs/hs0/tga-nlp-titech/matsumaru/entasum/data/entity_vector.model.txt",
                    "embedding_dim": 200,
                    "trainable": true
                }
            }
        },
        "encoder": {
            "type": "lstm",
            "input_size": 200,
            "hidden_size": 400,
            "num_layers": 1,
            "bidirectional": true
        },
        "similarity_function": {
            "type": "dot_product"
        },
        "projection_feedforward": {
            "input_dim": 3200,
            "hidden_dims": 400,
            "num_layers": 1,
            "activations": "relu"
        },
        "inference_encoder": {
            "type": "lstm",
            "input_size": 400,
            "hidden_size": 400,
            "num_layers": 1,
            "bidirectional": true
        },
        "output_feedforward": {
            "input_dim": 3200,
            "num_layers": 1,
            "hidden_dims": 400,
            "activations": "relu",
            "dropout": 0.5
        },
        "output_logit": {
            "input_dim": 400,
            "num_layers": 1,
            "hidden_dims": 2,
            "activations": "linear"
        },
        "initializer": [
            [".*linear_layers.*weight", {"type": "xavier_uniform"}],
            [".*linear_layers.*bias", {"type": "zero"}],
            [".*weight_ih.*", {"type": "xavier_uniform"}],
            [".*weight_hh.*", {"type": "orthogonal"}],
            [".*bias_ih.*", {"type": "zero"}],
            [".*bias_hh.*", {"type": "lstm_hidden_bias"}]
        ]
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [["premise", "num_tokens"],
                         ["hypothesis", "num_tokens"]],
        "batch_size": 32
    },
    "trainer": {
        "optimizer": {
            "type": "adam",
            "lr": 0.0004
        },
        "validation_metric": "+accuracy",
        "num_serialized_models_to_keep": 2,
        "num_epochs": 75,
        "grad_norm": 10.0,
        "patience": 10,
        "cuda_device": 0,
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "mode": "max",
            "patience": 0
        }
    }
}
