from effortless_config import Config as EffortlessConfig, setting
from transformers.configuration_utils import PretrainedConfig


class config(EffortlessConfig):
    groups = ["default", "small"]

    # huggingface settings
    n_patterns = 37771
    n_chords = 2223
    n_bar_numbers = 300
    n_beat_numbers = 16
    max_length = 128
    hidden_size = setting(768, small=128)
    num_hidden_layers = setting(12, small=4)
    num_attention_heads = setting(12, small=4)
    intermediate_size = setting(3072, small=512)
    hidden_act = "gelu"
    mask_prob = 0.15
    hidden_dropout_prob = 0.1
    attention_probs_dropout_prob = 0.1
    initializer_range = 0.02
    layer_norm_eps = 1e-12
    pad_token_id = 0
    mask_token_id = 1

    # training settings
    log_every = 10
    checkpoint_every = 100
    n_train = setting(5000, small=500)
    n_val = setting(1000, small=100)
    batch_size = 8
    learning_rate = 0.0001
    fp16_opt_level = "O1"
    continue_from_checkpoint = False
    gpu = True


class HuggingFaceConfig(PretrainedConfig):
    def __init__(
        self,
        **kwargs,
    ):
        if "pad_token_id" not in kwargs:
            kwargs["pad_token_id"] = config.pad_token_id

        super().__init__(**kwargs)

        self.n_patterns = config.n_patterns
        self.n_chords = config.n_chords
        self.n_bar_numbers = config.n_bar_numbers
        self.n_beat_numbers = config.n_beat_numbers
        self.max_length = config.max_length
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.num_attention_heads = config.num_attention_heads
        self.intermediate_size = config.intermediate_size
        self.hidden_act = config.hidden_act
        self.mask_prob = config.mask_prob
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob
        self.initializer_range = config.initializer_range
        self.layer_norm_eps = config.layer_norm_eps
        self.mask_token_id = config.mask_token_id
