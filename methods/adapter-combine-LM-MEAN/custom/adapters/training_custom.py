from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AdapterArguments:
    """
    The subset of arguments related to adapter training.
    """

    train_adapter: bool = field(default=False, metadata={"help": "Train an adapter instead of the full model."})
    load_adapter: Optional[str] = field(
        default="", metadata={"help": "Pre-trained adapter module to be loaded from Hub."}
    )
    adapter_config: Optional[str] = field(
        default="pfeiffer", metadata={"help": "Adapter configuration. Either an identifier or a path to a file."}
    )
    adapter_non_linearity: Optional[str] = field(
        default=None, metadata={"help": "Override the non-linearity of the adapter configuration."}
    )
    adapter_reduction_factor: Optional[int] = field(
        default=None, metadata={"help": "Override the reduction factor of the adapter configuration."}
    )
    language: Optional[str] = field(default=None, metadata={"help": "The training language, e.g. 'en' for English."})


@dataclass
class MultiLingAdapterArguments(AdapterArguments):
    """
    Arguemnts related to adapter training, extended by arguments for multilingual setups.
    """

    load_lang_adapter: Optional[str] = field(
        default=None, metadata={"help": "Pre-trained language adapter module to be loaded from Hub."}
    )
    lang_adapter_config: Optional[str] = field(
        default=None, metadata={"help": "Language adapter configuration. Either an identifier or a path to a file."}
    )
    lang_adapter_non_linearity: Optional[str] = field(
        default=None, metadata={"help": "Override the non-linearity of the language adapter configuration."}
    )
    lang_adapter_reduction_factor: Optional[int] = field(
        default=None, metadata={"help": "Override the reduction factor of the language adapter configuration."}
    )

@dataclass
class FusionArguments:
    """
    Arguments for fusion.
    """
    train_fusion: bool = field(
        default=False, metadata={"help": "train fusion. default is false."}
    )
    test_fusion: bool = field(
        default=False, metadata={"help": "test fusion. default is false."}
    )
    fusion_path: Optional[str] = field(default=None, metadata={"help": "Should contain the fusion files for the test."})
    temperature: bool = field(
        default=False, metadata={"help": "train fusion. default is false."}
    )
    pretrained_adapter_names: Optional[str] = field(
        default=None, metadata={"help": "list of pretrained adapter names"}
    )
    pretrained_adapter_dir_path: Optional[str] = field(
        default=None, metadata={"help": "dir_path of pretrained adapters"}
    )
    fusion_attention_supervision: bool = field(
        default=False, metadata={"help": "test fusion. default is false."}
    )
    residual_before: bool = field(
        default=False, metadata={"help": "residual_before"}
    )