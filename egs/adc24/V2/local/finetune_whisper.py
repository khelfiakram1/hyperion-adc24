from huggingface_hub import notebook_login
from datasets import load_dataset, DatasetDict
from transformers import WhisperTokenizer
from transformers import WhisperFeatureExtractor
from transformers import WhisperProcessor
from datasets import Audio
from transformers import WhisperForConditionalGeneration
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from transformers.integrations import NeptuneCallback