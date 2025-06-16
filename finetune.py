# This code is based on the revised code from fastchat based on tatsu-lab/stanford_alpaca.


from dataclasses import dataclass, field
import json
import math
import logging
import os
from typing import Dict, Optional, List
import torch
from torch.utils.data import Dataset
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import transformers
from transformers import Trainer, GPTQConfig
from transformers.trainer_pt_utils import LabelSmoother
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate.utils import DistributedType
from transformers import Qwen2_5_VLForConditionalGeneration
from PIL import Image
from transformers import BitsAndBytesConfig

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen-7B")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = False
    fix_vit: bool = True


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["c_attn", "attn.c_proj", "w1", "w2"] ##["in_proj","out_proj","c_fc"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

#def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str, bias="none"):
#    """Collects the state dict and dump to disk."""
#    # check if zero3 mode enabled
#    if deepspeed.is_deepspeed_zero3_enabled():
#        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
#    else:
#        if trainer.args.use_lora:
#            state_dict = get_peft_state_maybe_zero_3(
#                trainer.model.named_parameters(), bias
#            )
#        else:
#            state_dict = trainer.model.state_dict()
#    if trainer.args.should_save and trainer.args.local_rank == 0:
#        trainer._save(output_dir, state_dict=state_dict)

def get_peft_state_maybe_zero_3(named_params, bias):
    """Manually extract LoRA adapter weights if get_peft_state_maybe_zero_3 is missing."""
    from collections import OrderedDict
    to_return = OrderedDict()
    for k, v in named_params:
        if "lora_" in k or (bias == "all" and "bias" in k):
            to_return[k] = v.detach().cpu()
    return to_return

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str, bias="none"):
    """Collects the state dict and dumps to disk, supporting LoRA and DeepSpeed Zero3."""

    # Safely check if DeepSpeed Zero3 is enabled
    is_zero3 = getattr(getattr(trainer, "deepspeed", None), "is_zero3", False)

    if is_zero3:
        # Handle DeepSpeed Zero3 checkpointing
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        # Handle LoRA saving
        if getattr(trainer.args, "use_lora", False):
            state_dict = get_peft_state_maybe_zero_3(
                trainer.model.named_parameters(), bias
            )
        else:
            state_dict = trainer.model.state_dict()

    # Save only from rank 0
    if trainer.args.should_save and (not hasattr(trainer.args, "local_rank") or trainer.args.local_rank == 0):
        trainer._save(output_dir, state_dict=state_dict)


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
    system_message: str = "You are a helpful assistant."
) -> Dict:
    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

    #im_start = tokenizer.im_start_id
    #im_end = tokenizer.im_end_id
    #nl_tokens = tokenizer('\n').input_ids

    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    nl_tokens = tokenizer('\n', add_special_tokens=False).input_ids
    
    _system = tokenizer('system').input_ids + nl_tokens
    _user = tokenizer('user').input_ids + nl_tokens
    _assistant = tokenizer('assistant').input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["user"]:
            source = source[1:]

        input_id, target = [], []
        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system
        target += [im_start] + [IGNORE_TOKEN_ID] * (len(system)-3) + [im_end] + nl_tokens
        assert len(input_id) == len(target)
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            _input_id = tokenizer(role).input_ids + nl_tokens + \
                tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
            input_id += _input_id
            if role == '<|im_start|>user':
                _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id)-3) + [im_end] + nl_tokens
            elif role == '<|im_start|>assistant':
                _target = [im_start] + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
                    _input_id[len(tokenizer(role).input_ids)+1:-2] + [im_end] + nl_tokens
            else:
                raise NotImplementedError
            target += _target
        assert len(input_id) == len(target)
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))
        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])
    input_ids = torch.tensor(input_ids, dtype=torch.int)
    targets = torch.tensor(targets, dtype=torch.int)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
        super(SupervisedDataset, self).__init__()
        
        rank0_print("Formatting inputs...")

        #Add-in
        if isinstance(raw_data, str):  # if a path, load it
            with open(raw_data, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        assert isinstance(raw_data, list), "Expected list of dicts in raw_data"

        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer, max_len)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer, self.max_len)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret

class MultimodalSupervisedDataset(Dataset):
    def __init__(self, raw_data, processor, max_len: int):
        super(MultimodalSupervisedDataset, self).__init__()
        
        self.processor = processor
        self.max_len = max_len

        if isinstance(raw_data, str):
            with open(raw_data, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        assert isinstance(raw_data, list)

        self.samples = raw_data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample["conversations"][0]["value"].split("<img>")[1].split("</img>")[0]
        image = Image.open(image_path).convert("RGB").resize((336, 336))
        image_grid = [1, 14, 14]  # Fixed for Qwen2.5-VL

        text_prompt = sample["conversations"][0]["value"].replace(
            f"<img>{image_path}</img>", "<image>"
        ) + "\n" + sample["conversations"][1]["value"]

        print("Image shape before processor:", image.size)  # Expect (336, 336)

        inputs = self.processor(
            text=[text_prompt],
            images=[image],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            do_resize=False,
        )

        print("Pixel shape:", inputs["pixel_values"].shape)  # Expect (1, 3, 336, 336)

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        pixel_values = inputs["pixel_values"].squeeze(0)

        labels = input_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": labels,
            "image_grid_thw": image_grid
        }

from transformers import AutoProcessor

def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, model_args, data_args, max_len,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    #dataset_cls = (
    #    LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    #)

    rank0_print("Loading data...")

    #train_json = json.load(open(data_args.data_path, "r"))
    if os.path.isdir(data_args.data_path):
        train_json = []
        for root, _, files in os.walk(data_args.data_path):
            for fname in files:
                if fname.endswith(".json"):
                    with open(os.path.join(root, fname), "r") as f:
                        try:
                            item = json.load(f)
                            if isinstance(item, list):
                                train_json.extend(item)
                            else:
                                train_json.append(item)
                        except Exception as e:
                            print(f"Skipping {fname} due to: {e}")
    else:
        train_json = json.load(open(data_args.data_path, "r"))

    if data_args.eval_data_path:
        #eval_json = json.load(open(data_args.eval_data_path, "r"))
        if os.path.isdir(data_args.eval_data_path):
            eval_json = []
            for root, _, files in os.walk(data_args.eval_data_path):
                for fname in files:
                    if fname.endswith(".json"):
                        with open(os.path.join(root, fname), "r") as f:
                            try:
                                item = json.load(f)
                                if isinstance(item, list):
                                    eval_json.extend(item)
                                else:
                                    eval_json.append(item)
                            except Exception as e:
                                print(f"Skipping {fname} due to: {e}")
        else:
            eval_json = json.load(open(data_args.eval_data_path, "r"))
    else:
        eval_json = None

    if "VL" in model_args.model_name_or_path:
        dataset_cls = MultimodalSupervisedDataset
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True
        )
        train_dataset = dataset_cls(train_json, processor=processor, max_len=max_len)
        eval_dataset = dataset_cls(eval_json, processor=processor, max_len=max_len) if eval_json else None
    else:
        dataset_cls = LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset        

        train_dataset = dataset_cls(train_json, tokenizer=tokenizer, max_len=max_len)
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer, max_len=max_len) if eval_json else None

    print(f"Using {dataset_cls}...")
    print(train_dataset)
    
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def train():
    global local_rank
    
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    print(model_args)
    print(data_args)
    print(training_args)
    print(lora_args)

    if getattr(training_args, 'deepspeed', None) and getattr(lora_args, 'q_lora', False):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED
    
    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    local_rank = training_args.local_rank

    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
        if len(training_args.fsdp) > 0 or training_args.deepspeed is not None:
            logging.warning(
                "FSDP or ZeRO3 are not incompatible with QLoRA."
            )


    # Set RoPE scaling factor
    #config = transformers.AutoConfig.from_pretrained(
    #    model_args.model_name_or_path,
    #    cache_dir=training_args.cache_dir,
    #    trust_remote_code=True,
    #)
    #config.use_cache = False

    # Load model and tokenizer
    #model = transformers.AutoModelForCausalLM.from_pretrained(
    #    model_args.model_name_or_path,
    #    config=config,
    #    cache_dir=training_args.cache_dir,
    #    device_map=device_map,
    #    trust_remote_code=True,
    #    quantization_config=GPTQConfig(
    #        bits=4, disable_exllama=True
    #    )
    #    if training_args.use_lora and lora_args.q_lora
    #    else None,
    #)
    
    #model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #        model_args.model_name_or_path,
    #        config=config,
    #        cache_dir=training_args.cache_dir,
    #        device_map="auto",
    #        trust_remote_code=True,
    #        torch_dtype=compute_dtype,
    #        quantization_config=None,
    #        low_cpu_mem_usage=True
    #)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model_kwargs = dict(
        #config=config,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
        torch_dtype=compute_dtype,
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
        force_download=True,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs
    )

    model.config.use_cache = False
    
    if not training_args.use_lora:
        if training_args.fix_vit and hasattr(model,'transformer') and hasattr(model.transformer,'visual'):
            model.transformer.visual.requires_grad_(False)
            if hasattr(model.transformer.visual,'attn_pool'):
                model.transformer.visual.attn_pool.requires_grad_(True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    #tokenizer.pad_token_id = tokenizer.eod_id
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if training_args.use_lora:
        if lora_args.q_lora or "chat" in model_args.model_name_or_path.lower():
            modules_to_save = None
        else:
            modules_to_save = ["wte", "lm_head"]

        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            #target_modules=lora_args.lora_target_modules,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "dense"],
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
            modules_to_save=modules_to_save  # This argument serves for adding new tokens.
        )
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

        model = get_peft_model(model, lora_config)

        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

        model.print_trainable_parameters()

    # Load data
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, model_args=model_args, data_args=data_args, max_len=training_args.model_max_length
    )

    # Start trainner
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module,
    )

    trainer.train()
    trainer.save_state()

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir, bias=lora_args.lora_bias)


if __name__ == "__main__":
    train()
