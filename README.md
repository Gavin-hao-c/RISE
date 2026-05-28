# RISE
source code for our article "RISE: Repair-like Insertions for Stealthy Evasion of Vulnerability Detection Models"

## Experiments


### Create Environment


you need to install [srcml](https://www.srcml.org/) and [tree_sitter](https://tree-sitter.github.io/tree-sitter/) to run the code.

### Fine-tuning CodePTMs

Use `train.py` to train models.

Take an example:

```
cd CodeBERT/code
python train.py
```

### Fine-tuning CodeLlama

Install LLaMA-Factory:

```
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```
In the LLaMA-Factory, please modify the `dataset_info.json` in the `data` path according to your actual path, and change `do_sample` to `False` in `protocol.py` under `src/llamafactory/api` to avoid randomness in LLM output affecting the adversarial attacks.

Then, use the following command to run LoRA fine-tuning of the `CodeLlama-7B` model on Devign dataset.

```
llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path codellama/CodeLlama-7b-hf \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template default  \
    --flash_attn auto \
    --dataset_dir data \
    --dataset Devign \
    --cutoff_len 1024 \
    --learning_rate 5e-05 \
    --num_train_epochs 3.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0 \
    --optim adamw_torch \
    --packing False \
    --report_to none \
    --output_dir CodeLlama-Devign \
    --overwrite_output_dir True  \
    --fp16 True \
    --plot_loss True \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target q_proj,v_proj \
    --do_eval True  \
    --eval_steps 100 \
    --val_size 0.1 \
    --evaluation_strategy steps \
    --load_best_model_at_end
```

### Attacking CodePTMs

In our study, we employed two balanced datasets: Devign and DiverseVul. The dataset is in ./Adv-attack/dataset.


Take an example, when using RISE to attack Devign on CodeBERT, the script example is as follows.

```
CUDA_VISIBLE_DEVICES=0 python attack_ptm_att.py \
    --model_type roberta \
    --output_dir ../../CodeBERT/saved_models_Devign/ \
    --tokenizer_name microsoft/codebert-base \
    --model_name_or_path microsoft/codebert-base \
    --csv_store_path attack_results_ptm/attack_att_CodeBERT_Devign_test_target1.jsonl.csv \
    --eval_data_file ../../dataset/Devign/test_target1.jsonl \
    --block_size 512 \
    --eval_batch_size 64 \
    --seed 12345 2>&1 | tee attack_results_ptm/attack_att_CodeBERT_Devign_test_target1.jsonl.log
```
The result will be saved in `csv_store_path`
Run experiments of UniXcoder as well.

### Attacking CodeLlama 
```
CUDA_VISIBLE_DEVICES=0 python attack_llm_att.py \
    --base_model codellama/CodeLlama-7b-hf \
    --tuned_model /root/autodl-tmp/SLODA/LLaMA-Factory/CodeLlama-Devign \
    --eval_data_file ../../dataset/Devign/test_target1.json \
    --csv_store_path attack_results_llm/attack_att_CodeLlama_Devign_test_target1.csv \
    --seed 12345 2>&1 | tee attack_results_llm/attack_att_CodeLlama_Devign_test_target1.log
```

### Repair-like code Snippets 
The adversarial snippets are stored in `./features`. The code of getting-snippet is in `snippet_generate.py`
