# Databricks notebook source
# MAGIC %pip install -U 'vllm==0.6.1.post2'
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,get_default_ray_configs

NUM_WORKERS = int(spark.conf.get("spark.databricks.clusterUsageTags.clusterWorkers", "1"))
 
def get_gpus_per_worker(_):
  import torch
  return torch.cuda.device_count()
 
NUM_GPUS_PER_WORKER = sc.parallelize(range(4), 4).map(get_gpus_per_worker).collect()[0]
# num_gpus = 1

# COMMAND ----------

NUM_GPUS_PER_WORKER, NUM_WORKERS

# COMMAND ----------

from utils.uc_helpers import stage_registered_model
# this will roughly take 10-15 minutes
model_staged_path = stage_registered_model(
  catalog="system",#"databricks_dbrx_models",
  schema="ai",
  model_name="mixtral_8x7b_instruct_v0_1",
  version=3,
  local_base_path="/local_disk0/models",
  overwrite=False # if this is false it will use what ever is there existing
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Make fake data and table to simulate source table

# COMMAND ----------

import random
import pandas as pd
word_list = [
    "apple", "banana", "carrot", "dog", "elephant", 
    "frog", "guitar", "house", "ice cream", "jacket", 
    "kite", "lion", "moon", "nest", "orange", 
    "piano", "queen", "rabbit", "snake", "tree"
]

def generate_random_questions(word_list, n):
    questions = []
    for _ in range(n):
        question_type = random.choice(["Spell", "Define"])  # Choose between "Spell" or "Define"
        if question_type == "Spell":
            word = random.choice(word_list)
            questions.append(f"[INST]Answer to my quesiton in one sentence! Here is an example: \n Question: What day is today? Answer: Today is monday! \n How do you spell '{word}'? [/INST]")
        elif question_type == "Define":
            word = random.choice(word_list)
            questions.append(f"[INST]Answer to my quesiton in one sentence! Here is an example: \nQuestion: What day is today? Answer: Today is monday! \n What is the definition of '{word}'?[/INST] ")
    return questions
  
questions = generate_random_questions(word_list=word_list, n=100)

df = spark.createDataFrame(pd.DataFrame({"text": questions}))
display(df)

# COMMAND ----------

model_path = str(model_staged_path / "model")
tokenizer_path = str(model_staged_path / "components/tokenizer")
model_path, tokenizer_path

# COMMAND ----------

# import os
# # get your huggingface token and place it here. It is required for some tokenizer scripts to be run for dbrx from huggingface
# os.environ["HF_TOKEN"] = "<huggingface token>"

# COMMAND ----------

from vllm import LLM, SamplingParams
from typing import Iterator
from pyspark.sql.functions import pandas_udf

# this will roughly take ~4 minutes
model = LLM(model=model_path,
            tokenizer=tokenizer_path,
            tensor_parallel_size=NUM_GPUS_PER_WORKER)

# COMMAND ----------

params = SamplingParams( temperature = 0.1 , top_p = 0.6 , max_tokens=250)

def generate_in_batch(batch: pd.Series) -> pd.Series:
    responses = []
    outputs = model.generate(batch.tolist(), params)
    for output in outputs:
      responses.append(' '.join([o.text for o in output.outputs]))
    return pd.Series(responses)

def chunk_dataframe(df, chunk_size=4096):
    return [df[i:i + chunk_size] for i in range(0, df.shape[0], chunk_size)]

# Process each chunk
def process_chunk(chunk, column_name: str, new_column_name: str):
    chunk['column_to_process'] = generate_in_batch(chunk[column_name])
    return chunk

pdf = df.toPandas()
df_chunks = chunk_dataframe(pdf)
processed_chunks = [process_chunk(chunk, column_name="text", new_column_name="generated_text") for chunk in df_chunks]
processed_pdf = pd.concat(processed_chunks, ignore_index=True)
processed_df = spark.createDataFrame(processed_pdf)

# COMMAND ----------

display(processed_df)

# COMMAND ----------

import ray
ray.shutdown()

# COMMAND ----------


