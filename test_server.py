# Databricks notebook source
!pip install openai==1.40.3

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

driver_proxy_api = 'https://e2-demo-field-eng.cloud.databricks.com/driver-proxy-api/o/0/0911-205935-yyhtpuis/1234'
cluster_id = '0911-205935-yyhtpuis'
port = 1234

# COMMAND ----------

notebook_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# COMMAND ----------

import os
import openai
from openai import OpenAI

client = OpenAI(
    api_key="dapi-your-databricks-token",
    base_url="https://example.staging.cloud.databricks.com/serving-endpoints"
)

response = client.chat.completions.create(
    model="databricks-dbrx-instruct",
    messages=[
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "What is a mixture of experts model?",
      }
    ],
    max_tokens=256
)

# COMMAND ----------


from openai import OpenAI
client = OpenAI(
    base_url=driver_proxy_api,
    api_key=notebook_token
)

completion = client.chat.completions.create(
  model="deepseek-v2-lite",
      messages=[
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "What is a mixture of experts model?",
      }
    ],
    max_tokens=256
)

# COMMAND ----------


