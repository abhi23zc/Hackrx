

import os
import openai

os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENAI_PROJECT_ID"] ="proj_9z99wQLU9UWxMKrO8R9d5WVg"
openai.api_key = os.environ["OPENAI_API_KEY"]

import nltk

nltk.download("stopwords")

import llama_index.core

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
    StorageContext,
)
from IPython.display import Markdown, display


# load documents
documents = SimpleDirectoryReader("./Sample Data/").load_data()

index = VectorStoreIndex.from_documents(documents)

# save index to disk
index.set_index_id("vector_index")
index.storage_context.persist("./storage")

# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir="storage")
# load index
index = load_index_from_storage(storage_context, index_id="vector_index")

# set Logging to DEBUG for more detailed outputs
query_engine = index.as_query_engine(response_mode="tree_summarize")

questions = [
    "If an insured person takes treatment for arthritis at home because no hospital beds are available, under what circumstances would these expenses NOT be covered, even if a doctor declares the treatment was medically required?",
    "A claim was lodged for expenses on a prosthetic device after a hip replacement surgery. The hospital bill also includes the cost of a walker and a lumbar belt post-discharge. Which items are payable?",
    "An insured's child (a dependent above 18 but under 26, unemployed and unmarried) requires dental surgery after an accident. What is the claim admissibility, considering both eligibility and dental exclusions, and what is the process for this specific scenario?",
    "If an insured undergoes Intra Operative Neuro Monitoring (IONM) during brain surgery, and also needs ICU care in a city over 1 million population, how are the respective expenses limited according to modern treatments, critical care definition, and policy schedule?",
    "A policyholder requests to add their newly-adopted child as a dependent. The child is 3 years old. What is the process and under what circumstances may the insurer refuse cover for the child, referencing eligibility and addition/deletion clauses?",
    "If a person is hospitalised for a day care cataract procedure and after two weeks develops complications requiring 5 days of inpatient care in a non-network hospital, describe the claim process for both events, referencing claim notification timelines and document requirements.",
    "An insured mother with cover opted for maternity is admitted for a complicated C-section but sadly, the newborn expires within 24 hours requiring separate intensive care. What is the claim eligibility for the newborn's treatment expenses, referencing definitions, exclusions, and newborn cover terms?"
]

for question in questions:
  response = query_engine.query(question)
  print(f"Question: {question}")
  display(Markdown(f"<b>{response}</b>"))

display(Markdown(f"<b>{response}</b>"))