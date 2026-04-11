# RPA Benchmarks Prompt Analysis

A comprehensive breakdown of prompts used in MemoryAgentBench, LongMemEval, and LoCoMo for testing long-term memory capabilities of LLMs.

---

## Overview

| Framework | Conference | Tested Model Style | Judge Method |
|-----------|------------|-------------------|--------------|
| **MemoryAgentBench** | ICLR 2026 | Memorize-then-Query | GPT-4o (LLM-as-Judge) |
| **LongMemEval** | ICLR 2025 | Direct context injection | GPT-4o / Llama-3.1-70B |
| **LoCoMo** | ACL 2024 | Context + Question | Rule-based (F1, EM, BERTScore) |

---

## 1. MemoryAgentBench

### 1.1 Tested Model Prompts

**Source:** `RPA_data/MemoryAgentBench/utils/templates.py`

#### System Message (All Tasks)
```
You are a helpful assistant that can read the context and memorize it for future retrieval.
```

#### Memorize Phase (Example - LongMemEval)
```
Dialogue between User and Assistant
<User> The following context is the conversation between the user and the assistant:
{context}
<Assistant> I have memorized the conversation and I will answer the question you ask.
```

#### Query Phase Templates

| Task | Agent Type | Prompt Template |
|------|------------|-----------------|
| **LongMemEval** | Long Context / RAG | `The history chats are between you and a user. Based on the relevant chat history, answer the question as concisely as you can, using a single phrase if possible.\n\n {question} \n\n Answer:` |
| **Ruler QA** | Long Context / RAG | `Answer the question based on the memorized documents. Only give me the answer and do not output any other words. \n\nQuestion: {question} \n\n Answer:` |
| **EventQA** | Long Context / RAG | `Based on the context you memorized, complete the task below:\n\n{question}\n\n The event that happens next is:` |
| **In-Context Learning** | Long Context / RAG | `Use the provided mapping from the context to numerical label to assign a numerical label to the context. Only output "label: {{label}}" and nothing else. \n\n{question} \n\n label:` |
| **RecSys Redial** | Long Context / RAG | `Pretend you are a movie recommender system. You need to recommend movies based on the dialogues you have memorized. Now I will give you a new conversation between a user and you (a recommender system). Based on the conversation, you reply me with 20 recommendations without extra sentences.` |
| **InfBench Sum** | Long Context / RAG | `You are given a book above and you are tasked to summarize it. \n\n{question} \n\n Now summarize the book.` |
| **Detective QA** | Long Context / RAG | `Based on the context you memorized, answer the question below. You are required to answer the question based on the strict output format.\n\n {question} \n\n` |
| **FactConsolidation** | Long Context / RAG | `Pretend you are a knowledge management system. Each fact in the knowledge pool is provided with a serial number at the beginning, and the newer fact has larger serial number. You need to solve the conflicts of facts in the knowledge pool by finding the newest fact with larger serial number. You need to answer a question based on this rule. You should give a very concise answer without saying other words for the question **only** from the knowledge pool you have memorized rather than the real facts in real world.` |

#### Agentic Memory Agent Query Prefix
```
Search Archival Memory and answer the question...
```

### 1.2 Judge Model Prompts

**Source:** `RPA_data/MemoryAgentBench/llm_based_eval/longmem_qa_evaluate.py`

**Judge Model:** GPT-4o (default)

#### Standard QA Evaluation
```
I will give you a question, a correct answer, and a response from a model.
Please answer yes if the response contains the correct answer. Otherwise, answer no.
If the response is equivalent to the correct answer or contains all the intermediate
steps to get the correct answer, you should also answer yes. If the response only
contains a subset of the information required by the answer, answer no.

Question: {question}
Correct Answer: {answer}
Model Response: {response}

Is the model response correct? Answer yes or no only.
```

#### Temporal Reasoning Evaluation
```
I will give you a question, a correct answer, and a response from a model.
Please answer yes if the response contains the correct answer. Otherwise, answer no.
If the response is equivalent to the correct answer or contains all the intermediate
steps to get the correct answer, you should also answer yes. If the response only
contains a subset of the information required by the answer, answer no.
In addition, do not penalize off-by-one errors for the number of days.
If the question asks for the number of days/weeks/months, etc., and the model makes
off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's
response is still correct.

Question: {question}
Correct Answer: {answer}
Model Response: {response}

Is the model response correct? Answer yes or no only.
```

#### Knowledge Update Evaluation
```
I will give you a question, a correct answer, and a response from a model.
Please answer yes if the response contains the correct answer. Otherwise, answer no.
If the response contains some previous information along with an updated answer,
the response should be considered as correct as long as the updated answer is the
required answer.

Question: {question}
Correct Answer: {answer}
Model Response: {response}

Is the model response correct? Answer yes or no only.
```

#### Preference Evaluation
```
I will give you a question, a rubric for desired personalized response, and a response
from a model. Please answer yes if the response satisfies the desired response.
Otherwise, answer no. The model does not need to reflect all the points in the rubric.
The response is correct as long as it recalls and utilizes the user's personal
information correctly.

Question: {question}
Rubric: {rubric}
Model Response: {response}

Is the model response correct? Answer yes or no only.
```

#### Abstention Evaluation (Unanswerable Questions)
```
I will give you an unanswerable question, an explanation, and a response from a model.
Please answer yes if the model correctly identifies the question as unanswerable.
The model could say that the information is incomplete, or some other information
is given but the asked information is not.

Question: {question}
Explanation: {explanation}
Model Response: {response}

Does the model correctly identify the question as unanswerable? Answer yes or no only.
```

### 1.3 Summarization Judge Prompts

**Source:** `RPA_data/MemoryAgentBench/llm_based_eval/summarization_evaluate.py`

#### Fluency Evaluation (Binary 0/1)
```
Please act as an impartial judge and evaluate the fluency of the provided text.
The text should be coherent, non-repetitive, fluent, and grammatically correct.

Below is your grading rubric:
- Score 0 (incoherent, repetitive, or incomplete): Incoherent sentences, repetitive
  sentences (even if not by exact words), incomplete answers, or gibberish.
- Score 1 (coherent, non-repetitive answer): Coherent, non-repetitive, fluent,
  grammatically correct answers. If the text is coherent, non-repetitive, and fluent,
  but the last sentence is truncated, it should still be given a score of 1.
```

#### Relevance Evaluation (0-3 Scale)
Compares generated summary against reference summary for key information coverage.

---

## 2. LongMemEval

### 2.1 Tested Model Prompts

**Source:** `RPA_data/LongMemEval/src/generation/run_generation.py`

#### Standard Prompt (No CoT)
```
I will give you several history chats between you and a user. Please answer the
question based on the relevant chat history.

History Chats:

### Session 1:
Session Date: {date_1}
Session Content:
{session_1_content}

### Session 2:
Session Date: {date_2}
Session Content:
{session_2_content}
...

Current Date: {question_date}
Question: {question}
Answer:
```

#### Chain-of-Thought Prompt
```
I will give you several history chats between you and a user. Please answer the
question based on the relevant chat history. Answer the question step by step:
first extract all the relevant information, and then reason over the information
to get the answer.

History Chats:
...

Current Date: {question_date}
Question: {question}
Answer (step by step):
```

#### With Index Expansion (Facts Merged)
```
I will give you several history chats between you and a user, as well as the relevant
user facts extracted from the chat history. Please answer the question based on the
relevant chat history and the user facts.

History Chats:

### Session 1:
Session Date: {date_1}
Session Content:
{"session_summary_facts": "{extracted_facts}", "original_session": {session_content}}
...
```

#### With Index Expansion (Facts Replace)
```
I will give you several facts extracted from history chats between you and a user.
Please answer the question based on the relevant facts.

History Chats:
{extracted_facts_only}
...
```

#### CoN (Extract-then-Reason) Mode
First extracts notes from each session:
```
I will give you a chat history between you and a user, as well as a question from
the user. Write reading notes to extract all the relevant user information relevant
to answering the answer. If no relevant information is found, just output "empty".

Chat History:
Session Date: {date}
Session Content: {session_content}

Question Date: {question_date}
Question: {question}
Extracted note (information relevant to answering the question):
```

### 2.2 Judge Model Prompts

**Source:** `RPA_data/LongMemEval/src/evaluation/evaluate_qa.py`

**Judge Models:** GPT-4o or Llama-3.1-70B-Instruct

Uses **identical judge prompts** to MemoryAgentBench (see Section 1.2).

---

## 3. LoCoMo

### 3.1 Tested Model Prompts

**Source:** `RPA_data/locomo/task_eval/gpt_utils.py`

#### Conversation Context Header
```
Below is a conversation between two people: {name1} and {name2}. The conversation
takes place over multiple days and the date of each conversation is written at
the beginning of the conversation.

DATE: {session_1_date}
CONVERSATION:
{speaker_1} said, "{text_1}"
{speaker_2} said, "{text_2}"
...

DATE: {session_2_date}
CONVERSATION:
...
```

#### Single Question Prompt
```
Based on the above context, write an answer in the form of a short phrase for
the following question. Answer with exact words from the context whenever possible.

Question: {question}
Short answer:
```

#### Batch Questions Prompt
```
Based on the above conversations, write short answers for each of the following
questions in a few words. Write the answers in the form of a json dictionary
where each entry contains the question number as "key" and the short answer as
"value". Use single-quote characters for named entities and double-quote characters
for enclosing json elements. Answer with exact words from the conversations whenever
possible.

0: {question_0}
1: {question_1}
2: {question_2}
...
```

#### Category 2 (Temporal) Prompt Addition
```
{question} Use DATE of CONVERSATION to answer with an approximate date.
```

#### Category 5 (Adversarial) Prompt
```
{question} Select the correct answer: (a) {option_a} (b) {option_b}.
```
Where one option is the answer and the other is "Not mentioned in the conversation".

### 3.2 Evaluation Metrics (No LLM-as-Judge)

**Source:** `RPA_data/locomo/task_eval/evaluation.py`

LoCoMo uses **rule-based metrics** without requiring a judge model:

| Metric | Description | Implementation |
|--------|-------------|----------------|
| **F1 Score** | Token-level F1 with Porter stemming | `f1_score(prediction, ground_truth)` |
| **Exact Match** | Normalized set equality | `set(prediction.split()) == set(ground_truth.split())` |
| **BERTScore** | Semantic similarity using BERT | `bert_score.score([pred], [gt], lang='en')` |
| **ROUGE-L** | Longest common subsequence | `rouge.get_scores(pred, gt)` |

#### Category-Specific Evaluation

| Category | Type | Evaluation Method |
|----------|------|-------------------|
| **1** | Multi-hop | F1 with comma-separated answer splitting |
| **2** | Single-hop, temporal | Simple F1 |
| **3** | Open-domain | Simple F1 |
| **4** | Additional single-hop | Simple F1 |
| **5** | Adversarial | Binary check: `"no information available" or "not mentioned" in output.lower()` |

#### Text Normalization
```python
def normalize_answer(s):
    s = s.replace(',', "")
    # Remove articles: a, an, the, and
    # Remove punctuation
    # Lowercase
    # Fix whitespace
    return normalized_text
```

---

## 4. OpenAI-Compatible API Support

| Framework | Native Support | Configuration Method |
|-----------|----------------|---------------------|
| **MemoryAgentBench** | Partial | Needs code modification to add `base_url` |
| **LongMemEval** | Yes | `--openai_base_url` CLI argument |
| **LoCoMo** | No | Uses deprecated `openai.ChatCompletion.create()` |

### Quick Start with DeepSeek/GLM

**LongMemEval** (easiest):
```bash
python src/generation/run_generation.py \
    --openai_base_url "https://api.deepseek.com/v1" \
    --openai_key "your-api-key" \
    --model_name "deepseek-chat" \
    --in_file data/longmemeval_s.json \
    --out_dir outputs/deepseek \
    --retriever_type orig-session \
    --topk_context 40 \
    --history_format nl \
    --useronly false \
    --cot false
```

---

## 5. Key Insights

### For Testing Online Models (DeepSeek, GLM, etc.)

1. **LoCoMo is most independent** - Uses rule-based metrics (F1, BERTScore) without requiring a judge model

2. **LongMemEval and MemoryAgentBench require a judge** - Need GPT-4o or Llama-3.1-70B for evaluation

3. **Judge model can be different from tested model** - You can test DeepSeek while using GPT-4o as judge

### Prompt Design Philosophy

| Framework | Approach | Key Feature |
|-----------|----------|-------------|
| MemoryAgentBench | Memorize-then-Query | Simulates agent with memory system |
| LongMemEval | Direct context | Tests retrieval + reasoning |
| LoCoMo | Conversation format | Tests realistic multi-session recall |

---

## References

- **MemoryAgentBench**: https://arxiv.org/abs/2507.05257 (ICLR 2026)
- **LongMemEval**: https://arxiv.org/pdf/2410.10813.pdf (ICLR 2025)
- **LoCoMo**: https://arxiv.org/abs/2402.17753 (ACL 2024)
