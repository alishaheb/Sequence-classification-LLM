
Project Overview<br>
This coursework project focuses on sequence classification (token labeling) for detecting abbreviations (AC) and long forms (LF) in biomedical scientific literature 
from PLOS journals. It uses the PLOD-CW dataset (50k labeled tokens from Hugging Face), sourced from the PLOD dataset, with optional extensions from PLOD-Filtered.
Labels follow the BIO schema: B-O (other), B-AC (abbreviation start), B-LF (long form start), I-LF (long form inside). Each token includes POS tags (optional).
The goal is to build a prototype classifier for information extraction tasks like e-commerce, dialogue systems, or machine translation quality estimation.
Maximum sequence length: 323 tokens. Evaluation emphasizes methodology over raw accuracy, using F1-score as the primary metric.
No language models for generating code/text; use free GPUs (e.g., Heron lab).
![image]
