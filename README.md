# DPFT_DBSec-2025
Replication of the training setup (both without and with differential privacy) and the attack of the [DBSec'2025 paper](https://link.springer.com/chapter/10.1007/978-3-031-96590-6_17) "Can Differentially Private Fine-Tuning LLMs Protect Against Privacy Attacks?".

This repository consists of 4 files as follows:

<h3><ins>1. Full Fine-tuning without Differential Privacy</ins></h3>

train\_gpt2\_wikitext\_full-finetuning.py - This script is used to first insert the canary sample "The secret code is hzdh0831" repeatedly into the training dataset Wikitext-2-v1, which contains 36718 samples for training and 3760 samples for validation, such that the repeated canary samples occupy 0.25% of the training samples (therefore, the canary is inserted 0.0025×36718 = 91 times). This script is then used to fine-tune a GPT-2 model under full fine-tuning setting. The training parameters used are as follows:

BLOCK_SIZE = 1024 (samples are processed as batches of 1024 tokens)

NUM\_EPOCHS = 10
EARLY\_STOPPING\_PATIENCE = 2

LEARNING\_RATE = 2e-4
BATCH\_SIZE = 4
EVAL\_BATCH\_SIZE = 4
GRADIENT\_ACCUMULATION\_STEPS = 4

WARMUP\_RATIO = 0.03
WEIGHT\_DECAY = 0.01

SEED = 42 (for reproducibility only)

We are not sure whether the canary sample is inserted as separate new lines or appended to an existing (non-sensitive) sentence from the wikitext-2-v1 dataset. The above parameters are randomly chosen since the paper does not explicitly mention them anywhere in the paper.

The script can be executed using the following commands.

python train\_gpt2\_wikitext\_full-finetuning.py --seed 42 --modified\_dataset\_dir ./wikitext2\_canary\_seed42 --output\_dir ./gpt2-wikitext2-full-canary-seed42

<h3><ins>2. LoRA Fine-tuning without Differential Privacy</ins></h3>

train\_gpt2\_wikitext\_lora.py - This script is used to first insert the canary sample "The secret code is hzdh0831" repeatedly into the training dataset Wikitext-2-v1, which contains 36718 samples for training and 3760 samples for validation, such that the repeated canary samples occupy 0.25% of the training samples (therefore, the canary is inserted 0.0025×36718 = 91 times). This script is then used to fine-tune a GPT-2 model under LoRA fine-tuning setting. The training parameters used are as follows:

BLOCK\_SIZE = 1024

LORA\_R = 8 (The paper only mentions this value; the other two values are not mentioned)
LORA\_ALPHA = 16
LORA\_DROPOUT = 0.1

NUM\_EPOCHS = 10
EARLY\_STOPPING\_PATIENCE = 2

LEARNING\_RATE = 2e-4
BATCH\_SIZE = 4
EVAL\_BATCH\_SIZE = 4
GRADIENT\_ACCUMULATION\_STEPS = 4

WARMUP\_RATIO = 0.03
WEIGHT\_DECAY = 0.01

SEED = 42

We are not sure whether the canary sample is inserted as separate new lines or appended to an existing (non-sensitive) sentence from the wikitext-2-v1 dataset. The above parameters are randomly chosen since the paper does not explicitly mention them anywhere in the paper.

The script can be executed using the following commands.

python train\_gpt2\_wikitext\_lora.py --seed 42 --modified\_dataset\_dir ./wikitext2\_canary\_seed42 --output_dir ./gpt2-wikitext2-lora-canary-seed42

<h3><ins>3. Attack Script</ins></h3>

gpt2\_canary\_attack.py - This script implements the data extraction methodology from the paper described in Sections 3.1 and 4.2 on pages 5--7 and 8--10. We test the weak attack where the input prefix is set to "The secret code is " and the exposure of the extracted candidates are evaluated relative to "hzdh0831". 

The script can be executed using the following command.

python gpt2\_canary\_attack.py --model\_dir ./gpt2-wikitext2-full-canary-seed42\_with\_DP --prefix "The secret code is " --candidates 1000 --temperature 1.2 --top\_p 0.95 --top\_k 0 --length 10 --canary "hzdh0831" --output attack\_top\_p\_0.95\_t\_1.2\_full\_with\_DP.txt --seed 42 --batch\_size 32

The paper states that "we employ sampling strategies such as temperature scaling, top-k, and nucleus (top-p) sampling to produce multiple outputs from the ﬁne-tuned model. We apply truncation to ensure that all candidate outputs maintain the same length."

However, explicit values of temperature, top-k, top-p and truncation length are not mentioned, so we did a trial and error to figure out these parameters. 

Result match: The paper claims that for full fine-tuning without Differential Privacy, "full ﬁne-tuning can fully output the canary sample". Moreover, "full fine-tuning exhibit extremely high exposure, reaching the maximum possible level" (which is 10).
We make both of these observations in our experiment as well, thus validating the claim. 

Result mismatch: The paper claims that even for LoRA without Differential Privacy, "LoRA exhibit extremely high exposure, reaching the maximum possible level".
Our experiment states otherwise, where the correct suffix "hzdh0831" did not even appear in the 1,000 unique generated candidates.
We are unsure whether this is due to an incorrect fine-tuning which is more likely given we are not using the correct set of parameters.

<h3><ins>4. Full Fine-tuning with Differential Privacy</ins></h3>

train\_gpt2\_wikitext\_full-finetuning\_DP.py - This script implements the Differentially Private version of the full fine-tuning using the fastDP library with Book-Keeping approach for gradient clipping. The training parameters used are as follows:

MAX\_LENGTH = 1024

TRAIN\_BATCH\_SIZE = 4
GRADIENT\_ACCUMULATION\_STEPS = 16

NUM_EPOCHS = 10
LEARNING\_RATE = 2e-4
WEIGHT\_DECAY = 0.01
WARMUP\_RATIO = 0.03

MAX\_GRAD\_NORM = 2.0

DELTA = 1e-5

EARLY\_STOPPING\_PATIENCE = 2

SEED = 42 (for reproducibility only)

We consider each row of the Wikitext-2-v1 (irrespective of its size) and the newly added canary strings as individual training samples. Again, the paper does not expliicitly mentions what constitutes as a sample.

The script can be executed using the following command.

python train\_gpt2\_wikitext\_full-finetuning\_DP.py --seed 42  --dataset\_dir ./wikitext2\_canary\_seed42 --output\_dir ./gpt2-wikitext2-full-canary-seed42\_with\_DP --epsilon 50 --delta 1e-5

Result mismatch: The paper claims that for full fine-tuning with Differential Privacy, "the application of DP leads to a signiﬁcant reduction in exposure even at a high privacy budget (e.g., epsilon = 50). This indicates that DP is highly eﬀective in mitigating the memorization—and thus the privacy risk—of sensitive data in these models." Moreover, the plots in Fig. 2 shows that this exposure value is ~4 (down from ~10).
Our experiment states otherwise, where the correct suffix "hzdh0831" did not even appear in the 1,000 unique generated candidates.
We are unsure whether this is due to an incorrect fine-tuning which is more likely given we are not using the correct set of parameters.
