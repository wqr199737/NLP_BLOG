# Revolutionizing Domain-Specific Tasks with RAFT: A Deep Dive into Retrieval-Augmented Fine Tuning

# Blog outline

In this blog you will find an introduction to the domain-specific RAFT model. 
You will know that it is, what it is used for, how does it work, why is it needed and if it works well.

**What is RAFT?**

**The Need for RAFT**

**How RAFT Works - Retrieval-Augmented Setup**
**How RAFT Works - Focus Training**
**How RAFT Works - Chain-of-Thought Reasoning**

**Implementing RAFT in Real-Life Applications**

**Results and Impact**

**Conclusion**

**Future Directions**

**References**

# Motivation Figure

![image](https://github.com/wqr199737/NLP_BLOG/blob/main/figure%20files/figure1.png)

We know in some areas LLMs are overwhelmed by data and cannot have good performance. But with RAFT they can stay focused.

Domain-specific tasks pose unique challenges for LLMs, primarily due to the nuances and specialized knowledge required. Traditional training methods often fall short in preparing these models to handle such specificities effectively. RAFT fills this gap by enabling LLMs to leverage external, relevant documents during both the training and inference phases, akin to students using textbooks during an open-book exam. This method not only improves accuracy but also makes the models adaptable to evolving information, a critical need in sectors like medicine, law, and customer support.

Figure 1: How best to prepare for an Exam?(a) Fine-tuning based approaches implement "studying" by either directly
"memorizing" the input documents or answering practice QA without referencing the documents. (b) Alternatively, incontext retrieval methods fail to leverage the learning opportunity afforded by the fixed domain and are equivalent to
taking an open-book exam without studying. While these approaches leverage in-domain learning, they fail to prepare for
open-book tests. In contrast, our approach (c) RAFT leverages fine-tuning with question-answer pairs while referencing the
documents in a simulated imperfect retrieval setting — thereby effectively preparing for the open-book exam setting.

# system design 

How RAFT Works
The mechanism of RAFT can be likened to a targeted training regimen that sharpens an athlete's skills:

Retrieval-Augmented Setup: Initially, RAFT utilizes a set of both relevant (oracle) and irrelevant (distractor) documents. This setup simulates real-world scenarios where the model must discern useful from useless information.
Training with Context: During training, the model learns to focus on oracle documents that contain the answer to a query and ignore distractor documents. This is achieved by using a technique where the model is trained to recognize and utilize only the information pertinent to the query from the retrieved documents.
Chain-of-Thought Reasoning: An integral part of RAFT is encouraging the model to develop a chain-of-thought for each answer. This process involves the model articulating its reasoning pathway, thus ensuring a deeper understanding and better handling of complex queries.
Implementing RAFT in Real-Life Applications
RAFT has been successfully implemented across various domains, demonstrating significant improvements over traditional fine-tuning methods. For instance, in medical research, RAFT-enabled models have shown greater precision in answering complex queries from medical journals and databases. Similarly, in customer service, RAFT has empowered chatbots to provide more accurate responses based on the customer's historical data and the specifics of the inquiry.

![image](https://github.com/wqr199737/NLP_BLOG/blob/main/figure%20files/figure2.png)

Figure 2: Overview of our RAFT method. The top-left figure depicts our approach of adapting LLMs to reading solution
from a set of positive and negative documents in contrast to standard RAG setup where models are trained based on the
retriever outputs, which is a mixture of both memorization and reading. At test time, all methods follow the standard RAG
setting, provided with a top-k retrieved documents in the context.

Supervised Finetuning
Consider the supervised fine-tuning (SFT) setting for a Question-Answer dataset. The formulation consists of the Dataset (D) from which a set of Question (Q) and corresponding answer (A) pairs are derived or already available. In the classical SFT setting, the model is trained to improve its ability to answer the questions based on its knowledge obtained either during pre-training, or during the SFT training phase. The model so trained can also be used at test-time with the Retrieval Augmented Generation (RAG) setting, where additional documents can be introduced in the prompt to help the model answer the question. This can be represented as follows:
•	Train: Q→A
•	0-shot Inference: Q→A
•	RAG Inference: Q + D→A
RAFT
Retrieval Aware Fine-Tuning (RAFT), presents a novel recipe to prepare fine-tuning data to tailor the models for domain specific open-book settings, equivalent to in-domain RAG In RAFT, we prepare the training data such that each data point contains a question (Q), a set of documents (Dk), and a corresponding Chain-of-though style answer (A∗) generated from one of the document (D∗). We differentiate between two types of documents: ‘oracle’ documents (D∗) i.e. the documents from which the answer to the question can be deduced, and ‘distractor’ documents (Di) that do not contain answer-relevant information. As an implementation detail, the ‘oracle’ document doesn’t need to be a single document, but can be more than one document, as is the case in HotpotQA (Yang et al., 2018). Then, for P fraction of the questions (qi) in the dataset, we retain the oracle document (d∗i ) along with distractor documents (dk−1). For (1− P) fraction of the questions (qi) in the dataset, we include no oracle document and only include distractor documents (dk). We then fine-tune the language model using the standard supervised training (SFT) technique, training it to generate answers from the provided documents and questions. Fig. 2 illustrates the high-level design principal for RAFT .
We demonstrate that our approach trains the model to perform better RAG on the set of documents it is trained on i.e., in-domain. By removing the oracle documents in some instances, we are compelling the model to memorize answers instead of deriving them from the context. The training data for RAFT is as follows, and an example of training data can be seen in Fig. 3:
•	P % of data: Q + D∗ + D2 + ... + Dk →A∗
•	(1−P) % of data: Q + D1 + D2 + ... + Dk →A∗
Subsequently, for the test scenario, the model is provided with the Q and top-k documents retrieved by the RAG pipeline. Note that RAFT is independent of the retriever used.
A key factor in enhancing training quality is the generation of a reasoning process, such as Chain-of-Thought, to explain the provided answers.RAFT approach is similar: we demonstrate that creating a full reasoning chain and in addition, clearly citing sources enhances the model’s accuracy in answering questions. In Fig. 3, we illustrate this set-up. Generating the training data in this fashion, involves presenting the model with a question, context, and verified answers, and then requesting it to form a reasoning chain that appropriately references the original context.
For all the datasets in our experiments, we generate the answers using the technique described above. Note that the Gorilla APIBench dataset, already includes reasoning in the answers. We provide an example of the generation step in Fig. 3, the detailed reasoning answer includes a citation from the original context inside ##begin_quote## and ##end_quote## as well as the detailed explanation on how to reach the conclusion based on the citations. We demonstrate that adding detailed reasoning paragraphs helps boost the model’s performance in our experiment section.


# Results and Impact

We design our experiments to study how well RAFT performs compared to various baselines. We find that the RAFT7B model (a finetuned version of LlaMA-2) is better at reading and extracting information from in-domain documents,
than domain-specific finetuned model, and general-purpose model with RAG. As an ablation, we also demonstrate how important it is for the model to learn with Chain-of-Thought responses. In this section, we will first introduce all the datasets we used in the experiments, then all the baseline model/fine-tuning techniques that we benchmark against.

The impact of RAFT is quantifiable across multiple benchmarks. For instance, in the HotpotQA dataset, a challenging question-answering benchmark, RAFT-enhanced models have outperformed their non-RAFT counterparts by substantial margins. These improvements are reflected in the model's ability to handle complex multi-hop questions that require an understanding of multiple documents.

![image](https://github.com/wqr199737/NLP_BLOG/blob/main/figure%20files/figure3.png)
![image](https://github.com/wqr199737/NLP_BLOG/blob/main/figure%20files/figure4.png)
![image](https://github.com/wqr199737/NLP_BLOG/blob/main/figure%20files/figure5.png)
![image](https://github.com/wqr199737/NLP_BLOG/blob/main/figure%20files/figure6.png)


#  Conclusion
RAFT represents a significant leap forward in the application of large language models for domain-specific tasks. By enabling models to utilize external information effectively and focus on relevant data during their training, RAFT not only enhances the accuracy but also the adaptability of LLMs. As we continue to push the boundaries of what artificial intelligence can achieve, techniques like RAFT ensure that these technologies remain both powerful and practical for real-world applications.

References
Zhang, T., Patil, S. G., Jain, N., Shen, S., Zaharia, M., Stoica, I., & Gonzalez, J. E. (2024). RAFT: Adapting Language Model to Domain Specific RAG. University of California, Berkeley.
Wang, Q. (2024). Presentation on RAFT: Adapting Language Model to Domain Specific RAG.
