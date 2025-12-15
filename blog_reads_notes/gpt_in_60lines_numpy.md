<b> Link:</b> https://jaykmody.com/blog/gpt-from-scratch/ 

---

- Note on temperature: As sequences get longer, the model naturally becomes more confident in its predictions, so you can raise the temperature much higher for long prompts without going off topic. In contrast, using high temperatures on short prompts can lead to outputs being very unstable. From: https://docs.cohere.com/docs/temperature#how-to-pick-temperature-when-sampling
- Numerically stable softmax and cross entropy implementations: https://jaykmody.com/blog/stable-softmax/
- Different normalizations, their pros and cons: https://tungmphung.com/deep-learning-normalization-methods/
- Why not Batch norm in NLP: "statistics of NLP data across the batch dimension exhibit large fluctuations throughout training. This results in instability, if BN is naively implemented" (https://stats.stackexchange.com/questions/474440/why-do-transformers-use-layer-norm-instead-of-batch-norm)

    - In NLP tasks, the sentence length often varies -- thus, if using batchnorm, it would be uncertain what would be the appropriate normalization constant
    - straightforward to normalize each word independently of others in the same sentence

- Residual connections advantages: https://programmathically.com/understanding-the-exploding-and-vanishing-gradients-problem/