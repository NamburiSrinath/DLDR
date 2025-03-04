<b>Note:</b> Do refer the blogs for more clear explanations and pictures. The notes is just like a tl;dr.

---
<b>Blog link</b> - https://huggingface.co/blog/Kseniase/testtimecompute

###  Key notes 

- Test time compute (TTC) is basically the processing power used when the model is being used.
- System-2/slow thinking which gives correct reasoning outputs when needed at cost of more computational cost.
- Chain of Thought prompting if given at input side. Same way, the o1 models use CoT thinking to reason for complex problems. 
- DeepSeek has explored 3 areas
  * DeepSeek R1 Zero - the model is rewarded for it's accuracy on outputting the structured output and detailed instructions on how it reasoned for the output. This uses GRPO (Group Relative Policy Optimization) 
    * This model eventually learned to spend more time on at test to solve complex problems and was able to reflect on it's reasoning and backtrack and correct itself - the famous "Aha moment"
    * But the outputs sometimes are hard to understand, so they used a sort of SFT to move from DeepSeek R1 Zero -> DeepSeek R1
  * DeepSeek R1 - This like like applying SFT on Zero model but the data is more reasoning focused. Just collect good reasoning examples and finetune the Zero on that.
  * Distillation - R1 is huge, so they transferred the reasoning capabilities to smaller models
    * They basically got the training data from R1 and finetuned the smaller models (7B sized) on that dataset.
- MLLM test time compute was helpful only using the text reasoning data and not "text + image" reasoning data. Part of this is because the "deep thinking" is needed and makes more sense for text type of data and not for image kind of data as that's more about image understanding.
  * Easy problems doesn't need more time to think. So, if we force them to think longer, it might result in -ve results.
- MLLMs are bad at providing good reasons for the outputs. So, CoMCTS (Collective Monte Carlo Tree Search) was proposed to get reasoning from multiple models
  * By using multiple models for reasoning, it tends to get good reasoning outputs for MLLMs.
  * By reflecting on both correct and incorrect responses, the model not only knows what's right but also knows what's wrong and why that's wrong and corrects itself in future!
  * Created a multimodal reasoning dataset (https://github.com/HJYao00/Mulberry?tab=readme-ov-file)
- Using CoT for generating images. PARM and PARM++ are interesting reads
  * Potential Assessment reward model evaluates the intermediate steps based on clarity (how good the picture is), if there's a potential for this image to make it to final output and if there are many, pick the best-of-N.
  * The ++ uses reasoning in the intermediate steps so the model can understand why something didn't work well for the feedback loop.
  * These outperform the ORM (Outcome reward model) which only looks at final outputs and PRM (Process Reward model) which evaluates the intermediate steps
- Test time for Searching (RAGs)
  * Traditional o1 (non-agentic) - Great at reasoning but has limited knowledge
  * o1 with agentic support (web based interactions) - Great at reasoning, can search for knowledge from other external resources
    * But the knowledge can be redundant, not useful or misleading. So, they added a "Reason-In-Documents" block which adds the retrieved context *only* if it was helpful.
- Underthinking issues - Model might give up on an idea too early missing a promising solution down the path!
- Over/under allocation of resources - Dynamically adjusting the thinking based on difficulty of question is still a pretty tricky problem.

---
<b>Blog link</b> - https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute

- Train-time compute - Increasing model size, dataset size and compute budgets during training.
- Scaling test time compute
  * Self refinement - Think about it's reasoning process and correct the thought process. 
    * This has a limitation that the model should have the capability to think and correct itself.
  * Search against a Verifier - Generate possible answers and let a verifier tell which works and which doesn't.
    * This can be best-of-N sampling (rejection sampling) or a reward model or a simple heuristic based verifier. 
    * Search can also be of different types like beam search, verifying the intermediate steps, Diverse verifier tree search (DVTS) etc;
    * Best of N - Generate N responses and score all of them using a reward model. Pick the best one
    * Beam search - Here not only we score the final response but also the intermediate reasoning steps using PRMs (Process Reward Models) thus improving the answer and reasoning capability
    * DVTS - Split the initial beam to multiple sub-trees and do greedy till the end for all the sub-trees using PRM. This improves diversity in responses compared to beam search.
- PRM gives the probability that the intermediate step can reach the final correct answer.
  * They are trained using "Process supervision" where the models receive feedback at every step not just at the output step. 
  * As they are trained in that way, they can give probability of correct path.
- Majority voting/Self consistency decoding - Generate N responses and pick the most frequently generated one.
- When solving math, multiple answers can be reduced to same form (2^5 or 2*2*2*2*2 are the same) thus need to be accounted properly. So, we need to reduce it to canonical form and get the frequency of the generated answer only after converting it; this way we can reduce pairwise comparisons among all generations.
- System prompt matters a lot more than one thinks from their experiments! We can restrict the final output to be `\boxed{answer}` if we want it to be (useful for math)
- Best-of-N is a simple extension where rather than simply looking at the most frequent generated answer will have a reward model to give score for these N outputs and based on the scores pick the answer
  * Vanilla - Pick the answer which has the maximum score given by RM. But an issue is the max score given by RM might not be the most frequent generated one which means the output won't be consistent across runs
  * Weighted best-of-N - Pick the one which has high score and more frequent. That's the metric we want to optimize for.
  * If we use PRM instead of ORM, we will get scores for every reasoning step. So that score needs to be reduced to one score instead and that can be done by either min, prod or last score in the reasoning step!
  * Reducing the scores using `last` will make PRM ~ ORM as it will only consider the last step while making decision. 
- Beam search with PRMs
  * At each step generate N responses (N will be the beam size). Make sure temperature is > 0 to introduce diversity.
  * Score each of the N responses using PRM at this step.
  * Out of N, keep only M responses and continue to generate/expand the tree in depth in those M responses (N/M).
  * Continue it until the end of sequence is reached or depth is completed
  * This way, Beam Search + PRM can keep the promising results early enough! Beam Search can bring the diversity in thought process and PRM can help with checking every step and maintaining promising directions.
- A naive computation of pass@k leads to variance as sampling multiple times leads to different answers. So, an unbiased estimator has `n` generated samples, `c` correct samples and `k` desired samples.  
- The term (n-c k)/(n k) represents the probability of selecting incorrect samples from the total, and subtracting from 1 gives the probability of having at least one correct sample among the top k.
- Looking at pass@k for various problem difficulties showed that beam search is better for difficult problems while majority or best-of-n is quite good enough for easy problems! i.e beam search is failing on easy problems and when there's high test-time compute!. This might be because of lack of diversity in generation of beam i.e if a single reasoning step is given high score in the beam, the answer is collapsing to that beam.
- To come across that, there's DVTS (Diverse Verifier Tree Search) to increase diversity at large N. 
  * This is an extension of beam search i.e at initial step, the N beams will be divided to N/M independent subtrees and for each subtree, select the step with highest PRM score.
  * From there, generate M new steps from each subtree and again select the step with highest PRM score. Repeat it until EOS or depth is reached!
- For smaller N (beams), beam search will be better, for larger N, DVTS will be better as it gives diversity! But for difficult questions, beam search might be overall better.
- We can see that picking the method depending on problem difficulty and compute budget, ~ `compute-optimal scaling strategy` gives best performance. An approximate way to get this strategy is to allocate the test-time compute according to how difficult the problem is i.e easy and medium problems (best-of-N) will be good, hard problems, beam search will be good! 
- The power of these methods depends on verifiers! Because only the ones that they give score to has a potential!
