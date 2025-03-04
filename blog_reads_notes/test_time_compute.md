Blog link - https://huggingface.co/blog/Kseniase/testtimecompute

Note: Do refer the blogs for more clear explanations and pictures. The notes is just like a tl;dr.

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
- Over/under allocation of resources - Dynamically adjusting the thinking based on difficulty of question is a hard problem.

---
Blog link - https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute

