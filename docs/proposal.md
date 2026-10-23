# Post-Training Portfolio Project: Selection Bias and Variance Sources in RL

Oct 21, 2026 · @Aarti Darji

## At a glance

Train small reasoning models with SFT then GRPO using several seeds, log every sampled answer, and run two analyses on the same logs. Part A asks how much picking the best RL checkpoint on a small benchmark overstates its accuracy, and whether a correlation-aware formula predicts it. Part B splits run-to-run variation in RL results into training-seed, problem-sampling and evaluation-sampling parts, and re-checks the pass@k finding of Yue et al. on your own models. Both are small, checkable extensions of known work, built to show post-training and evaluation skills rather than to be a paper. The math follows the lecture notes (ch. 2, §2.3–2.5) and was sanity-checked numerically.

*(Pipeline diagram: see the README.)*

Everything after the log runs without a GPU, so the analyses are cheap to redo and extend.

## How it maps to the lecture notes

Equation numbers refer to the generalization notes (ch. 2, §2.3–2.5). The project uses the notes' tools on a new object: the hypothesis set is the saved RL checkpoints, and the randomness in θ is the training seed.

| Notes | Idea | Where it appears here |
| --- | --- | --- |
| Eq. 2.99, 2.104–2.105 | Hoeffding: a bounded average concentrates around its mean for one fixed θ | One checkpoint's benchmark score vs. its true accuracy |
| Eq. 2.106–2.109 | Union bound over a finite Θ adds a √(log\|Θ\|/n) price for choosing | Θ = the K saved checkpoints; the price of picking the best is Part A's optimism |
| Eq. 2.110–2.115 | PAC-Bayes bound with a posterior over θ | Not used. Averaging over seeds is the practical stand-in for a distribution over θ |
| Eq. 2.116–2.124 | Bias–variance with θ random given the data | Part B asks where the randomness sits: training seed, problem sample or evaluation sampling |
| Eq. 2.125–2.129 | CLT and confidence intervals | Confidence intervals on scores, and a Gaussian model for the vector of K scores |
| Eq. 2.134–2.137 | Monte Carlo over M models drawn from the learner | S training seeds play the role of the M models |
| Eq. 2.138 | Bootstrap over training sets | Resampling problems (and seeds) for intervals; no retraining on bootstrap training sets |
| Eq. 2.139–2.142 | Learn/Tune with randomness ε; pick the hyperparameter with the best ε-averaged validation score | The checkpoint step is the hyperparameter; "argmax of the across-seed mean curve" is the Eq. 2.140 rule |
| Eq. 2.143–2.150 | Black-box tuning, SMBO | Not used. Picking the best of K is the simplest black-box search |
| §2.5.2 | Report on a separate test set | The oracle set: never select and report on the same problems |

## Literature check

I searched for prior work on each question before keeping it. Every item below is a replication of something known plus at most one small new measurement, which is the intended size.

| Question | What is already known | What this project does |
| --- | --- | --- |
| How noisy are LLM benchmark scores? | [Miller (2024)](https://arxiv.org/abs/2411.00640) gives error-bar recipes: clustered standard errors over problems, paired differences. [Madaan et al. (2024)](https://arxiv.org/abs/2406.10229) quantify seed and benchmark variance. | Reuses their recipes on your own logs (reproduction) |
| Are RL reasoning gains robust? | [Hochlehnert et al. (2025)](https://arxiv.org/abs/2504.07086) report that RL gains on small models are sensitive to seeds, decoding settings and benchmark size. [Colas et al. (2018)](https://arxiv.org/abs/1806.08295) do seed power analysis for deep RL. | Compares your variance estimates with theirs (reproduction) |
| How much does picking the best candidate inflate its score? | [Cawley & Talbot (2010)](https://www.jmlr.org/papers/v11/cawley10a.html) and [Dwork et al. (2015)](https://arxiv.org/abs/1411.2664) cover selection bias in general. [SIREN](https://arxiv.org/abs/2605.05973) corrects the winner's curse for prompt and program search. [UGCS](https://arxiv.org/abs/2511.09864) picks better RL checkpoints without a validation set. | **The small gap.** Measure the inflation for correlated RL checkpoints and test whether a Gaussian plug-in with the estimated covariance predicts it. My searches found no paper that does this exact measurement, but I cannot rule one out. |
| Does RL add capacity or only sharpen? | [Yue et al. (2025)](https://arxiv.org/abs/2504.13837): RL models beat their base model at small k and lose at large k. [Shao et al.](https://arxiv.org/abs/2506.10947) and [Wu et al.](https://arxiv.org/abs/2507.10532) show Qwen2.5 results can reflect contamination or spurious rewards. | Light replication of the pass@k curves on your models, plus a non-Qwen check (see Shared pipeline) |
| Where does run-to-run variation come from? | Seed effects on fine-tuned language models: [Dodge et al. (2020)](https://arxiv.org/abs/2002.06305). Variance components are classical random-effects ANOVA. | Applies the decomposition to GRPO scores, including the seed × problem interaction |

## The math, checked

Setup. There are K saved checkpoints and a pool of problems. For checkpoint k and problem x, let c̄\_k(x) ∈ \[0,1\] be the fraction of s sampled answers that are correct, and let A\_k be the checkpoint's true accuracy. A selection set S of n problems gives Â\_k = (1/n) Σ\_{x∈S} c̄\_k(x), which is unbiased for A\_k at fixed k (Eq. 2.104). Select k\* = argmax\_k Â\_k. The optimism is Â\_{k\*} − A\_{k\*}.

**1. Upper bound, no independence needed.** Each Â\_k − A\_k is an average of n independent variables in \[0,1\], so it is sub-Gaussian with variance proxy 1/(4n) (Hoeffding's lemma, the step behind Eq. 2.99 and 2.104). The expected maximum of K sub-Gaussians with proxy σ² is at most σ√(2 log K), whatever their dependence ([Rigollet & Hütter](https://arxiv.org/abs/2310.19244), ch. 1). This is the expectation form of the union bound in Eq. 2.106–2.109.

```latex
\mathbb{E}\big[\hat A_{k^*}-A_{k^*}\big]\;\le\;\mathbb{E}\max_{k}\big(\hat A_k-A_k\big)\;\le\;\sqrt{\frac{\log K}{2n}}
```

For K = 60 the bound is 0.26 at n = 30, 0.14 at n = 100 and 0.08 at n = 300. Sampling without replacement from a finite pool keeps the same bound (Hoeffding 1963, §6). It is an upper bound, and it uses the worst-case variance 1/4, so expect it to be loose.

**2. The simple independent-K formula.** The expression σ√(2 log K)/√n is only an upper bound for independent scores, and checkpoints from one run are far from independent. For independent scores the sharper value is σ·E\[max of K standard normals\]/√n.

**3. A prediction that accounts for correlation.** By the multivariate CLT (Eq. 2.125–2.129 applied to the vector of K scores), the noise is approximately Gaussian with a covariance you can estimate from the logs. Simulate it and read off the selection bias.

```latex
\hat A-A\approx Z\sim\mathcal N\!\big(0,\Sigma/n\big),\quad \Sigma_{kl}=\mathrm{Cov}_x\big(\bar c_k(x),\bar c_l(x)\big),\quad \text{optimism}\approx\mathbb{E}\,Z_{k^*},\;\; k^*=\arg\max_k (A_k+Z_k)
```

Using c̄ (the per-problem mean over s samples) puts both problem-sampling and answer-sampling noise into Σ automatically.

**4. Effective number of tests is descriptive only.** Meff estimators from genetics (Nyholt 2004; Li & Ji 2005) summarize the eigenvalues of the correlation matrix. There is no theorem linking them to the expected maximum, so report them but do not rely on them.

**5. The oracle design and a finite-pool identity.** The oracle set is disjoint from the selection set, so the choice of k\* is independent of the oracle score, and the oracle score is an unbiased estimate of the selected checkpoint's accuracy on the remaining pool problems (this is §2.5.2). With pool size N, selection size n and A\_pool the selected checkpoint's accuracy on the whole pool:

```latex
\text{sel}-\text{oracle}\;=\;\frac{N}{N-n}\,\big(\text{sel}-A_{\text{pool}}\big)
```

So the gap you measure against the oracle is the gap against the pool mean times N/(N−n), about 1.02 at n = 30 and 1.2 at n = 300 for a pool of 1,800. Report both.

**6. Variance components.** For seed s, problem i and replicate r, write the 0/1 outcome as a random-effects model (details in Part B). Its method-of-moments estimators can come out negative, so truncate at zero and say so. With S seeds the sample variance of per-seed accuracy has S − 1 degrees of freedom, so its interval is wide.

**7. A per-problem identity.** With p\_s the chance of solving a problem under seed s and p̄ the mean over seeds, the Bernoulli variance of a pooled outcome splits exactly into within-seed sampling and between-seed parts:

```latex
\bar p(1-\bar p)\;=\;\mathbb{E}_s\big[p_s(1-p_s)\big]+\mathrm{Var}_s(p_s)
```

**Numeric check.** I ran a synthetic simulation to test each formula (20,000 problems, 60 correlated checkpoints, 4 samples each; mean pairwise correlation of correctness 0.23). These numbers check the algebra and the estimators, not real model behavior.

| n | Measured optimism | Hoeffding bound | Independent-K formula | Gaussian plug-in |
| --- | --- | --- | --- | --- |
| 30 | 0.163 | 0.261 | 0.210 | 0.164 |
| 100 | 0.079 | 0.143 | 0.115 | 0.081 |
| 300 | 0.041 | 0.083 | 0.066 | 0.041 |

- The bound held at every n and was loose by roughly 2×. The independent-K formula overshot by 25–60%. The plug-in matched within 0.002.
- Nyholt and Li–Ji gave Meff of 57 and 47 out of 60 and overpredicted (0.208 and 0.201 at n = 30). The K that would match the measurement was 17, 11 and 8 at n = 30, 100 and 300, so no single effective K fits. This is why the plug-in is the main predictor and K\_eff is only a description.
- The pool-split identity held to about 1e-17 on a 1,319-problem example, and the per-problem identity held exactly.
- The ANOVA estimators were unbiased in simulation (S = 5, n = 300, R = 4). The seed variance had a relative standard deviation of about 74% and came out negative in 0.8% of replicates.
- Chi-square 95% intervals for a variance from S seeds: S = 3 gives 0.27× to 39.5× the estimate, S = 5 gives 0.36× to 8.3×, S = 8 gives 0.44× to 4.1×, S = 10 gives 0.47× to 3.3×. This is why the plan uses at least 5 seeds.

**Limits to state in the write-up.** Benchmark problems are not true i.i.d. draws, and they may overlap with pretraining data. The bound is on an expectation, not a high-probability statement. K counts every saved checkpoint, chosen before looking at results. All of this is for a fixed sampling temperature.

## Shared pipeline

Build one train-and-sample pipeline and save everything at the per-sample level; both parts are then analyses of the same logs.

| Piece | Choice | Why |
| --- | --- | --- |
| Model | Qwen2.5-0.5B or 1.5B, base and instruct (Qwen3-0.6B/1.7B also work) | Small enough for one GPU; widely used in public RLVR recipes, so you can sanity-check against them |
| Training data | GSM8K train, optionally MATH train | Verifiable answers, cheap rewards |
| Eval pool | GSM8K test + MATH-500, about 1,800 problems | Large enough to act as an "oracle" for Part A |
| SFT | Your own short SFT on GSM8K train solutions with TRL's `SFTTrainer` | Gives a clean base → SFT → RL chain you control |
| RL | GRPO with TRL's [`GRPOTrainer`](https://huggingface.co/docs/trl/main/grpo_trainer), exact-match answer reward | The standard RLVR setup used in post-training today |
| Sampling | vLLM, fixed temperature (e.g. 0.7) and max length | Fast batch generation for many samples per problem |

**Runs.** Train GRPO from the SFT model with 5 seeds (8 if compute allows) for a few hundred steps each. Seeds change only the random state (data order, sampling), not the data or hyperparameters. Save a checkpoint every 25 steps: about 20 per seed, so K is about 100 across 5 seeds.

**Logging.** For every (checkpoint, problem, sample), store the extracted answer, the sample index and whether it is correct. Score every checkpoint with 4 independent samples per problem at the fixed temperature (Part B needs the replicates), and take 64 samples per problem for the base, SFT and final RL models. Score all checkpoints on the same problems so comparisons are paired. Store raw text too, so you can debug the answer extractor. This table is the single input to both analyses.

**The grader is a hidden risk.** Answer extraction and normalization errors look exactly like model errors. Hand-check 100 graded samples before trusting any number.

**Model-choice caveat.** Qwen2.5 models can show gains under random or spurious rewards, which suggests contamination or pretraining effects ([Shao et al.](https://arxiv.org/abs/2506.10947), [Wu et al.](https://arxiv.org/abs/2507.10532)). Part A and Part B measure noise, so they are less exposed, but any claim about what RL teaches is model-specific. Repeat the final base/SFT/RL pass@k comparison on one non-Qwen small model (for example Llama-3.2-1B-Instruct) if time allows.

**Compute (rough estimate, to replace once timed).** GRPO for a 1.5B model is a few hours to a day per seed on one A100/H100. Evaluating about 100 checkpoints at 4 samples per problem, plus 64 samples per problem for three models in Part B, is roughly 30–55 GPU-hours with vLLM.

## Part A: Winner's curse in checkpoint selection

Question: if you pick the best of K RL checkpoints by a small benchmark score, how much does that score overstate the checkpoint's true accuracy, and can the notes' tools predict it?

The prediction comes from the math section: Eq. 2.106–2.109 gives the upper bound √(log K / 2n), and Eq. 2.125–2.129 gives a Gaussian plug-in that uses the covariance between checkpoints. Checkpoints from one run are strongly correlated, so the real inflation should sit below the bound.

### Measurements (all post hoc on the logs)

1. Before looking at results, split the eval pool once at random (seeded) into a fit half and a split half. Estimate A\_k and Σ from the fit half only.
2. On the split half, repeat 1,000 times: draw a selection set of n problems (n = 30, 100 or 300; n = 30 mimics AIME) and use the rest as the oracle set. Pick the checkpoint by its selection-set score, breaking ties at random.
3. Record optimism two ways: selection score minus oracle score, and selection score minus the pool mean (they differ by the factor N/(N−n) in the math section).
4. Vary K: one seed vs. all seeds, and every checkpoint vs. every fourth.
5. Overlay three predictions on the measured curve: the Hoeffding bound, the independent-K formula and the Gaussian plug-in with Σ from the fit half. Report Nyholt and Li–Ji Meff only as descriptions.
6. Compare selection rules by oracle accuracy: per-run argmax on the selection set, the last checkpoint, the mean of the last three, and the argmax of the across-seed mean curve (the Eq. 2.140 rule). Also report each rule's selection-set score, so the gap is visible.

### What the check suggests, and what you test

In the synthetic check the plug-in matched the measured optimism and the bound was about 2× loose. Real RL checkpoints may not behave like that simulation. Whether the plug-in holds out of sample, on logs from real training, is the question you answer.

### What you learn and can show

- How noisy small reasoning benchmarks really are, with numbers you produced yourself.
- Why "best checkpoint on AIME" claims need a held-out report set (§2.5.2).
- Which selection rule loses the least true accuracy for a given evaluation budget.

## Part B: Where does run-to-run variation in RL results come from?

Question: when you rerun GRPO with a different seed and re-evaluate, how much of the change in measured accuracy comes from the training seed, the choice of problems, the seed × problem interaction, or evaluation sampling?

This is the notes' bias–variance setting (Eq. 2.116–2.124) with the randomness made explicit: θ is random given the data through the training seed ε (Eq. 2.139–2.142), and the S seeds are the M models of Eq. 2.134–2.137. The earlier plan split pass@1 into bias and variance. I dropped it: in the binary form the notes' terms reduce to accuracy (¼E(y−ŷ)² = p(1−p) + (1−p)² = 1−p, checked), and the multi-answer version fit the notes loosely.

### The model

For seed s, problem i and sample r, write the 0/1 outcome as a random-effects model.

```latex
c_{s,i,r}=\mu+a_s+b_i+g_{si}+e_{sir},\qquad \mathrm{Var}(\text{one run's score})=\sigma_a^2+\frac{\sigma_b^2+\sigma_g^2}{n}+\frac{\sigma_e^2}{nR}
```

Here a\_s is the seed effect, b\_i the problem effect, g\_si the seed × problem interaction and e\_sir sampling noise, with R samples per problem and n problems. Only the seed term does not shrink with a bigger benchmark or more samples, so it is the floor on how well you can compare two runs. If the benchmark is fixed and you only care about it, drop the σ\_b²/n term.

The components come from balanced two-way ANOVA mean squares: E\[MS\_seed\] = σ\_e² + Rσ\_g² + nRσ\_a², E\[MS\_seed×problem\] = σ\_e² + Rσ\_g², E\[MS\_problem\] = σ\_e² + Rσ\_g² + SRσ\_b², E\[MS\_error\] = σ\_e². Solve for each σ². Because the outcome is Bernoulli, σ\_e² is the average of p(1−p), and the estimators stay unbiased in the simulation check.

### Measurements

1. Evaluate each seed's final checkpoint on all problems with R = 4 samples (reuse Part A's logs).
2. Fit the ANOVA and report the four components. Truncate negative estimates at zero and say how many there were. Give intervals by bootstrapping over problems and, for the seed term, chi-square intervals with S − 1 degrees of freedom.
3. Repeat at several training steps to see whether seed variance grows or shrinks during training.
4. Compute the benchmark size n\* = (σ\_b² + σ\_g²)/σ\_a² beyond which more problems barely help against the seed floor.
5. Per problem, split the Bernoulli variance with the identity in the math section, and list the problems where one seed solves it reliably and another does not.
6. Light replication of Yue et al.: pass@k curves from 64 samples per problem with the unbiased estimator 1 − C(n−c, k)/C(n, k), for the base, SFT and final RL models, with a band across the RL seeds. Check whether the RL curve crosses below the base at large k, as they report.

### What you learn and can show

- A clear statement of how much of an RL "improvement" you can trust, with numbers from your own runs.
- Hands-on pass@k, clustered standard errors and variance-components estimation.
- A result you can explain in an interview in two minutes, with your own figures.

## Plan and deliverables

Six weeks part-time gets both parts done, because Part B reuses Part A's runs. The non-Qwen check is a stretch goal.

1. **Week 1: eval harness.** vLLM sampling, answer extraction, grader and the per-sample log table. Hand-check 100 graded samples. Evaluate the base and instruct models as a sanity baseline.
2. **Week 2: SFT and first GRPO run.** Short SFT on GSM8K train, then one GRPO seed with checkpoints every 25 steps. Watch reward, response length and KL in Weights & Biases.
3. **Week 3: remaining seeds and checkpoint evals.** Four more GRPO seeds; evaluate all checkpoints on the eval pool.
4. **Week 4: Part A analysis.** Split simulation, optimism plots, plug-in vs. bound vs. independent-K, and the selection-rule comparison.
5. **Week 5: Part B.** Variance components, pass@k with 64 samples, and the stretch non-Qwen check.
6. **Week 6: write-up.** Clean the repo, write the README and a blog-style report.

### Deliverables

- A public repo with configs, training and eval scripts, and a one-command way to reproduce each figure.
- Released per-sample logs, so anyone can rerun the analyses without a GPU.
- A 6–8 page report or blog post: setup, four or five key figures, and plain-language takeaways, with the limits listed in the math section.

### How it reads on a CV

> Trained GRPO reasoning models (Qwen2.5-1.5B, 5 seeds) and built a vLLM evaluation harness. Measured that best-checkpoint selection on small benchmarks overstates accuracy by X points, tested a correlation-aware predictor of that inflation, and decomposed run-to-run variance into training-seed, problem and sampling components.

Replace X with your number. The bullet signals post-training, RL infrastructure and evaluation rigor at once, which matches what research-intern and ML-engineer roles in this area ask for.

## Sources

- Course lecture notes on generalization (ch. 2, §2.3–2.5), the source of all equation numbers.
- [Yue et al., Does RL Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?](https://arxiv.org/abs/2504.13837)
- [Hochlehnert et al., A Sober Look at Progress in Language Model Reasoning](https://arxiv.org/abs/2504.07086)
- [Miller, Adding Error Bars to Evals](https://arxiv.org/abs/2411.00640)
- [Madaan et al., Quantifying Variance in Evaluation Benchmarks](https://arxiv.org/abs/2406.10229)
- [Colas et al., How Many Random Seeds? Statistical Power Analysis in Deep RL Experiments](https://arxiv.org/abs/1806.08295)
- [Dodge et al., Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping](https://arxiv.org/abs/2002.06305)
- [Xu et al., Towards Reliable LLM Evaluation: Correcting the Winner's Curse in Adaptive Benchmarking (SIREN)](https://arxiv.org/abs/2605.05973)
- [Nguyen et al., Uncertainty-Guided Checkpoint Selection for Reinforcement Finetuning of LLMs](https://arxiv.org/abs/2511.09864)
- [Shao et al., Spurious Rewards: Rethinking Training Signals in RLVR](https://arxiv.org/abs/2506.10947)
- [Wu et al., Reasoning or Memorization? Unreliable Results of RL Due to Data Contamination](https://arxiv.org/abs/2507.10532)
- [Rigollet & Hütter, High-Dimensional Statistics](https://arxiv.org/abs/2310.19244)
- [Cawley & Talbot, On Over-fitting in Model Selection (JMLR 2010)](https://www.jmlr.org/papers/v11/cawley10a.html)
- [Dwork et al., Preserving Statistical Validity in Adaptive Data Analysis](https://arxiv.org/abs/1411.2664)
- Hoeffding (1963), Probability inequalities for sums of bounded random variables; Nyholt (2004) and Li & Ji (2005) on effective numbers of tests
- [TRL GRPOTrainer documentation](https://huggingface.co/docs/trl/main/grpo_trainer)
