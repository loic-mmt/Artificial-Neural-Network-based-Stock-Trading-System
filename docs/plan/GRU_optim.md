# Advanced GRU improvement candidates

Je conserve ici les hypothèses et le plan de recherche, y compris le point de
départ historique. Les passages « current baseline » ne remplacent pas les
décisions prises ensuite dans les benchmarks. Pour l'état au 3 octobre 2026,
je prends le [résumé général](../benchmark-summary.md) et les fiches de résultats.

Benchmark decisions and frozen protocol: [benchmark summary](../benchmark-summary.md).
Results of this GRU optimization series: [GRU benchmark journal](../benchmarks/gru-optim/README.md).

## Scope

This document covers changes to representation learning, sequence aggregation,
normalization, multimodal fusion, uncertainty, and cross-asset modeling. Generic
tuning such as learning rate, batch size, hidden width, dropout, or epoch count is
outside scope, except when required to compare an architectural change fairly.

The local evidence base uses the Markdown versions in [`docs/papers`](../papers/).
For RevTransLSTM-AR, whose paper is unavailable, the checked-in source code and
README are used only to verify implementation behavior.

## Current baseline and unused capacity

The default configuration in
[`gru.py`](../../src/trading_system/models/neural/gru.py) uses the final hidden state
to remain compatible with the historical baseline. The implementation now also
supports mean pooling, flattening, additive attention, and final-state plus
attention pooling over the full GRU output sequence.

The historical baseline was:

```python
_, hidden = self.recurrent(sequences)
```

For the last recurrent layer, forward and backward states are flattened when the
GRU is bidirectional, then passed directly to one linear classifier:

```text
input sequence
    -> GRU
    -> final hidden state only
    -> Linear
    -> Sell / Hold / Buy logits
```

Consequences when `temporal_pooling="last"` is selected:

- no explicit aggregation over all hidden states;
- no learned feature gating conditioned on market state;
- no normalization between recurrent representation and classifier;
- no nonlinear classification head;
- no explicit interaction between assets observed on the same date;
- no modality-specific branch or uncertainty-aware fusion;
- no calibrated confidence policy for choosing Hold.

This does not imply that the final hidden state contains no earlier information.
It means that the classifier has no direct access to individual hidden states and
cannot explicitly reweight dates inside the context window.

## Evidence map

| Source | Relevant evidence | Transfer limit |
|---|---|---|
| [RevTransLSTM-AR source](../papers/RevTransLSTM-AR/models/revtranslstm_ar/RevTransLSTM-AR.py) | Verifiable implementation combines per-instance RevIN, a Transformer memory, autoregressive LSTM decoding, cross-attention, and learned latent prediction feedback. The repository also supplies isolated ablation implementations and a simpler [RevIN-GRU classifier](../papers/RevTransLSTM-AR/models/revin-GRU.py). | Paper text and result tables are unavailable, so no performance conclusion is accepted. Main model is multi-step price regression, not three-class classification. Its training loop also evaluates test loss every epoch, so it is not a strict untouched-test reference protocol. |
| [GAS-Norm](../papers/GAS-Norm/GAS-Norm.md) | Online, score-driven estimates of time-varying mean and variance improve 21 of 25 reported forecasting configurations, and 13 of 16 configurations in the paper's broader model comparison. | Experiments concern probabilistic regression forecasting, not three-class movement classification. Output denormalization does not transfer directly to logits. |
| [MASTER](../papers/MASTER/MASTER.md) | Market-guided feature gating, intra-stock attention, inter-stock attention, then temporal attention. Reported gains support conditioning features on market state and modeling synchronized cross-sections. | Uses regression, date-synchronized stock batches, and transformer blocks. It is not a drop-in GRU head. |
| [MCI-GRU](../papers/MCI-GRU/MCI-GRU.md) | Combines a modified GRU cell, a stock graph, and latent-market cross-attention. Ablations support combining temporal and relational representations. | Full model changes several components together. Reset-gate formulation is difficult to isolate and appears underspecified for a single key/value per timestep. Reproduction risk is high. |
| [FusionLSTM-CNF](../papers/FusionLSTM-CNF/FusionLSTM-CNF.md) | Separate technical, sentiment, and correlation branches; uncertainty-weighted late fusion; MC dropout; confidence-filtered trading. Reported full-model ablation improves accuracy, F1, AUC, Sharpe, and ECE over fusion without confidence. | Binary next-day movement setup differs from our three-class triple-barrier task. Learned variance and reported thresholds require independent reproduction. |
| [Attention-Based Autoencoder](../papers/Attention-Based-Autoencoder/Attention-Based-Autoencoder.md) | Denoising before the downstream trader improves several profit and risk metrics; expanding-window evaluation and transaction costs are included. | AGRUA reshapes each feature vector to one timestep before its GRU. Its GRU is a gated feature transform, not evidence for temporal GRU denoising. Top Sharpe is shared by simpler denoisers. |
| [StockMixer](../papers/StockMixer/StockMixer.md) | Indicator mixing, causal multi-scale time mixing, then low-rank stock-to-market-to-stock mixing. Its ablation ranks time, stock, then indicator mixing by impact. `LSTM + Stock Mixing` is competitive with the paper's stronger relational baseline. | Predicts next-day returns with a regression plus ranking loss on a fixed, synchronized universe. Stock-axis weights depend on universe size and do not directly support missing or changing constituents. |
| [LSTM-GNN](../papers/LSTM-GNN/LSTM-GNN.md) | Two degree-normalized graph-convolution layers aggregate weighted neighbours, then relational and temporal embeddings are concatenated before prediction. The hybrid reports 10.6% lower MSE than its standalone LSTM. | Only ten selected stocks, normalized-price regression, no trading metrics, no GNN-only or edge-construction ablation, and no clear proof that graph estimation is fold-causal. The paper also acknowledges that its expanding-window setup lacks a separate validation set. |
| [CARA utility](../papers/CARA/CARA.md) | Additive risk-averse utility has an unbiased mini-batch gradient, remains interpretable for negative PnL, and exposes risk aversion through `gamma`. The paper also learns deviations from a baseline portfolio. | The paper outputs cross-sectional portfolio weights, assumes zero investment and unit gross leverage, and omits transaction costs. Our Sharpe implementation already uses the exact full chronological portfolio, so its mini-batch bias critique does not apply directly. |

## Prioritized experiments

Priority reflects expected information gained per implementation cost, not paper
headline performance.

| Priority | Experiment | Minimal comparison | Decision signal |
|---|---|---|---|
| 1 | Temporal pooling over all GRU outputs | `last`, `mean`, `attention`, `last + attention` | Better out-of-sample trading metrics across seeds without worse calibration or excessive turnover |
| 1 | Selective trading, then separate class calibration | raw financial position, validation-selected signal gate; cross-entropy temperature scaling separately | Better net PnL or Sharpe after costs at acceptable coverage, without mistaking signal size for confidence |
| 1 | Market-conditioned feature gating | no gate, static learned gate, market-conditioned gate | Stable gain beyond simply concatenating market features |
| 2 | Causal adaptive normalization | train-fitted global, rolling causal, RevIN-like, GAS-inspired | Robustness across regimes and context lengths without erasing volatility signal |
| 2 | Nonlinear normalized head | Linear, LayerNorm + Linear, LayerNorm + small MLP | Gain survives equal parameter-budget control |
| 2 | CARA utility objective | net PnL, exact full-path Sharpe, CARA over the same net returns | Better controllable return-risk trade-off after costs, especially when training PnL is negative |
| 3 | Modality-specific late fusion | early concatenation, separate branches, confidence-aware fusion | Each branch adds complementary value in ablation |
| 3 | Cross-sectional asset mixing | independent assets, sector pooling, StockMixer bottleneck, causal GCN, learned attention | Gain survives date-grouped evaluation, graph controls, and realistic compute cost |
| 4 | Denoising representation | raw features, simple denoising AE, attention AE | Better downstream metrics on untouched folds, not reconstruction loss alone |
| 4 | MCI-GRU reset-gate replacement | stock `nn.GRU`, custom-cell parameter control, custom cell | Improvement attributable to cell change across seeds |
| 5 | Autoregressive auxiliary decoder | classifier only, direct multi-horizon auxiliary head, autoregressive head, cross-attentive autoregressive head | Classification or trading gain justifies added sequential decoder cost and exposure bias |

## 1. Temporal aggregation over GRU outputs

### Hypothesis

The last state is a compressed summary. Direct access to every state may help when
the useful event occurs before the final date or when multiple regimes coexist in
one context window.

### Minimal architecture

```text
GRU outputs H = [h1, h2, ..., hT]
    -> temporal aggregator
    -> LayerNorm
    -> small MLP
    -> Sell / Hold / Buy logits
```

Additive temporal attention:

$$
e_t = v^\top \tanh(W h_t + b)
$$

$$
\alpha_t = \frac{\exp(e_t)}{\sum_{j=1}^{T}\exp(e_j)}
$$

$$
h_{\mathrm{att}} = \sum_{t=1}^{T}\alpha_t h_t
$$

Candidate fused representation:

$$
z = [h_{\mathrm{final}}; h_{\mathrm{att}}]
$$

For a bidirectional GRU, `h_final` means the concatenated final state from both
directions, matching the current implementation. It is not identical to the last
vector in the output sequence.

Attention only reweights dates inside the observed context. It cannot gather
information outside that window.

### Required ablation

```text
A0: final hidden state                 current baseline
A1: masked mean pooling                parameter-free control
A2: flatten all hidden states           source-derived parameter-heavy control
A3: additive attention                 learned temporal weighting
A4: final state + additive attention   preserves recency summary
A5: A4 + LayerNorm + small MLP         secondary head control
A6: A3 + LayerNorm + small MLP         preferred next head experiment
```

Implementation status: A0 through A4 are available through
`GRUConfig.temporal_pooling`. The head is configured independently through
`GRUConfig.head_type`: `linear` (default), `layernorm_linear`, `mlp`, or
`layernorm_mlp`. Thus A5 and A6 are both available without changing the pooling
implementation. The temporal-pooling comparison favored A3 over A4 at context
length 60, so test A6 before A5. `head_hidden_size` defaults to half the pooled
embedding width and `head_dropout` defaults to zero.

The shared GRU module now exposes `encode(sequences)` before the classification
head and extension hooks for input adaptation and the head. Future GRU variants
can reuse its temporal pooling and training interface without changing the
baseline GRU or its saved-state keys.

Keep encoder depth, hidden size, loss, labels, preprocessing, folds, and decision
policy fixed. Also compare a parameter-matched MLP control so a gain is not
mistaken for an attention gain.

The RevIN-GRU source uses `GELU`, dropout, then flattens all GRU outputs before
one linear classifier. Include this as `A2`, not as default. Its head has
`T * hidden_size * num_classes` weights, binds the model to one context length,
and can win through parameter count rather than better temporal aggregation.

### Diagnostics

- attention entropy and average attention by lag;
- attention stability across seeds;
- gradient norm at early versus late timesteps;
- performance by volatility regime;
- prediction coverage and turnover.

Attention maps are diagnostics, not causal explanations.

## 2. Calibration and selective trading

### Hypothesis

Positive F1 can coexist with negative PnL because classification quality does not
encode trade selectivity or costs. The current Sharpe objective decodes class
softmax values into a continuous position, `q = p(Buy) - p(Sell)`. This loss
optimizes the position path, not class-probability calibration. Different
probability vectors can produce the same `q`, so `max(p)` is not a validated
confidence measure for this model.

### First implementation: Sharpe signal-strength gate

Keep the trained GRU, financial loss, and probability-to-position decoder fixed.
On each fold's inner validation period, compare the raw position against gates
that map `|q| < tau` to zero. Candidate thresholds are quantiles of inner
validation `|q|`, plus `tau = 0` as an unchanged control. Select by the same
predeclared financial score as the outer CV, with a minimum signal-coverage
constraint; prefer less filtering on ties. Apply the frozen threshold to that
fold's untouched outer period. Do not call `|q|` calibrated confidence.

Implementation: `--position-gate-quantiles 0.2,0.4,0.6,0.8` on
`run_loss_comparison.py` with `--cv-folds` and `--losses sharpe`. The report
stores paired raw and gated outer metrics plus the inner threshold search;
the saved model artifact stores the selected gate. The default pipeline is
unchanged when this flag is absent. Leave the final holdout sealed.

### Separate experiment: class calibration

For a cross-entropy classifier, fit one temperature on validation logits only:

$$
p(y \mid x) = \operatorname{softmax}(\ell(x) / T), \qquad T > 0
$$

Then trade only when both conditions hold:

```text
predicted class != Hold
max calibrated probability >= tau
```

Choose `T` and any trading threshold inside each walk-forward fold. Never fit
them on final test data. Compare:

```text
C0: raw argmax
C1: temperature scaling + argmax
C2: temperature scaling + one global confidence threshold
C3: temperature scaling + separate Buy/Sell confidence thresholds
```

### Measurements

- Sharpe gate: signal coverage, turnover, exposure, net PnL, Sharpe and drawdown;
- class calibration: negative log-likelihood, Brier score, expected calibration
  error with bin definition recorded, and reliability diagram;
- confidence-policy coverage, trade count, turnover, and exposure;
- net PnL, Sharpe, Sortino, and maximum drawdown after costs;
- selective risk as confidence threshold changes.

MC dropout can follow only if deterministic calibration helps. It requires
multiple stochastic inference passes and measures model variability, not a
guaranteed probability of correctness.

### Link to FusionLSTM-CNF

The paper reports better results from confidence-aware fusion and from trading
only on low-uncertainty predictions. Its filter uses MC-dropout and learned
variance, not `max(p)` from a Sharpe-trained position model. The simple `|q|`
gate is a financial signal-strength control, not a reproduction of that
uncertainty mechanism. Reproducing its variance head, MC dropout, and fusion
loss in one step would prevent useful attribution.

## 3. Market-conditioned feature gating

### Hypothesis

Concatenation asks one recurrent encoder to discover both market regime and
stock-specific signal. A gate can make market state explicitly control which
stock features enter the GRU.

Let `x_{i,t}` be stock features and `m_t` market or sector context. A small market
encoder produces a gate:

$$
g_t = F \cdot \operatorname{softmax}(W_g m_t / \beta + b_g)
$$

$$
\widetilde{x}_{i,t} = g_t \odot x_{i,t}
$$

Multiplication by feature count `F` makes a uniform gate equal to one, matching
MASTER's scaling logic. Alternatives worth testing:

```text
G0: concatenate stock + market features
G1: one learned static gate
G2: gate from current causal market context
G3: residual gate, x_t * (1 + delta_t)
```

`G1` separates generic feature selection from regime conditioning. `G3` reduces
the risk of fully suppressing a feature early in training.

### Leakage constraints

- market state at prediction time may use information available through that
  timestamp only;
- sector aggregates must exclude unavailable future closes;
- if same-day execution occurs before close, same-day close-derived market
  features are invalid;
- gate temperature and all preprocessing statistics belong inside each fold.

## 4. Causal adaptive normalization

### Why it is architectural

For a non-stationary series, normalization defines what signal reaches the model.
It is part of the representation, not only an optimizer convenience.

### Comparison ladder

```text
N0: statistics fitted on training fold
N1: expanding causal mean and variance
N2: rolling causal mean and variance
N3: RevIN-like per-sequence normalization
N4: GAS-inspired online mean and variance filter
```

The first executable ladder is configured in
[`configs/benchmark/gru_normalization_ladder.json`](../../configs/benchmark/gru_normalization_ladder.json).
It keeps the existing train-fitted `SequenceStandardizer` for every candidate,
then applies an optional GRU input normalizer to the observed 60-step context.
`N1` and `N2` use prefix and trailing-window statistics respectively; both
restart at the beginning of each input context. `N4` is a fixed-rate Gaussian
score update, not a fitted reproduction of GAS-Norm. All three are causal with
respect to the decision, but are **window-local**, not persistent per-asset
filters across splits. A full-history stateful GAS experiment remains separate.
The first benchmark normalizes all selected features; feature-family ablations
are still needed before promoting a winner, because indicators and flags can
lose meaning under per-window normalization. No inverse transform is applied to
classification logits.

### Exact RevIN candidate from source

The audited RevIN computes one mean and population standard deviation per sample
and feature across the context axis:

$$
\mu_{b,f} = \frac{1}{T}\sum_{t=1}^{T}x_{b,t,f},
\qquad
s_{b,f} = \sqrt{\operatorname{Var}_t(x_{b,t,f}) + \epsilon}.
$$

It detaches both statistics from autograd, standardizes the sequence, then applies
one learned affine scale and bias per feature:

$$
\widehat{x}_{b,t,f} =
\gamma_f\frac{x_{b,t,f}-\mu_{b,f}}{s_{b,f}} + \beta_f.
$$

For regression, source code reverses the transform on predictions. For
classification, it normalizes inputs only and never transforms logits. That is
the correct transfer here.

Source benchmark stacks train-fitted `StandardScaler` and RevIN. Reproduce this
as a distinct condition instead of silently replacing existing preprocessing:

```text
R0: current train-fitted standardization
R1: RevIN only
R2: current standardization + RevIN
R3: R2 + detached mean and scale supplied as side features
```

Test RevIN by feature family. It was written for homogeneous OHLC levels. Applying
it indiscriminately to bounded RSI, binary flags, already relative returns, or
sparse sentiment may destroy meaning. Minimum comparison:

```text
prices and level features only
continuous features only
all features
```

Use `epsilon = 1e-5` as source-matching baseline. Assert finite output for constant
windows. Keep affine parameters enabled for faithful reproduction, then ablate
them only if RevIN itself helps.

GAS-Norm forecasts and restores output statistics in a regression setting. For
our classifier, first test only causal input filtering. Do not denormalize logits.
A faithful GAS adaptation would fit one filter per feature or feature family on
training history, freeze its learned parameters, then update filtered state
causally as observations arrive.

### Information that must remain available

Per-sequence normalization can remove absolute scale and volatility. Preserve or
reintroduce causal descriptors such as:

- rolling volatility;
- ATR normalized by price;
- relative volume;
- distance to moving averages;
- market and sector volatility;
- normalization mean and scale as optional model inputs.

Fit and update all states independently per asset unless a deliberately shared
cross-sectional normalizer is tested. Carry state across validation or test only
by replaying historical observations in chronological order, never by fitting on
the full split.

## 5. Nonlinear normalized classification head

The current linear head assumes classes are linearly separable in the final GRU
representation. Test a small residual head before larger encoder changes:

```text
representation
    -> LayerNorm
    -> Linear(d, d / 2)
    -> GELU
    -> Dropout
    -> Linear(d / 2, 3)
```

Required controls:

- `Linear`;
- `LayerNorm + Linear`;
- parameter-matched two-layer MLP without temporal attention;
- identical early stopping and loss.

For the next experiment, keep `temporal_pooling="attention"` and vary only
`head_type` among `linear`, `layernorm_linear`, `mlp`, and `layernorm_mlp`.
Repeat the head controls with `temporal_pooling="last"` to check whether a gain
comes from the head rather than an interaction with attention. The small MLP
uses `Linear(d, d / 2) -> GELU -> Dropout -> Linear(d / 2, 3)`; override the
intermediate width with `head_hidden_size` when matching parameter budgets.

This experiment determines whether apparent gains from larger architectures come
from better sequence modeling or merely a stronger output head.

## 6. Modality-specific branches and late fusion

Use separate encoders only for modalities with distinct sampling, noise, or
missingness patterns. Plausible branches:

```text
technical and price sequence by ticker -> GRU -> logits_temporal, representation_temporal
macro/micro/VIX sequence by session -> market Transformer -> market state (optional market logits)
stock/market node features + causal graph -> GNN -> logits_graph, representation_graph
publication-dated sentiment scores -> optional sentiment encoder -> logits_sent, representation_sent
```

The multimodal implementation routes the Transformer only to market-state
features, independently of the stock-sequence GRU. The legacy 3D benchmark
Transformer remains a stock-sequence comparator, not this new branch. The GNN
is a separate relational model, not a graph layer over GRU outputs. Sentiment
scores come from an external library through the historical point-in-time
source contract; this repository does not run FinBERT.

Start with late fusion of branch logits:

$$
\ell_{\mathrm{fused}} = \sum_m \alpha_m \ell_m,
\qquad
\sum_m \alpha_m = 1
$$

Experiment ladder:

```text
F0: each branch alone
F0b: early-concatenation GRU only where input modalities align
F1: separate branches, equal-weight logit average
F2: separate branches, learned static weights
F3: separate branches, confidence-conditioned weights
```

For `F3`, prefer calibrated branch probabilities or validation reliability as
confidence inputs before adding learned heteroscedastic variance. Record branch
weight distributions and behavior when one modality is absent. Normalize weights
over available branches only. Keep calibration, fusion weighting, and
confidence-based abstention as separate ablation axes.

Every branch must earn inclusion through single-branch and leave-one-branch-out
ablations. FinBERT outputs, DCC-GARCH spillovers, and specific feature counts from
FusionLSTM-CNF are examples from that paper, not defaults for this repository.

## 7. Cross-sectional asset interaction

This is the largest data-contract change. A graph or inter-stock attention layer
needs all relevant assets aligned at the same prediction timestamp:

```text
[batch, asset, time, feature]
```

The current per-sequence interface is insufficient if unrelated samples are mixed
in one batch. The shared GRU encoder serves GRU variants only. The planned GNN
branch uses separate stock/market node features, not GRU embeddings. Neither
branch currently supplies date-grouped batches, an asset mask, or causal graph
edges. The multimodal implementation plan is in [next_steps.md](next_steps.md). Required
work precedes model code:

1. build date-grouped batches and an asset mask;
2. define missing-asset behavior;
3. prevent survivorship leakage in the universe;
4. construct relations from historical data only;
5. preserve asset identity through prediction and backtest.

Incremental comparison:

```text
X0: independent per-asset GRU baseline
X1: independent GNN, identity graph (node-MLP control)
X2: independent GNN, sector graph
X3: independent GNN, train-fold correlation graph
X4: independent GNN, rolling causal correlation graph
X5: late fusion of GRU and GNN branch outputs
```

Compare `X3` and `X4` against identity and sector graphs before claiming that
learned relations add value. For learned attention, inspect complexity in asset
count and use masks for absent or newly listed securities.

### Optional StockMixer hybrid comparator

StockMixer offers a separate GRU-based hybrid comparator to pairwise attention
or graph message passing. It is not the independent GNN branch in `X1`-`X5`.
Given date-aligned GRU representations

$$
H \in \mathbb{R}^{N \times d},
$$

compress `N` assets into `m` latent market states, then expand them back:

$$
\widehat{H} = H + M_2\,\sigma(M_1\,\operatorname{LayerNorm}(H)),
$$

with

$$
M_1 \in \mathbb{R}^{m \times N},
\qquad
M_2 \in \mathbb{R}^{N \times m},
\qquad
m \ll N.
$$

Fuse each asset's own representation with its market-conditioned representation:

```text
GRU per asset -> H
H -> N-to-m market bottleneck -> m-to-N expansion -> H_market
concat(H, H_market) -> shared classifier
```

This directly tests the paper's useful claim without replacing the GRU temporal
encoder. The paper's ablation reports `LSTM + Stock Mixing` close to or above its
strong hypergraph baseline, making this hybrid more relevant here than a full
StockMixer replacement.

Required controls:

```text
S0: independent GRU
S1: GRU + mean market vector broadcast to every asset
S2: GRU + learned low-rank stock mixing
S3: indicator mixer + GRU + learned low-rank stock mixing
```

`S1` checks whether a learned asset-specific mixer beats simple market context.
`S3` tests StockMixer's claim that recurrent encoders under-model same-day feature
interactions. Add a residual MLP across features before the GRU, not an entire new
temporal encoder, to keep attribution clear.

Implementation risks:

- `M1` and `M2` bind parameters to asset count and asset ordering;
- missing constituents require masks that block both compression and expansion;
- a changing point-in-time universe needs fixed slots, shared asset-independent
  projections, or another design;
- the complete cross-section for a date must stay in one logical training sample;
- relation gains in the paper use regression and ranking losses, not our
  three-class objective;
- its reported experiments use only three seeds, so local variance testing remains
  mandatory.

StockMixer's full temporal alternative can be evaluated separately only after the
hybrid. Its time block mixes causal triangular MLPs across multi-scale pooled
windows. The reported ablation makes time mixing its most important component,
but replacing the GRU would answer a different question from improving the GRU.

### Independent GNN branch and late fusion

The LSTM-GNN paper uses separate inputs: price sequences for its temporal LSTM
and stock-specific node attributes plus a graph for its GNN. It fuses the two
resulting embeddings only after branch encoding. Our classification adaptation
keeps this separation: the GNN does not consume GRU embeddings.

For prediction date `t`, let `S_t` contain point-in-time stock/market node
features, `X_{t-L+1:t}` contain per-stock temporal sequences, and `A_t` be a
causal adjacency. The branches compute independently:

$$
H_t = \operatorname{GRU}(X_{t-L+1:t}), \qquad
G_t^{(0)} = \phi(S_t).
$$

Add self-loops to the graph and normalize it:

$$
\widetilde{A}_t = A_t + I,
\qquad
\widehat{A}_t = D_t^{-1/2}\widetilde{A}_tD_t^{-1/2}.
$$

One graph-convolution layer in the independent GNN becomes

$$
G_t^{(l+1)} =
\sigma\!\left(\widehat{A}_tG_t^{(l)}W^{(l)}\right).
$$

Each branch receives its own three-class head. Later experiments fuse branch
logits or representations, with the same branch availability mask:

```text
per-asset causal windows -> GRU -> H_t -> logits_gru
stock/market node features + causal graph A_t -> GNN -> G_t -> logits_gnn
(logits_gru, logits_gnn, availability mask) -> optional late fusion
```

Benchmark GRU-only and GNN-only before their fusion. One GCN layer aggregates
immediate neighbours. Two layers also reach neighbours-of-neighbours but
increase over-smoothing risk. Start with one.

### Graph construction

The paper combines absolute Pearson correlation above `0.7` with Apriori
association edges whose lift exceeds `1.7`. It does not provide enough detail to
reproduce how both edge types and weights are merged. First implementation should
therefore use Pearson returns only.

Two valid graph schedules answer different questions:

```text
static fold graph:
    estimate from training fold only
    freeze for validation and test

dynamic causal graph:
    at every date t, estimate from trailing returns ending at t
    update without target-period or future observations
```

Static graph tests persistent relations cheaply. Dynamic graph tests regime-varying
relations but costs more and introduces another lookback. Precompute every graph
causally and store its source interval in artifacts.

Do not feed signed correlations directly into standard degree normalization. With
negative weights, degree and normalization semantics become unstable. Compare:

```text
positive graph: A_ij = max(rho_ij, 0) above threshold
absolute graph: A_ij = abs(rho_ij) above threshold
signed graph: separate positive and negative adjacency channels
```

An absolute graph treats strong anti-correlation as similarity and loses sign.
The signed variant preserves meaning but requires two message-passing channels.
Tune correlation threshold on validation only. Always add self-loops and apply
asset masks before computing degrees.

Defer Apriori edges until event encoding, support, confidence, lift, edge direction,
and weight-combination rules are fully specified. Mining rules from the complete
dataset would leak validation and test structure.

### Required GCN ablation

```text
G0: GRU-only on the same eligible ticker/date rows
G1: GNN-only with identity graph (node-MLP control)
G2: GNN-only with fixed sector graph
G3: GNN-only with training-fold Pearson graph
G4: GNN-only with rolling causal Pearson graph
G5: G4 with separate positive and negative channels
```

`G1` controls for graph-branch nonlinear layers and parameters. Add a randomly
permuted graph as a sanity check, not as a model candidate. If the true graph
does not beat identity, sector, and permuted controls, relational structure has
not earned its complexity. Test GRU+GNN late fusion only after these standalone
comparisons.

Track:

- node degree distribution, isolated nodes, and graph density;
- edge turnover and sign stability through time;
- performance by node degree and sector;
- over-smoothing via pairwise embedding similarity across GCN layers;
- incremental memory, training time, and inference time;
- final classification, calibration, and trading metrics after costs.

### Evidence limits from LSTM-GNN

Treat the reported 10.6% MSE reduction as motivation, not expected effect size.
The paper evaluates normalized closing-price regression on ten hand-selected
stocks, reports no PnL or transaction costs, and does not isolate GNN depth,
Pearson edges, association edges, or fusion. It also states that its expanding
window lacks a separate validation set despite using early stopping and tuning.
Its graph-estimation interval is not specified clearly enough to establish absence
of leakage. Our implementation must not inherit those evaluation weaknesses.

## 8. Denoising representation

Treat denoising as an independently trained representation stage, not an assumed
improvement.

```text
raw causal window
    -> denoising encoder
    -> reconstructed or latent representation
    -> unchanged GRU classifier
```

Strict fold protocol:

1. fit denoiser on training fold only;
2. choose denoiser settings on validation only;
3. transform validation and test without refitting;
4. evaluate downstream trading metrics, not reconstruction loss alone.

Compare raw features, a simple denoising autoencoder, and attention-based
denoising. Add injected noise only to training inputs while keeping causal clean
targets explicitly defined.

Important evidence limit: AGRUA's reported GRU receives a sequence length of one.
It therefore supports gated nonlinear feature transformation inside a denoiser,
not long-range temporal denoising. Its results also do not show universal Sharpe
dominance over simpler denoisers.

Implementation: benchmark 13 compares raw post-open features, a frozen pointwise
DAE, and a frozen feature-attention DAE before the unchanged GRU. Discrete inputs
bypass reconstruction. See [denoising-post-open.md](denoising-post-open.md) for
the train-only protocol, limitations, and command.

## 9. MCI-GRU-style custom recurrent cell

MCI-GRU replaces the reset gate with an attention-derived quantity using previous
hidden state as query and current input as key and value. This cannot be reproduced
by configuring `nn.GRU`; it requires a custom recurrent cell and explicit unrolling.

Do this late because:

- paper formulation uses one current key/value per timestep, making the attention
  normalization axis ambiguous;
- dimensions in the written equations are not fully consistent;
- full-paper gains also include graph and latent-market modules;
- custom unrolling costs more and loses optimized `nn.GRU` kernels.

Before implementation, define exact tensor shapes and normalization axis. Require:

```text
M0: stock nn.GRU
M1: custom standard GRU cell, same equations and parameter budget
M2: custom attention-reset cell
```

`M1` is mandatory. Without it, framework and kernel differences are confounded
with the proposed gate.

Benchmark 14 implements native/manual PyTorch-equivalent controls and an
explicitly adapted channel-attention reset. The singleton-key ambiguity is not
treated as resolved by the paper. See
[attention-reset-post-open.md](attention-reset-post-open.md) for equations,
limits, and the post-open command.

## 10. CARA utility objective

### What transfers

CARA is relevant only to the repository's direct financial objectives. It is not
a replacement for cross-entropy in a standard three-class classifier unless the
class probabilities are first decoded into differentiable positions.

For per-period net portfolio return `R_t` and risk aversion `gamma`, use the
numerically centered form

$$
L_{\mathrm{CARA}} =
\frac{1}{|\Omega|}
\sum_{t \in \Omega}
\frac{\exp(-\gamma R_t)-1}{\gamma},
\qquad \gamma > 0,
$$

with continuous limit

$$
L_{\mathrm{CARA}} =
-\frac{1}{|\Omega|}\sum_{t \in \Omega}R_t,
\qquad \gamma = 0.
$$

Subtracting one does not change the optimizer for fixed positive `gamma`, but
improves interpretation and enables a stable `expm1` implementation. Use the same
`net[t]` path already defined for PnL and Sharpe, including execution delay,
turnover costs, asset boundaries, and terminal liquidation.

### Why test it despite exact Sharpe training

The paper proves bias when independently computed mini-batch Sharpe losses are
averaged. This repository does not do that. It computes one exact chronological
portfolio objective and one global update per epoch, so CARA's unbiased mini-batch
advantage is not a fix for a current bug.

Two differences remain useful:

- negative Sharpe can create an unintuitive preference for more volatility,
  while CARA utility remains monotone in return;
- `gamma` directly controls aversion to losses instead of optimizing only a
  return-to-volatility ratio.

### Required comparison

```text
L0: cross-entropy with unchanged discrete policy
L1: direct net-PnL objective, gamma = 0
L2: exact full-path regularized Sharpe
L3: CARA on net returns, several validation-selected gamma values
L4: fixed-scale mixture of PnL and Sharpe, validation-selected weight
```

`L0` to `L2` have already been trained in earlier benchmarks. In particular,
`L1` is exactly `L3` at `gamma = 0`; verify this by unit test rather than
training a duplicate. The 60/ATR PnL control in
[`01-temporal-pooling`](../../artifacts/comparisons/gru-optim/01-temporal-pooling/context-60-atr/report.json)
has a different CV data fingerprint and execution device from the current
Sharpe control. Reuse it as historical context, not as a strictly paired
control. The current Sharpe control is reproduced exactly in the head and
normalization benchmarks; do not retrain it merely to run CARA.

The first new benchmark trains only positive `gamma` CARA and interior-weight
combined candidates. `L1` to `L4` share the position decoder and net return path,
including costs. With a predeclared daily-return scale `s = 1e-4`, the combined
loss is

$$
L_{\mathrm{combined}} = (1-\lambda)L_{\mathrm{Sharpe}}
  + \lambda\left(-\frac{\overline{R}}{s}\right),
\qquad 0 < \lambda < 1.
$$

The fixed scale makes the weight interpretable; it must not be fitted on outer
folds or the final holdout. At `lambda = 0` the loss is exactly Sharpe, and at
`lambda = 1` it is PnL up to a positive constant. Both endpoints are unit-test
controls, not benchmark candidates. Select `gamma` and `lambda` with the frozen
outer-CV selection metric, then evaluate at most one frozen choice on final test.
Report the complete return-risk frontier, not only the best Sharpe point.

Numerical safeguards:

- define `gamma` in inverse units of per-period return;
- compute with `expm1` and float64 reduction where practical;
- monitor `gamma * R_t` and fail on non-finite values;
- do not clip losses silently because clipping changes tail-risk preferences;
- verify the `gamma -> 0` result against the existing net-PnL loss;
- gradient-check a tiny deterministic portfolio including turnover and liquidation.

### Baseline-relative positions

The paper's second proposal predicts deviations from a causal baseline portfolio,
then enforces zero net investment and unit gross leverage. This is a separate
experiment from CARA and requires synchronized cross-sectional outputs:

```text
learned adjustment + frozen causal baseline
    -> cross-sectional centering
    -> gross-leverage normalization
    -> portfolio weights
```

Do not combine this with the first CARA test. Otherwise loss and portfolio
construction effects become inseparable. A baseline-relative head also changes
the current long-only or long-short exposure contract and needs its own CLI model,
artifacts, backtest semantics, and borrow-cost assumptions.

## 11. RevTransLSTM-AR source-derived candidates

### Verified architecture

The available source implements this forecasting path:

```text
input window
    -> RevIN
    -> embedding + Transformer encoder memory
    -> LSTM decoder seeded by last observed decoder token
    -> cross-attention over full encoder memory
    -> one normalized prediction
    -> learned projection back to latent space
    -> residual feedback into next LSTM step
    -> repeat over forecast horizon
    -> inverse RevIN
```

Cross-attention is non-causal over encoder memory, but that memory contains only
the observed input window, so this is valid for forecasting. Future targets are
not teacher-forced into the loop. Each predicted step influences later steps in
latent space through `Linear(c_out, d_model)`.

Six source ablations remove AR feedback, cross-attention, learnable feedback,
RevIN, or combinations of them. These files verify intended controls, not their
empirical outcome. No result table is available locally.

### Relevance to current classifier

The current task emits one Sell, Hold, or Buy decision. It has no multi-step
decoder horizon. Therefore:

- RevIN transfers directly as input normalization;
- using all GRU states transfers as a temporal-head comparison;
- cross-attention from one final query to GRU outputs mostly duplicates temporal
  attention already proposed in section 1;
- autoregressive prediction feedback has no natural role for one class output;
- adding a Transformer before a recurrent decoder replaces the temporal backbone
  rather than improving the existing GRU cleanly;
- regression output denormalization must never be applied to class logits.

Do not port the full model into the classifier. Its complexity would combine
normalization, a second temporal encoder, cross-attention, recurrent decoding, and
feedback in one unidentifiable change.

### Optional multi-horizon auxiliary task

Autoregressive decoding becomes coherent only if the GRU learns an auxiliary
future path in addition to the class label. One possible design:

```text
causal input -> shared GRU -> hidden sequence H
                         |-> classification aggregator -> 3 logits
                         `-> recurrent return decoder -> r_1, ..., r_K
```

Use future returns or normalized price changes, not raw price levels. Let the
decoder query `H` through cross-attention and feed its previous predicted return
back through a learned projection. Train with

$$
L = L_{\mathrm{class}} + \lambda_{\mathrm{aux}}L_{\mathrm{path}}.
$$

The auxiliary horizon must lie entirely inside the label-information interval
already purged by the split logic. Otherwise it extends target overlap and
requires a longer purge or embargo.

Required ladder:

```text
Q0: GRU classifier only
Q1: classifier + direct multi-horizon projection
Q2: classifier + autoregressive GRU decoder
Q3: Q2 + cross-attention over encoder outputs
Q4: Q3 + learned prediction feedback
```

`Q1` tests whether auxiliary supervision alone helps. `Q2` tests recurrent
decoding. `Q3` tests memory access. `Q4` isolates learned feedback. Keep RevIN
fixed across `Q0` to `Q4` or run the full ladder without it first. Never infer an
AR-feedback benefit from a comparison that also introduces RevIN.

Closed-loop training can accumulate forecast error because later decoder inputs
depend on earlier predictions. Report path error by horizon step and verify that
any classification gain survives removal of the auxiliary head at inference.

### Source audit caveats

- available project README describes paper results, but paper text and result
  tables are unavailable, so those claims are not evidence here;
- long-forecast training computes and prints test loss after every epoch, although
  early stopping uses validation loss. This exposes test performance during model
  development and must not be copied;
- `StandardScaler` is fit on the training split only, which is appropriate;
- CUDA deterministic algorithms are not enabled, so multi-seed evaluation remains
  necessary;
- proposed RevTransLSTM-AR class is written for forecasting and is not a drop-in
  implementation for this repository's classification interface;
- source RevIN-GRU classification flattens every timestep, so its parameter count
  changes with `context_len` and needs parameter-matched controls.

## Context-length prerequisite

`context_len` is not an architectural novelty, but every sequence aggregation and
normalization result depends on its receptive field. Benchmark it before comparing
advanced variants while keeping triple-barrier horizon unchanged:

```text
context_len in {10, 20, 40, 60}
```

Use the same candidate set for baseline and proposed model. Avoid selecting one
length on final-test results. Report whether attention or adaptive normalization
changes sensitivity to context length.

## Common evaluation contract

All experiments must reuse:

- identical temporal folds, purge, embargo, label definition, and test period;
- train-only preprocessing and feature construction;
- identical transaction costs and execution timing;
- same seeds for paired comparisons;
- fixed final test used only after validation selection;
- equal search budget across variants;
- separate artifacts and CLI model names for structurally different models.

Primary measurements:

```text
macro F1
Buy F1 / Sell F1 / Hold F1
net PnL after costs
Sharpe and Sortino
maximum drawdown
turnover, exposure, trade count, coverage
calibration error and Brier score
mean and standard deviation across seeds and folds
train / validation gap
parameter count, training time, inference time
```

No architecture should be accepted from mean performance alone. Require paired
fold and seed deltas, confidence intervals or a paired nonparametric test, and a
predefined economic acceptance rule. Example:

```text
accept if median net-PnL delta > 0
and Sharpe does not degrade materially
and maximum drawdown stays within the predefined limit
and result is not driven by one fold or one seed
```

## Recommended execution order

```text
0. freeze baseline, folds, costs, seeds, and context-length comparison
1. temporal pooling ablation
2. LayerNorm and MLP head controls
3. Sharpe signal-strength gate; class temperature scaling as a separate experiment
4. causal normalization ladder
5. CARA utility against net PnL and exact full-path Sharpe
6. date-aligned multimodal contract and independent GNN input features
7. GNN-only graph controls: identity, sector, static, rolling causal
8. market-guided gating, disabled/static/conditioned, independent of fusion
9. GRU-only, GNN-only, then equal/static/confidence-aware late fusion
10. independent market Transformer on macro/micro/VIX inputs, with context-only and predictive ablations
11. post-open execution pilot and opening-gap control
12. optional StockMixer GRU hybrid and learned inter-asset attention controls
13. denoising stage
14. custom MCI-GRU cell
15. multi-horizon auxiliary decoder, only if simpler changes plateau
```

This order favors cheap, attributable tests before changes requiring new data
contracts or custom recurrent kernels. The active multimodal plan and its
independent switches are tracked in [next_steps.md](next_steps.md).

## Idea not validated from the local source corpus

The original draft also cited GCFin. Its paper is unavailable on free basis, hence its detailed claims were not used to rank experiments.

- [GCFin](https://doi.org/10.1016/j.engappai.2025.110834).

It will be added to the ranked plan only after its methodology, evaluation protocol, and
transfer limits have been checked from an accessible local source.
