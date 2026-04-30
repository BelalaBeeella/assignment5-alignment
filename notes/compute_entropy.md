假设某个位置的 logits 是 $z = [z_1, z_2, \cdots, z_V]$, 其中 $V$ 是词表大小.

softmax 概率是 $p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$

取对数得 $\log p_i = \log \frac{e^{z_i}}{\sum_j e^{z_j}}$

展开得 $\log p_i = z_i - \log \sum_j e^{z_j}$, 其中 $\log \sum_j e^{z_j}$ 就是 `logsumexp(logits)`.

令 $\mathrm{LSE}(z) = \log \sum_j e^{z_j}$, 那么 $\log p_i = z_i - \mathrm{LSE}(z)$.

熵的定义是 $H(p) = -\sum_i p_i \log p_i$

代入 $\log p_i = z_i - \mathrm{LSE}(z)$

$H(p) = -\sum_i p_i (z_i - \mathrm{LSE}(z))$

展开得 $H(p) = -\sum_i p_i z_i + \sum_i p_i \mathrm{LSE}(z)$

因为 $\mathrm{LSE}(z)$ 对所有 $i$ 都是同一个常数，所以可以提出来 $H(p) = -\sum_i p_i z_i + \mathrm{LSE}(z) \sum_i p_i$

又因为概率和为 1: $\sum_i p_i = 1$

所以 $H(p) = -\sum_i p_i z_i + \mathrm{LSE}(z)$

也就是 $H(p) = \mathrm{LSE}(z) - \sum_i p_i z_i$

对应代码就是 entropy = lse - exp_logits

其中

lse = logsumexp(logits)

exp_logits = sum(probs * logits)

