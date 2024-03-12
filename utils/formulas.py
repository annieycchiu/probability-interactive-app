# Binomial distribution
binomial_notation = r'$X \sim \mathrm{Bin}(n, p)$'
binomial_pmf = r'$f(k) = \Pr(X=k) = \binom{n}{k} p^k (1-p)^{n-k} \quad \text{for } k = 0, 1, 2, ..., n$'
binomial_cdf = r'$F(k) = \Pr(X \leq k) = \sum_{i=0}^{k} \binom{n}{i} p^{i} (1-p)^{n-i}$'
binomial_exp = r'$E[X] = np$'
binomial_var = r'$Var(X) = np(1-p)$'

# Poisson distribution
poisson_notation = r'$X \sim \mathrm{Poi}(\lambda)$'
poisson_pmf = r'$f(k) = \Pr(X=k) = \frac{\lambda^k e^{-\lambda}}{k!} \quad \text{for } k = 0, 1, 2, ..., n$'
poisson_cdf = r'$F(k) = \Pr(X=k) \sum_{i=0}^{k} \frac{\lambda^i e^{-\lambda}}{i!}$'
poisson_exp = r'$E[X] = \lambda$'
poisson_var = r'$Var(X) = \lambda$'

# Uniform distribution
uniform_notation = r'$X \sim U(\alpha, \beta)$'

uniform_pdf = r'''
$f(x) = \begin{cases}
\frac{1}{b-a} & \text{for } x \in [a,b], \\
0 & \text{otherwise}.
\end{cases}$
'''

uniform_cdf = r'''
$F(x) = \begin{cases}
0 & \text{for } x < a, \\
\frac{x - a}{b - a} & \text{for } x \in [a,b], \\
1 & \text{for } x > b.
\end{cases}$
'''

uniform_exp = r'$E[X] = \frac{1}{2}(a + b)$'
uniform_var = r'$Var(X) = \frac{1}{12}(b - a)^2$'

# Normal (Gaussian) distribution
normal_notation = r'$X \sim N(\mu, \sigma^2)$'
normal_pdf = r'$f(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(x - \mu)^2}{2\sigma^2}}$'
normal_cdf = r'$F(x) = \frac{1}{2} \left[1 + \text{erf}\left(\frac{x - \mu}{\sigma \sqrt{2}}\right)\right]$'
normal_exp = r'$E[X] = \mu$'
normal_var = r'$Var(X) = \sigma^2$'

# Exponential distribution
exponential_notation = r'$X \sim Exp(\lambda)$'

exponential_pdf = r'''
$f(x) = 
\begin{cases}
\lambda e^{-\lambda x} & \text{if } x \geq 0, \\
0 & \text{if } x < 0.
\end{cases}$
'''

exponential_cdf = r'''
$F(x) = 
\begin{cases}
1 - e^{-\lambda x} & \text{if } x \geq 0, \\
0 & \text{if } x < 0.
\end{cases}$
'''

exponential_exp = r'$E[X] = \frac{1}{\lambda}$'
exponential_var = r'$Var(X) = \frac{1}{\lambda^2}$' 

# Gamma distribution
gamma_notation = r'$X \sim \Gamma (\alpha, \beta) \equiv \operatorname{Gamma}(\alpha, \beta)$'
gamma_pdf = r'$f(x) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha - 1} e^{-\beta x} \quad \text{ for } x \ge 0 \quad \alpha, \beta \ge 0$'
gamma_cdf = r'$F(x) = \frac{1}{\Gamma(\alpha)} \gamma(\alpha, \beta x)$'
gamma_exp = r'$E[X] = \frac{\alpha}{\beta}$'
gamma_var = r'$Var(X) = \frac{\alpha}{\beta^2}$'