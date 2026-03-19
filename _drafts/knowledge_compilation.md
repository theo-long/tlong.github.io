---
layout: default
title: "Efficient Discrete Inference with Knowledge Compilation"
date: "2026-03-18"
tags: ["probabilistic programming", "discrete inference", "knowledge compilation"]
permalink: "/knowledge-compilation"
---

# Discrete Probabilistic Programs

Many existing probabilistic programming languages such as Stan, Pyro, and Gen allow us to write expressive programs with a wide range of discrete and continuous distributions, containing complex logic that combines latent variables and allowing for models with stochastic support. This expressivity does come at a price: in most cases we can only hope to perform *approximate* inference, using something like MCMC or variational inference. Exactly computing (and representing) the posterior distribution becomes quickly intractable once you allow non-conjugate priors or arbitrary interactions between latent variables.

Although there are many restricted classes of models and PPLs where exact inference is possible (affine combinations of Gaussians, sum-product networks, to name two), one natural family of models are those involving only *discrete* parameters. In fact, we can simplify even further and consider only parameters taking boolean values, since more complex categorical distributions can instead be constructed using multiple boolean parameters (e.g. by considering the joint distribution on `(b1, b2)` which has 4 different possible values).

We can write a simple model that flips three coins, and combines the results:

$$
\begin{aligned}
b_1 &\sim \text{Bernoulli}(0.5) \\
b_2 &\sim \text{Bernoulli}(0.2) \\
b_3 &\sim \text{Bernoulli}(0.1) \\
\text{result} &= (b_1 \land b_2) \lor (\neg b_1 \land \neg b_2) \\
\end{aligned}
$$

Our goal is now to infer the posterior distribution on $b_1$, given that we observe $\text{result}$ is true, which we can calculate using Bayes' Theorem:
$$
P(b_1 = \text{TRUE} \mid \text{result} = \text{TRUE}) = \frac{P(b_1 \land \text{result})}{P(\text{result})}
$$
where for a boolean expression $b$ we use the shorthand $P(b) \equiv P(b = \text{TRUE})$.

Because there are only a finite amount of values that the variables $(b_1, b_2, b_3)$ can take, we can calculate both the numerator and denominator by straightforward enumeration. That is, we iterate over all possible values of $(b_1, b_2, b_3)$, and calculate the probability for each (using the Bernoulli parameters which are 0.5, 0.2, and 0.1). Then for each assignment $(b_1, b_2, b_3)$, if $\text{result} \coloneq (b_1 \land b_2) \lor (\neg b_1 \land \neg b_2)$ is equal to $\text{TRUE}$, we add that probability to our value of $P(\text{result})$, and if $b_1$ is also $\text{TRUE}$, we add the probability to our counter for $P(b_1 \land \text{result})$.

If we write this out in python we get:
```python
bernoulli_params = {"b1":0.5, "b2":0.2, "b3":0.1}
p_result = 0.
p_result_and_b1 = 0.
for b1 in (True, False):
    for b2 in (True, False):
        for b3 in (True, False):
            prob_b1 = bernoulli_params["b1"] if b1 else 1 - bernoulli_params["b1"]
            prob_b2 = bernoulli_params["b2"] if b2 else 1 - bernoulli_params["b2"]
            prob_b3 = bernoulli_params["b3"] if b3 else 1 - bernoulli_params["b3"]
            prob = prob_b1 * prob_b2 * prob_b3

            result = (b1 and b2) or (not b1 and not b3)
            if result:
                p_result += prob
                if b1:
                    p_result_and_b1 += prob
                
posterior_prob = p_result_and_b1 / p_result
```

If you run this, you'll get $2 / 11 = 0.181818\ldots$ which is the correct answer!

However there's one major caveat - enumeration is *slow*. We have to iterate over all $2^N$ possible assignments to $N$ boolean variables, which becomes infeasible for even modest values of $N$. Enumeration is a brute force approach; it doesn't do anything smart, such as only calculating the probability of repeated subexpressions once, or avoid calculating probabilities for assignments where we can easily deduce that the result will be false.

If we want to be able to do discrete inference for larger models, we have to turn to something more efficient, which is where Knowledge Compilation comes in. But first, let's define a simple embedded DSL in python that will allow us to express discrete models and automate inference over them.

## A simple language for discrete probabilistic programs

Before we go further, let's define a simple functional language that allows us to express probabilistic models with discrete parameters. It has the following terms:
- `Let(name, expr, body)`: Assigns the expression `expr` to the variable `name` in the `body`, like a `let x = 5 in _` statement
- `Var(name)`: Allows us to reference a variable assigned using a `Let` expression
- `Flip(p)`: Represents a boolean variable which is `True` with probability `p` and `False` with probability `1-p`
- `True, False`: Represents the constant values `True` and `False`
- `IfThenElse(cond_expr, true_expr, false_expr)`: Allows us to branch based on a (possibly random) expression `cond_expr`
- `Observe(expr)` observe that `expr` evaluates to `True`. We can think of this statement as producing a *side effect* - it tells us that we should evaluate probabilities under the conditional distribution $P(\cdot \mid \texttt{expr == True})$. However the `Observe` expression itself always evaluates to true.

Now let's write some programs! First we can recreate the example from the previous section:
```python
expr = Let(
    "b1",
    Flip(0.5),
    Let(
        "b2",
        Flip(0.2),
        Let(
            "b3",
            Flip(0.1),
            Let(
                "result",
                IfThenElse(
                    Var("b1"),
                    Var("b2"),
                    # This expression is just ~b3
                    IfThenElse(
                        Var("b3"),
                        False,
                        True,
                    ),
                ),
                # We assign the Observe to a dummy variable since it always evals to True
                Let(
                    "_", 
                    Observe(Var("result")), 
                    Var("b1"), # Finally, we return b1 which is the value of our expression
                )
            )
        )
    )
)
```

But we also write more complex models that represent more interesting situations: