# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python [conda env:mbml_numpyro]
#     language: python
#     name: conda-env-mbml_numpyro-py
# ---

# +
import operator
import sys
from functools import reduce
from typing import List, Tuple

import arviz as az
import daft
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Latex as lt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

import numpyro
import numpyro.distributions as dist
from numpyro.contrib.funsor import config_enumerate, enum, infer_discrete
from numpyro.handlers import seed, trace
from numpyro.infer import MCMC, NUTS, DiscreteHMCGibbs
from numpyro.infer.util import Predictive

# +
#sys.path.append("..")
# -

# %load_ext autoreload
# %autoreload 2

from mbml_numpyro.chapter_02.plot import DisplaySkill, plot_inferred_true_skills
from mbml_numpyro.chapter_02.data import chapter_02_data

plt.rcParams["figure.dpi"] = 75
font = {"size": 18}
plt.rc("font", **font)

rng_key = jax.random.PRNGKey(2)
rng_key

# # Chapter 2
# ## Assessing People’s Skills
# ## 2.1 A model is a set of assumptions
#
# **A model = A set of assumptions about the data**
#
# #### Figure 2.2

# +
pgm = daft.PGM(node_ec="k",)
pgm.add_node(
    "csharp_param",
    r"$Bernoulli(0.5)$",
    0.5,
    2,
    offset=(0.1, 10),
    scale=3,
    fixed=True,
    shape="rectangle",
    label_params={"color": "k"},
)
pgm.add_node("csharp", r"$csharp$", 0.5, 1, aspect=4, label_params={"color": "k"})
pgm.add_node(
    "sql_param",
    r"$Bernoulli(0.5)$",
    4.5,
    2,
    offset=(0.1, 10),
    scale=3,
    fixed=True,
    shape="rectangle",
    label_params={"color": "k"},
)
pgm.add_node("sql", r"$sql$", 4.5, 1, aspect=4, label_params={"color": "k"})

pgm.add_edge("csharp_param", "csharp", plot_params={"head_width": 0.25,})
pgm.add_edge("sql_param", "sql", plot_params={"head_width": 0.25,})
pgm.render();
# -

# #### 2.1
# $P\,(\texttt{csharp},\texttt{sql}) = \textrm{Bernoulli}(\texttt{csharp};0.5)\;\textrm{Bernoulli}(\texttt{csharp};0.5)$

# #### Table 2.1

isCorrect1_cpt = np.array([[0.9, 0.1], [0.2, 0.8]])
pd.DataFrame(
    isCorrect1_cpt,
    columns=["isCorrect1=true", "isCorrect1=false"],
    index=["csharp=true", "csharp=false"],
)

# #### Figure 2.3

# +
pgm = daft.PGM(node_ec="k",)
pgm.add_node(
    "csharp_param",
    r"$Bernoulli(0.5)$",
    0.5,
    2,
    offset=(0.1, 10),
    scale=3,
    fixed=True,
    shape="rectangle",
    label_params={"color": "k"},
)
pgm.add_node("csharp", r"$csharp$", 0.5, 1, aspect=4, label_params={"color": "k"})
pgm.add_node(
    "addnoise_param_00",
    r"$AddNoise$",
    0.5,
    0.0625,
    offset=(70, -10),
    scale=3,
    fixed=True,
    shape="rectangle",
    label_params={"color": "k"},
)
pgm.add_node(
    "isCorrect1", r"$isCorrect1$", 0.5, -1, aspect=5, label_params={"color": "k"}
)

pgm.add_node(
    "sql_param",
    r"$Bernoulli(0.5)$",
    4.5,
    2,
    offset=(0.1, 10),
    scale=3,
    fixed=True,
    shape="rectangle",
    label_params={"color": "k"},
)
pgm.add_node("sql", r"$sql$", 4.5, 1, aspect=4, label_params={"color": "k"})
pgm.add_node(
    "addnoise_param_01",
    r"$AddNoise$",
    4.5,
    0.0625,
    offset=(70, -10),
    scale=3,
    fixed=True,
    shape="rectangle",
    label_params={"color": "k"},
)
pgm.add_node(
    "isCorrect2", r"$isCorrect2$", 4.5, -1, aspect=5, label_params={"color": "k"}
)

pgm.add_edge("csharp_param", "csharp", plot_params={"head_width": 0.25,})
pgm.add_edge("csharp", "addnoise_param_00", directed=False)
pgm.add_edge("addnoise_param_00", "isCorrect1", plot_params={"head_width": 0.25,})

pgm.add_edge("sql_param", "sql", plot_params={"head_width": 0.25,})
pgm.add_edge("sql", "addnoise_param_01", directed=False)
pgm.add_edge("addnoise_param_01", "isCorrect2", plot_params={"head_width": 0.25,})

pgm.render();
# -

# #### 2.2
# $P\,(\texttt{csharp},\texttt{sql},\texttt{isCorrect1},\texttt{isCorrect2}) = \textrm{Bernoulli}(\texttt{csharp};0.5)\;\textrm{Bernoulli}(\texttt{csharp};0.5)\; \textrm{AddNoise}(\texttt{isCorrect1}|\texttt{csharp})\; \textrm{AddNoise}(\texttt{isCorrect2}|\texttt{sql})$

# #### Figure 2.4

# +
pgm = daft.PGM(node_ec="k",)
pgm.add_node(
    "csharp_param",
    r"$Bernoulli(0.5)$",
    0.5,
    2,
    offset=(0.1, 10),
    scale=3,
    fixed=True,
    shape="rectangle",
    label_params={"color": "k"},
)
pgm.add_node("csharp", r"$csharp$", 0.5, 1, aspect=4, label_params={"color": "k"})

pgm.add_node(
    "sql_param",
    r"$Bernoulli(0.5)$",
    4.5,
    2,
    offset=(0.1, 10),
    scale=3,
    fixed=True,
    shape="rectangle",
    label_params={"color": "k"},
)
pgm.add_node("sql", r"$sql$", 4.5, 1, aspect=4, label_params={"color": "k"})

pgm.add_node(
    "And_param",
    r"$And$",
    2.5,
    0,
    offset=(0.1, 20),
    scale=3,
    fixed=True,
    shape="rectangle",
    label_params={"color": "k"},
)
pgm.add_node(
    "hasSkills", r"$hasSkills$", 2.5, -1, aspect=4, label_params={"color": "k"}
)

pgm.add_edge("csharp_param", "csharp", plot_params={"head_width": 0.25,})
pgm.add_edge("sql_param", "sql", plot_params={"head_width": 0.25,})
pgm.add_edge("csharp", "And_param", directed=False)
pgm.add_edge("sql", "And_param", directed=False)
pgm.add_edge("And_param", "hasSkills", plot_params={"head_width": 0.25,})

pgm.render();
# -

# #### 2.3
# $P\,(\texttt{csharp},\texttt{sql},\texttt{hasSkill}) = \textrm{Bernoulli}(\texttt{csharp};0.5)\;\textrm{Bernoulli}(\texttt{csharp};0.5)\; \textrm{And}(\texttt{hasSkill}|\texttt{csharp},\texttt{sql})$

# #### Figure 2.5

# +
pgm = daft.PGM(node_ec="k",)
pgm.add_node(
    "csharp_param",
    r"$Bernoulli(0.5)$",
    0.5,
    3,
    offset=(0.1, 10),
    scale=3,
    fixed=True,
    shape="rectangle",
    label_params={"color": "k"},
)
pgm.add_node("csharp", r"$csharp$", 0.5, 2.25, aspect=4, label_params={"color": "k"})
pgm.add_node(
    "addnoise_param_00",
    r"$AddNoise$",
    0.5,
    -0.25,
    offset=(70, -10),
    scale=3,
    fixed=True,
    shape="rectangle",
    label_params={"color": "k"},
)
pgm.add_node(
    "isCorrect1", r"$isCorrect1$", 0.5, -1, aspect=5, label_params={"color": "k"}
)

pgm.add_node(
    "sql_param",
    r"$Bernoulli(0.5)$",
    4.5,
    3,
    offset=(0.1, 10),
    scale=3,
    fixed=True,
    shape="rectangle",
    label_params={"color": "k"},
)
pgm.add_node("sql", r"$sql$", 4.5, 2.25, aspect=4, label_params={"color": "k"})
pgm.add_node(
    "addnoise_param_01",
    r"$AddNoise$",
    4.5,
    -0.25,
    offset=(70, -10),
    scale=3,
    fixed=True,
    shape="rectangle",
    label_params={"color": "k"},
)
pgm.add_node(
    "isCorrect2", r"$isCorrect2$", 4.5, -1, aspect=5, label_params={"color": "k"}
)


pgm.add_node(
    "And_param",
    r"$And$",
    8.5,
    1.25,
    offset=(0.1, 20),
    scale=3,
    fixed=True,
    shape="rectangle",
    label_params={"color": "k"},
)
pgm.add_node(
    "hasSkills", r"$hasSkills$", 8.5, 0.5, aspect=4, label_params={"color": "k"}
)
pgm.add_node(
    "addnoise_param_02",
    r"$AddNoise$",
    8.5,
    -0.25,
    offset=(70, -10),
    scale=3,
    fixed=True,
    shape="rectangle",
    label_params={"color": "k"},
)
pgm.add_node(
    "isCorrect3", r"$isCorrect3$", 8.5, -1, aspect=5, label_params={"color": "k"}
)

pgm.add_edge("csharp_param", "csharp", plot_params={"head_width": 0.25,})
pgm.add_edge("csharp", "addnoise_param_00", directed=False)
pgm.add_edge("addnoise_param_00", "isCorrect1", plot_params={"head_width": 0.25,})

pgm.add_edge("sql_param", "sql", plot_params={"head_width": 0.25,})
pgm.add_edge("sql", "addnoise_param_01", directed=False)
pgm.add_edge("addnoise_param_01", "isCorrect2", plot_params={"head_width": 0.25,})

pgm.add_edge("csharp", "And_param", directed=False)
pgm.add_edge("sql", "And_param", directed=False)
pgm.add_edge("And_param", "hasSkills", plot_params={"head_width": 0.25,})
pgm.add_edge("hasSkills", "addnoise_param_02", directed=False)
pgm.add_edge("addnoise_param_02", "isCorrect3", plot_params={"head_width": 0.25,})

pgm.render();
# -

# #### 2.4
# $P\,(\texttt{csharp},\texttt{sql},\texttt{isCorrect1},\texttt{isCorrect2},\texttt{isCorrect3}) = \textrm{Bernoulli}(\texttt{csharp};0.5)\;\textrm{Bernoulli}(\texttt{csharp};0.5)\; \textrm{AddNoise}(\texttt{isCorrect1}|\texttt{csharp})\; \textrm{AddNoise}(\texttt{isCorrect2}|\texttt{sql})\; \textrm{And}(\texttt{hasSkill}|\texttt{csharp},\texttt{sql})\; \textrm{AddNoise}(\texttt{isCorrect3}|\texttt{hasSkills})$

# ## 2.2 Testing out the model

# #### Table 2.4

expected = pd.DataFrame(
    [
        (False, False, False, 0.101, 0.101),
        (True, False, False, 0.802, 0.034),
        (False, True, False, 0.034, 0.802),
        (True, True, False, 0.561, 0.561),
        (False, False, True, 0.148, 0.148),
        (True, False, True, 0.862, 0.326),
        (False, True, True, 0.326, 0.862),
        (True, True, True, 0.946, 0.946),
    ],
    columns=["IsCorrect1", "IsCorrect2", "IsCorrect2", "P(cshapr)", "P(sql)"],
)
expected

responses_check = jnp.array(
    [
        [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    ]
)
skills_needed_check = [[0], [1], [0, 1]]


def model_00(
    graded_responses, skills_needed: list[list[int]], prob_mistake=0.1, prob_guess=0.2
):
    n_questions, n_participants = graded_responses.shape
    n_skills = max(map(max, skills_needed)) + 1

    participants_plate = numpyro.plate("participants_plate", n_participants)

    with participants_plate:
        skills = []
        for s in range(n_skills):
            skills.append(numpyro.sample("skill_{}".format(s), dist.Bernoulli(0.5)))

    for q in range(n_questions):
        has_skills = reduce(operator.mul, [skills[i] for i in skills_needed[q]])
        prob_correct = has_skills * (1 - prob_mistake) + (1 - has_skills) * prob_guess
        isCorrect = numpyro.sample(
            "isCorrect_{}".format(q),
            dist.Bernoulli(prob_correct).to_event(1),
            obs=graded_responses[q],
        )


# + tags=[]
numpyro.render_model(
    model_00, (responses_check, skills_needed_check), render_distributions=True
)

# +
nuts_kernel = NUTS(model_00)

kernel = DiscreteHMCGibbs(nuts_kernel, modified=True)

mcmc = MCMC(kernel, num_warmup=200, num_samples=1000, num_chains=4)
mcmc.run(rng_key, responses_check, skills_needed_check)
mcmc.print_summary()
# -

ds = az.from_numpyro(mcmc)

az.plot_trace(ds);

# ## 2.2 Testing out the model

four_skills_needed_check = skills_needed_check
four_skills_needed_check.append([0, 1])
four_skills_needed_check

four_responses_check = jnp.array(
    [
        [0, 0, 0, 0,],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 1, 1],
        [1, 0, 0, 0],
        [1, 0, 0, 1],
        [1, 0, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 0],
        [1, 1, 0, 1],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
    ]
)
four_responses_check = four_responses_check.T
four_responses_check

numpyro.render_model(
    model_00, (four_responses_check, four_skills_needed_check), render_distributions=True
)

# +
nuts_kernel = NUTS(model_00)

kernel = DiscreteHMCGibbs(nuts_kernel, modified=True)

mcmc = MCMC(kernel, num_warmup=200, num_samples=1000, num_chains=4)
mcmc.run(rng_key, four_responses_check, four_skills_needed_check)
mcmc.print_summary()
# -

ds = az.from_numpyro(mcmc)

az.plot_trace(ds);

# +
# with numpyro.handlers.seed(rng_seed=0):
#     print(enum(learning_skills)(example_data, example_skills_needed).shape)
# -

# ## 2.4 Moving to real data 

chapter_02_data

# #### Table 2.6

chapter_02_data.raw_data

# #### Figure 2.14

# + tags=[]
fig, ax = plt.subplots(2, 1, figsize=(15, 15))
ax[0].spy(chapter_02_data.skills_key.values.T, markersize=15, marker="s", color="k")
ax[0].set_ylabel("skills")
ax[1].spy(
    chapter_02_data.raw_data.loc[1:, "Q1":"Q48"]
    == chapter_02_data.raw_data.loc[0, "Q1":"Q48"],
    markersize=15,
    marker="s",
    color="k",
)
ax[1].set_ylabel("People")
plt.tight_layout()
# -

# #### Figure 2.15

# +
color_skills = {str(i - 2): i for i in range(2, 8)}
color_skills["1,6"] = 8
color_skills["0,6"] = 9
color_skills["1,4"] = 10

second_skill = {6: "magenta", 4: "orange"}
# -

figure_2pt15 = DisplaySkill(
    ListedColormap(
        [
            "white",
            "red",
            "orange",
            "yellow",
            "green",
            "cyan",
            "blue",
            "orange",
            "red",
            "cyan",
        ]
    ),
    color_skills,
    skills_needed=chapter_02_data.skills_needed,
    second_skill_color=second_skill,
    skill_legend=dict(
        red="Core",
        orange="OOP",
        yellow="Life Cycle",
        green="Web Apps",
        cyan="Desktop apps",
        blue="SQL",
        magenta="C#",
    ),
)

# + tags=[]
figure_2pt15.plot(chapter_02_data.responses.T)
# -

# #### Figure 2.17

# + tags=[]
numpyro.render_model(
    model_00,
    (
        jnp.array(chapter_02_data.responses),
        chapter_02_data.skills_needed,
    ),
    render_distributions=True,
)

# + tags=[]
nuts_kernel = NUTS(model_00)

kernel = DiscreteHMCGibbs(nuts_kernel, modified=True)

mcmc = MCMC(kernel, num_warmup=200, num_samples=1000, num_chains=4)
mcmc.run(rng_key, jnp.array(chapter_02_data.responses), chapter_02_data.skills_needed)
mcmc.print_summary()
# -

ds = az.from_numpyro(mcmc)

# + tags=[]
az.plot_trace(ds);

# + tags=[]
az.mcse(ds)
# -

az.rhat(ds)

# + tags=[]
az.ess(ds)
# -

inferred_skills_model_00 = np.array(
    jnp.vstack([value.mean(0) for value in mcmc.get_samples().values()]).T
)

from numpyro.infer import log_likelihood
from jax.scipy.special import logsumexp

from numpyro.infer.util import log_density, potential_energy

log_density_model_00 = log_density(model_00, (jnp.array(chapter_02_data.responses), chapter_02_data.skills_needed), dict(prob_mistake=0.1, prob_guess=0.2), mcmc.get_samples())

# + tags=[]
log_density_model_00[0]
# -

ll_model_00 = log_likelihood(model_00, mcmc.get_samples(), jnp.array(chapter_02_data.responses), chapter_02_data.skills_needed)

# + tags=[]
ll_model_00
# -

# #### Figure 2.18

plot_inferred_true_skills(
    inferred_skills_model_00,
    chapter_02_data.self_assessed.values.astype("int32"),
    titles=("Inferred skills", "Self-assessed skills"),
    fontdict=dict(fontsize=12)
)


# ## 2.5 Diagnosing the problem

# ### Checking the inference algorithm

def model_01(
    skills_needed: list[list[int]],
    graded_responses=None,
    self_skills=None,
    prob_mistake=0.1,
    prob_guess=0.2,
    size_as: tuple[int, int] = (48, 22),
):
    n_questions, n_participants = (
        graded_responses.shape if graded_responses is not None else size_as
    )
    if graded_responses is None:
        graded_responses = [None] * n_questions
    n_skills = max(map(max, skills_needed)) + 1

    participants_plate = numpyro.plate("participants_plate", n_participants)

    with participants_plate:
        skills = []
        for s in range(n_skills):
            skill_obs = None if self_skills is None else self_skills[s]
            skills.append(numpyro.sample("skill_{}".format(s), dist.Bernoulli(0.5), obs=skill_obs))

    has_skills_as = []
    isCorrect_as = []
    for q in range(n_questions):
        has_skills = reduce(operator.mul, [skills[i] for i in skills_needed[q]])
        has_skills_as.append(has_skills)
        prob_correct = has_skills * (1 - prob_mistake) + (1 - has_skills) * prob_guess
        isCorrect = numpyro.sample(
            "isCorrect_{}".format(q),
            dist.Bernoulli(prob_correct).to_event(1),
            obs=graded_responses[q],
        )
        isCorrect_as.append(isCorrect)

    return skills, has_skills_as, isCorrect_as


# #### Ancestral sampling

with numpyro.handlers.seed(rng_seed=0):
    s = model_01(chapter_02_data.skills_needed)

skills_as, has_skills_as, isCorrect_as = s
skills_as = jnp.vstack(skills_as)
has_skills_as = jnp.vstack(has_skills_as)
isCorrect_as = jnp.vstack(isCorrect_as)

# #### Figure 2.19

# + tags=[]
fig_2pt19 = plt.figure(constrained_layout=True, figsize=(20,20));

gs = fig_2pt19.add_gridspec(nrows=3, ncols=3)
fig_2pt19_ax1 = fig_2pt19.add_subplot(gs[0, 0])
fig_2pt19_ax2 = fig_2pt19.add_subplot(gs[0, 1:])
fig_2pt19_ax3 = fig_2pt19.add_subplot(gs[1, 1:])

fig_2pt19_ax1.imshow(skills_as.T, cmap="Greys_r")
fig_2pt19_ax1.set_aspect("equal");
fig_2pt19_ax1.set_ylabel("People");
fig_2pt19_ax1.set_xlabel("Skills");
fig_2pt19_ax1.set(yticklabels=[]);
fig_2pt19_ax1.set(xticklabels=[]);

figure_2pt15.plot(pd.DataFrame(has_skills_as.T), ax=fig_2pt19_ax2, plot_legend=False)

figure_2pt15.plot(pd.DataFrame(isCorrect_as.T), ax=fig_2pt19_ax3, bbox_to_anchor=(0.5, -0.2))

# +
nuts_kernel = NUTS(model_00)

kernel = DiscreteHMCGibbs(nuts_kernel, modified=True)

mcmc = MCMC(kernel, num_warmup=200, num_samples=1000, num_chains=4)
mcmc.run(rng_key, isCorrect_as, chapter_02_data.skills_needed)
mcmc.print_summary()
# -

inferred_skills_from_as = np.array(
    jnp.vstack([value.mean(0) for value in mcmc.get_samples().values()]).T
)

# #### Figure 2.20

plot_inferred_true_skills(
    inferred_skills_from_as,
    skills_as.T,
    titles=("Inferred skills", "Self-assessed skills"),
    fontdict=dict(fontsize=12)
)

# ### Working out what is wrong with the model

with numpyro.handlers.seed(rng_seed=0):
    s = model_01(chapter_02_data.skills_needed, self_skills=chapter_02_data.self_assessed.astype(int).T.values)

skills_as, has_skills_as, isCorrect_as = s
skills_as = jnp.vstack(skills_as)
has_skills_as = jnp.vstack(has_skills_as)
isCorrect_as = jnp.vstack(isCorrect_as)

# #### Figure 2.21

# + tags=[]
fig, ax = plt.subplots(2, 1, figsize=(10,10))
fig.tight_layout()

figure_2pt15.plot(pd.DataFrame(isCorrect_as.T), ax=ax[0], plot_legend=False)

figure_2pt15.plot(chapter_02_data.responses.T, ax=ax[1])
# -

# #### Figure 2.22

# + tags=[]
ind = np.arange(chapter_02_data.responses.shape[0])
width = 0.4

fig, ax = plt.subplots(figsize=(30,5))
ax.bar(ind, isCorrect_as.T.mean(0), width, label="Predicted", color="b")
ax.bar(ind + width, chapter_02_data.responses.mean(1).values, width, label="Actual", color="r")

ax.set_xlabel("Question Number")
ax.set_ylabel("Fraction correct")

plt.xticks(ind + width / 2, map(str, ind + 1))
plt.legend(loc='best')
plt.grid(axis="y")
plt.show()


# -

# ## 2.6 Learning the guess probabilities

def model_02(
    graded_responses, skills_needed: list[list[int]], prob_mistake=0.1,
):
    n_questions, n_participants = graded_responses.shape
    n_skills = max(map(max, skills_needed)) + 1
    
    with numpyro.plate("questions_plate", n_questions):
        prob_guess = numpyro.sample("prob_guess", dist.Beta(2.5, 7.5))

    participants_plate = numpyro.plate("participants_plate", n_participants)

    with participants_plate:
        skills = []
        for s in range(n_skills):
            skills.append(numpyro.sample("skill_{}".format(s), dist.Bernoulli(0.5)))

    for q in range(n_questions):
        has_skills = reduce(operator.mul, [skills[i] for i in skills_needed[q]])
        prob_correct = has_skills * (1 - prob_mistake) + (1 - has_skills) * prob_guess[q]
        isCorrect = numpyro.sample(
            "isCorrect_{}".format(q),
            dist.Bernoulli(prob_correct).to_event(1),
            obs=graded_responses[q],
        )


numpyro.render_model(
    model_02, (responses_check, skills_needed_check), render_distributions=True
)

# + tags=[]
nuts_kernel = NUTS(model_02)

kernel = DiscreteHMCGibbs(nuts_kernel, modified=True)

mcmc = MCMC(kernel, num_warmup=200, num_samples=1000, num_chains=4)
mcmc.run(rng_key, jnp.array(chapter_02_data.responses), chapter_02_data.skills_needed)
mcmc.print_summary()
# -

ds = az.from_numpyro(mcmc)

# + tags=[]
az.plot_trace(ds);
# -

mcmc.get_samples()

inferred_skills_model_02 = np.array(
    jnp.vstack([value.mean(0) for value in list(mcmc.get_samples().values())[1:]]).T
)

log_density_model_00 = log_density(model_00, (jnp.array(chapter_02_data.responses), chapter_02_data.skills_needed), dict(prob_mistake=0.1, prob_guess=0.2), mcmc.get_samples())

# + tags=[]
log_density_model_02 = log_density(model_02, (jnp.array(chapter_02_data.responses), chapter_02_data.skills_needed), dict(prob_mistake=0.1), mcmc.get_samples())
# -

ll_model_02 = log_likelihood(model_02, mcmc.get_samples(), jnp.array(chapter_02_data.responses), chapter_02_data.skills_needed)

# + tags=[]
ll_model_02
# -

plot_inferred_true_skills(
    inferred_skills_model_02,
    chapter_02_data.self_assessed.values.astype("int32"),
    titles=("Inferred skills", "Self-assessed skills"),
    fontdict=dict(fontsize=12)
)

import seaborn as sns

iris = sns.load_dataset("iris")
iris

np.array(mcmc.get_samples()["prob_guess"])[:, slice(0,-1,5)].shape

np.array(mcmc.get_samples()["prob_guess"])[:, 0:-1:5].shape

sns.kdeplot(data=np.array(mcmc.get_samples()["prob_guess"])[:, 0:-1:5])

mcmc.get_samples()["prob_guess"]

ds.posterior.data_vars["prob_guess"].shape

np.array(mcmc.get_samples()["prob_guess"].mean(0))

np.array(mcmc.get_samples()["prob_guess"].std(0))

np.quantile(np.array(mcmc.get_samples()["prob_guess"]), q=0.5, axis=0)

ax.errorbar(ind, np.array(mcmc.get_samples()["prob_guess"].mean(0)), yerr=np.array(mcmc.get_samples()["prob_guess"].std(0)), label="Predicted", color="b")

# + tags=[]
(np.percentile(np.array(mcmc.get_samples()["prob_guess"]), q=25, axis=0), np.percentile(np.array(mcmc.get_samples()["prob_guess"]), q=75, axis=0))

# +
ind = np.arange(1,49)

fig, ax = plt.subplots(figsize=(30,5))
ax.bar(ind, np.array(mcmc.get_samples()["prob_guess"].mean(0)), label="Predicted", color="b")
ax.errorbar(ind, np.array(mcmc.get_samples()["prob_guess"].mean(0)), yerr=(np.percentile(np.array(mcmc.get_samples()["prob_guess"]), q=25, axis=0), np.percentile(np.array(mcmc.get_samples()["prob_guess"]), q=75, axis=0)), label="Predicted", color="k", fmt="o", capsize=5, capthick=2)
ax.grid(axis="y")
ax.set_xticks(range(1,49));
ax.set_xlim((0,49))

# [1::2] means start from the second element in the list and get every other element
for tick in ax.xaxis.get_major_ticks()[1::2]:
    tick.set_pad(20)
# -

az.plot_forest(
    {
        "prob_guess": mcmc.get_samples()["prob_guess"].T,
    }
);


