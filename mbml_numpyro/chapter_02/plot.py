"""
Module for plotting
"""
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from dataclasses import dataclass, field


@dataclass
class DisplaySkill:
    cmap: ListedColormap
    skills_key: List[List[int]]
    color_skills: Dict
    extent: Optional[Tuple[int, int, int, int]] = None
    rectangles: List[Rectangle] = field(
        default_factory=list, compare=False, hash=False, repr=False
    )
    figsize: tuple[int, int] = (20, 10)
    copy: bool = True
    return_fig: bool = False

    def __post_init__(self):
        pass

    def _pre_process(self, responses: pd.DataFrame):
        if self.extent is None:
            self.extent = (0, responses.shape[1], responses.shape[0], 0)

        answers = responses.copy() if self.copy else responses

        for wrong_answer_xy in np.transpose(np.where(answers == 0)):
            person, question = wrong_answer_xy
            required_skills = self.skills_key[question]
            answers.at[person, question] = self.color_skills[
                ",".join(map(str, required_skills))
            ]

        # https://stackoverflow.com/a/32768985/3587374
        wrong_answers = answers[answers != 1].stack()

        for idx, value in wrong_answers.iteritems():
            person, question = idx
            skill = self.skills_key[question]
            if len(skill) > 1 and value != 0:
                c = self.color_skills[skill[-1]]
                r = Rectangle((question, person + 0.5), 1.0, 0.5, fc=c, ec="none")
                self.rectangles.append(r)

        return answers

    def plot(self, responses: pd.DataFrame):

        if self.rectangles:
            self.rectangles.clear()

        responses = self._pre_process(responses)

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.imshow(responses.values, cmap=self.cmap, extent=self.extent)
        ax.set_xticks(np.arange(0, responses.shape[-1], 1))
        ax.set_yticks(np.arange(0, responses.shape[0], 1))
        ax.grid(color="k", linewidth=1)
        for rectange in self.rectangles:
            ax.add_patch(rectange)
        ax.set_aspect("equal")
        ax.set_ylabel("People")
        ax.set_xlabel("Questions")
        ax.set(yticklabels=[])
        ax.set(xticklabels=[])

        if self.return_fig:
            return fig, ax


def plot_inferred_true_skills(
    inferred: np.array, true: np.array, figsize=(5, 5), return_fig=False
):
    assert inferred.shape == true.shape
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax[0].imshow(inferred, cmap="Greys_r")
    ax[0].set_aspect("equal")
    ax[0].set_ylabel("People")
    ax[0].set_xlabel("Skills")
    ax[0].set(yticklabels=[])
    ax[0].set(xticklabels=[])
    ax[1].imshow(true, cmap="Greys_r")
    ax[1].set_aspect("equal")
    ax[1].set_xlabel("Skills")
    ax[1].set(yticklabels=[])
    ax[1].set(xticklabels=[])

    if return_fig:
        return fig, ax