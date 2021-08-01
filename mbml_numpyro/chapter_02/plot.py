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
    """
    Plots Figure 2.15 from `response` array, where incorrect answers
    are colored with the relevant skill needed to answer the question.
    Will overlay, at most, two skills onto a single question.

    :param cmap ListedColormap: Colormap object generated from a list of colors.
    :param skills_key List[List[int]]: Sparse list of length `n_questions`, where
        each value is the list of required skills for question n
    :param color_skills: Dict[str, int]: mapping between the skill needed for question
        and index of `cmap`. Used to repalce values of `respones` array
    :param second_skill color Optional[Dict[int, str]]: Optional mapping between the second
        skill of a question, int, and the color, str name, to overlay onto the first skill
    :param skill_legend Optional[Dict[str, str]]: Optinal dict for the figure legend where the
        keys are color name, and the values are the labels for the skills
    :param figsize Tuple[int, int]: size of the figure, passed to `matplotlib.pyplot.subplots`
    :param copy Bool: Either modify the `response` array inplace when plotting or not
    :param return_fig_ax Bool: Either return the `fig` and `ax` or not
    """

    cmap: ListedColormap
    skills_key: List[List[int]]
    color_skills: Dict[str, int]
    second_skill_color: Optional[Dict[int, str]] = None
    skill_legend: Optional[Dict[str, str]] = None
    figsize: Tuple[int, int] = (20, 10)
    copy: bool = True
    return_fig_ax: bool = False
    _extent: Optional[Tuple[int, int, int, int]] = field(
        default=None, repr=False, init=False
    )
    _rectangles: List[Rectangle] = field(
        default_factory=list, compare=False, hash=False, repr=False, init=False
    )

    def __post_init__(self):
        pass

    @property
    def extent(self) -> Tuple[int, int, int, int]:
        return self._extent

    @property
    def rectangles(self) -> List[Rectangle]:
        return self._rectangles

    @rectangles.deleter
    def rectangles(self) -> None:
        self._rectangles.clear()

    @property
    def n_skills(self) -> int:
        return max(map(max, self.skills_key)) + 1

    def _pre_process(self, responses: pd.DataFrame):
        if self.extent is None:
            self._extent = (0, responses.shape[1], responses.shape[0], 0)

        answers = responses.copy() if self.copy else responses

        for wrong_answer_xy in np.transpose(np.where(answers == 0)):
            person, question = wrong_answer_xy
            required_skills = self.skills_key[question]
            answers.at[person, question] = self.color_skills[
                ",".join(map(str, required_skills))
            ]

        if self.second_skill_color is not None:
            # https://stackoverflow.com/a/32768985/3587374
            wrong_answers = answers[answers != 1].stack()

            for idx, value in wrong_answers.iteritems():
                person, question = idx
                skill = self.skills_key[question]
                if len(skill) > 1 and value != 0:
                    c = self.second_skill_color[skill[-1]]
                    r = Rectangle((question, person + 0.5), 1.0, 0.5, fc=c, ec="none")
                    self.rectangles.append(r)

        return answers

    def plot(self, responses: pd.DataFrame):
        """
        Plot the Figure 2.15 from `responses`.

        :param responses pd.DataFrame: DataFrame of the array of the correct and incorrect
            responses.
            Except `responses.shape` == (n_questions, n_participants)
            Except `responses.dtype` == int
            Except `respones.columns` == range(n_participants)
            Except `respones.index` == range(n_questions)
        """

        assert (responses.index == pd.RangeIndex(responses.shape[0])).all()
        assert (responses.columns == pd.RangeIndex(responses.shape[1])).all()

        if self.rectangles:
            del self.rectangles

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

        if self.skill_legend is not None:

            # https://stackoverflow.com/a/46267860/3587374
            legend_handles = [
                ax.plot(
                    [],
                    marker="s",
                    markersize=15,
                    linestyle="",
                    color=color,
                    label=skill,
                )[0]
                for (color, skill) in self.skill_legend.items()
            ]

            ax.legend(
                handles=legend_handles,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.08),
                ncol=self.n_skills if self.n_skills <= 7 else 7,
            )

        if self.return_fig_ax:
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
