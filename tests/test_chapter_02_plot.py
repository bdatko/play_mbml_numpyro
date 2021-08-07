import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import ListedColormap
from mbml_numpyro import DisplaySkill, chapter_02_data


@pytest.mark.parametrize(
    "input,color_skills,skills_key,expected",
    [
        (
            pd.DataFrame(np.array([[1, 1, 1], [0, 1, 1], [0, 1, 0]], dtype="int32")),
            {"0": 2, "0,6": 9},
            [[0], [0], [0, 6]],
            np.array([[1, 1, 1], [2, 1, 1], [2, 1, 9]], dtype="int32"),
        )
    ],
)
def test_DisplaySkill___pre_process(input, color_skills, skills_key, expected):
    # Arrange
    figure_2pt15 = DisplaySkill(
        cmap=ListedColormap(
            [
                "white",
            ]
        ),
        color_skills=color_skills,
        skills_key=skills_key,
    )
    # Act
    answers = figure_2pt15._pre_process(input).values
    # Assert
    np.testing.assert_equal(expected, answers)


@pytest.mark.parametrize(
    "color_skills,skills_key,second_skill_color,expected",
    [
        (
            {"0": 2, "1": 3, "0,1,2": 4},
            [[0], [0], [0, 1, 2]],
            {1: "magenta", 2: "orange"},
            ValueError,
        ),
        ({"0": 2, "0,6": 9}, [[1], [2]], None, ValueError),
    ],
)
def test_DisplaySkill__init__raise_expectation(
    color_skills, skills_key, second_skill_color, expected
):
    # Arrange
    # Act
    # Assert
    with pytest.raises(expected):
        _ = DisplaySkill(
            cmap=ListedColormap(
                [
                    "white",
                ]
            ),
            color_skills=color_skills,
            skills_key=skills_key,
            second_skill_color=second_skill_color,
        )


@pytest.mark.parametrize(
    "responses,color_skills,skills_key,second_skill_color,expected",
    [
        (
            pd.DataFrame(
                np.array(
                    [[1, 1, 1, 1, 1], [0, 1, 1, 0, 1], [0, 1, 0, 0, 0]], dtype="int32"
                )
            ),
            {"0": 2, "2": 3, "0,1": 4, "1,2": 5},
            [[0], [0], [0, 1], [2]],
            {1: "magenta", 2: "orange"},
            AssertionError,
        ),
        (
            pd.DataFrame(
                np.array(
                    [[1, 1, 1, 1, 1], [0, 1, 1, 0, 1], [0, 1, 0, 0, 0]], dtype="bool"
                )
            ),
            {"0": 2, "2": 3, "0,1": 4, "1,2": 5},
            [[0], [0], [0, 1], [2], [1, 2]],
            {1: "magenta", 2: "orange"},
            AssertionError,
        ),
        (
            pd.DataFrame(
                np.array(
                    [[1, 1, 1, 1, 1], [0, 1, 1, 0, 1], [0, 1, 0, 0, 0]], dtype="int32"
                ),
                columns=["Q{}".format(i) for i in range(5)],
            ),
            {"0": 2, "2": 3, "0,1": 4, "1,2": 5},
            [[0], [0], [0, 1], [2], [1, 2]],
            {1: "magenta", 2: "orange"},
            AssertionError,
        ),
        (
            pd.DataFrame(
                np.array(
                    [[1, 1, 1, 1, 1], [0, 1, 1, 0, 1], [0, 1, 0, 0, 0]], dtype="int32"
                ),
                index=range(1, 4),
            ),
            {"0": 2, "2": 3, "0,1": 4, "1,2": 5},
            [[0], [0], [0, 1], [2], [1, 2]],
            {1: "magenta", 2: "orange"},
            AssertionError,
        ),
    ],
)
def test_plot_raise_expectation(
    responses, color_skills, skills_key, second_skill_color, expected
):
    # Arrange
    figure_2pt15 = DisplaySkill(
        cmap=ListedColormap(
            [
                "white",
            ]
        ),
        color_skills=color_skills,
        skills_key=skills_key,
        second_skill_color=second_skill_color,
    )
    # Act
    # Assert
    with pytest.raises(expected):
        figure_2pt15.plot(responses)


@pytest.mark.mpl_image_compare
def test_plot_figure_2pt15():
    # Arrange
    color_skills = {str(i - 2): i for i in range(2, 8)}
    color_skills["1,6"] = 8
    color_skills["0,6"] = 9
    color_skills["1,4"] = 10

    second_skill = {6: "magenta", 4: "orange"}

    figure_2pt15 = DisplaySkill(
        cmap=ListedColormap(
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
        color_skills=color_skills,
        skills_key=chapter_02_data.skills_needed,
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
        return_fig_ax=True,
    )
    # Act
    fig, _ = figure_2pt15.plot(chapter_02_data.responses.T)
    # Assert
    return fig
