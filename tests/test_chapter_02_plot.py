import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import ListedColormap
from mbml_numpyro import DisplaySkill, chapter_02_data


@pytest.mark.parametrize(
    "expected", [(np.array([[1, 1, 1], [2, 1, 1], [2, 1, 9]], dtype="int32"))]
)
def test_figure_2pt15(expected):
    # Arrange
    color_skills = {str(i - 2): i for i in range(2, 8)}
    color_skills["1,6"] = 8
    color_skills["0,6"] = 9
    color_skills["1,4"] = 10
    color_skills[6] = "magenta"
    color_skills[4] = "orange"

    temps = (
        chapter_02_data["raw_data"].loc[1:, "Q1":"Q48"]
        == chapter_02_data["raw_data"].loc[0, "Q1":"Q48"]
    ).copy()
    temps = temps.astype("int32")

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
        chapter_02_data["skills_needed"],
        color_skills,
    )
    # Act
    answers = figure_2pt15._pre_process(
        pd.DataFrame(temps.values).iloc[: expected.shape[0], : expected.shape[1]]
    ).values
    # Assert
    np.testing.assert_equal(expected, answers)
