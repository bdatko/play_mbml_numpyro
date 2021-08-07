"""
Data for Chapter 2
"""
from dataclasses import dataclass, field
from typing import List

import pandas as pd


@dataclass(frozen=True, order=True)
class ChapterData02:
    """
    DataClass for Chapter 02 data

    :param raw_data pd.DataFrame: raw data
        `raw_data.shape`: (23,56)
            columns[0]: indicator, dtype == object
            columns[1:8]: self assessed skills from each participants, dtype == object
            columns[8:]: respones for each each question, dtype == int64
        `raw_data.iloc[0,8:]` answers to the test
    :param self_assessed pd.DataFrame: self assessed skills from each participants
        `self_assessed` == `raw_data.iloc[1:, 1:8]`
        `self_assessed.shape`: (22,7)
            columns: self assessed skills from each participants, dtype == bool
    :param skills_key pd.DataFrame: skills nesscary for each question as boolean indicator, where columns are the skills and rows are the test questions
        `skils_key.shape`: (48,7)
            columns: n_skills
            rows: n_questions
    :params skills_needed List[List[int]]: sparse version of skills_key
    :params responses pd.DataFrame: Graded responses, where columns are participants and rows are their grades prespones
        `responses.dtypes`: int32, because of ploting
        `responses.shape`: (48,22)
            columns: n_participants
            rows: n_questions
    """

    raw_data: pd.DataFrame
    self_assessed: pd.DataFrame
    skills_key: pd.DataFrame
    skills_needed: List[List[int]]
    responses: pd.DataFrame


# data code is form the post from the pyro forum
# https://forum.pyro.ai/t/model-based-machine-learning-book-chapter-2-skills-example-in-pyro-tensor-dimension-issue/464/3?u=bdatko

# Get data from Book's Website
raw_data = pd.read_csv(
    "http://www.mbmlbook.com/Downloads/LearningSkills_Real_Data_Experiments-Original-Inputs-RawResponsesAsDictionary.csv"
)
self_assessed = raw_data.iloc[1:, 1:8].copy()
self_assessed = self_assessed.astype(bool)

skills_key = pd.read_csv(
    "http://www.mbmlbook.com/Downloads/LearningSkills_Real_Data_Experiments-Original-Inputs-Quiz-SkillsQuestionsMask.csv",
    header=None,
)
skills_needed = []
for index, row in skills_key.iterrows():
    skills_needed.append([i for i, x in enumerate(row) if x])

# col = person, row = question
responses = pd.read_csv(
    "http://www.mbmlbook.com/Downloads/LearningSkills_Real_Data_Experiments-Original-Inputs-IsCorrect.csv",
    header=None,
)

responses = responses.astype("int32")

chapter_02_data = ChapterData02(
    raw_data, self_assessed, skills_key, skills_needed, responses
)
