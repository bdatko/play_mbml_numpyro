"""
Data for Chapter 2
"""
import pandas as pd

# data code is form the post from the pyro forum
# https://forum.pyro.ai/t/model-based-machine-learning-book-chapter-2-skills-example-in-pyro-tensor-dimension-issue/464/3?u=bdatko

# Get data from Book's Website
raw_data = pd.read_csv(
    "http://www.mbmlbook.com/Downloads/LearningSkills_Real_Data_Experiments-Original-Inputs-RawResponsesAsDictionary.csv"
)
self_assessed = raw_data.iloc[1:, 1:8]
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

chapter_02_data = {
    "raw_data": raw_data,
    "self_assessed": self_assessed,
    "skills_key": skills_key,
    "skills_needed": skills_needed,
    "responses": responses,
}
