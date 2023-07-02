# Model-Based Machine Learning with NumPyro

I enjoy the book [Model-Based Machine Learning](https://mbmlbook.com/index.html), and inspired by [Du Phan's port of Statistical Rethinking](https://github.com/fehiepsi/rethinking-numpyro) I decided to play around to port some examples of the code to NumPyro.

# Notes
Pyro MBML Has anyone already done this?
- [MBML source pdf](https://mbmlbook.com/MBMLbook.pdf)
- [MBML early access](https://mbmlbook.com/index.html)

<span style="color:red">**Q:**</span> Are there any posts on the [Pyro forums](https://forum.pyro.ai/) about Model-Based Machine Learning?

<span style="color:blue">**A:**</span>
Searching the forums for two keyword phrases finds a total of three posts.

1. [*"mbml"*](https://forum.pyro.ai/search?q=%22mbml%22)
     - [Jan 3, '19 5:32 PM - Importance sampling and Empirical Margin](https://forum.pyro.ai/t/importance-sampling-and-empirical-margin/627)
     - [Oct 24, '18 2:28 PM - Model Based Machine Learning Book Chapter 2 Skills example in Pyro- tensor dimension issue](https://forum.pyro.ai/t/model-based-machine-learning-book-chapter-2-skills-example-in-pyro-tensor-dimension-issue/464)
2. [*"model based machine learning"*](https://forum.pyro.ai/search?q=%22model%20based%20machine%20learning%22)
     - [Feb 17, '19 9:49 AM - Model Based Machine Learning chapter 3, loss oscillating](https://forum.pyro.ai/t/model-based-machine-learning-chapter-3-loss-oscillating/738)
     - Oct 24, '18 2:28 PM - Model Based Machine Learning Book Chapter 2 Skills example in Pyro- tensor dimension issue

### Oct 24, '18 2:28 PM - Model Based Machine Learning Book Chapter 2 Skills example in Pyro- tensor dimension issue
* The post is very long and I think the original code from OG is not code to emulate, But the post did bring out other authors to contribute to cleaning up the code
* Need to be careful since pyro was a lot younger and some of the bugs or code might be work aronds
* [fritzo code](https://forum.pyro.ai/t/model-based-machine-learning-book-chapter-2-skills-example-in-pyro-tensor-dimension-issue/464/12) seems to be something to target
* The post also has some discussion about [tensor shape tutorial notebook](https://github.com/pyro-ppl/pyro/blob/dev/tutorial/source/tensor_shapes.ipynb), `to_event()`, `event_shape` see also [this post titled Dependency tracking in Pyro](https://forum.pyro.ai/t/dependency-tracking-in-pyro/500)
* Even though the code snippets are posted there are no repos shared

### Jan 3, '19 5:32 PM - Importance sampling and Empirical Margin](https://forum.pyro.ai/t/importance-sampling-and-empirical-margin/627
* post is more focused, but I think suffers from the earlier days of pyro
* the OG was able to make progress and posted his [Github for code](https://github.com/MicPie/pyro)
* [OG's repo](https://github.com/MicPie/pyro) has some great visualization for both chapters [1](https://github.com/MicPie/pyro/blob/master/MBML_Chapter1_MurderMystery.ipynb) and [2](https://github.com/MicPie/pyro/blob/master/MBML_Chapter2_PeoplesSkills.ipynb)
* OG links to [slides titled Introduction to Probabilistic Programming: Models and Inference in Pyro](https://mltrain.cc/wp-content/uploads/2018/02/MLTrain@UAI_-Introduction-to-Pyro.pdf) which he claims helped to get the program to work
* The slides are from a workshop titled [MLTrain @UAI 2018](https://mltrain.cc/events/mltrain-uai-2018/) with videos for the presentations on [YouTube](https://www.youtube.com/watch?v=f3GGwt4FD-g&list=PLqDaBXsXAF8px54HwZk8dWUfzfhYTrPDH).
* These slides and presentation are also linked to from the [post title Dependency tracking in Pyro](https://forum.pyro.ai/t/dependency-tracking-in-pyro/500/8)

### Feb 17, '19 9:49 AM - Model Based Machine Learning chapter 3, loss oscillating](https://forum.pyro.ai/t/model-based-machine-learning-chapter-3-loss-oscillating/738
* Has code snippets for chapter 3
* community seems to show progress with SVI but recommends MCMC

## Community members List
* [erlebach](https://forum.pyro.ai/u/erlebach)
* [Rubiel](https://forum.pyro.ai/u/rubiel/summary)
* [jeffmax](https://forum.pyro.ai/u/jeffmax)
* [MicPie](https://forum.pyro.ai/u/MicPie)

<span style="color:red">**Q:**</span> Are there any Github repositories outside the Pyro community working on a similar project?

<span style="color:blue">**A:**</span>
Searching via DuckDuckGo and Google I can find one dedicated repository and a mention to a course that teaches out of the MBML book.

1. DuckDuckGo search: github MBML "pyro" model based machine learning
     - [mengqvist/data_analysis_mbml](https://github.com/mengqvist/data_analysis_mbml)
2. Google search github MBML "pyro" model based machine learning:
     - [jmontalvo94/mbml-energy-predictor](https://github.com/jmontalvo94/mbml-energy-predictor)

### mengqvist/data_analysis_mbml
* The repo has all the data downloaded but not many examples finished
* There is a single [notebook](https://github.com/mengqvist/data_analysis_mbml/blob/master/MBML_book.ipynb) containing Chapter 1 and Chapter 2
* Looks like the author couldn't get past Chapter 2 *Moving to real data*

### jmontalvo94/mbml-energy-predictor
* Mentions the course *42186 - Model-based Machine Learning* from the Technical University of Denmark (DTU)
* I can find several courses with the MBML title searching [Technical University of Denmark (DTU)](https://www.dtu.dk/english/resultat?qt=NetmesterSearch&fr=1&sw=42186%20-%20Model-based%20Machine%20Learning#tabs)
* [Course page for 42186 - Model-based Machine Learning from the Technical University of Denmark (DTU)](https://kurser.dtu.dk/course/42186)
- [MBML source pdf](https://mbmlbook.com/MBMLbook.pdf)
- [MBML early access](https://mbmlbook.com/index.html)

<span style="color:red">**Q:**</span> Are there any posts on the [Pyro forums](https://forum.pyro.ai/) about Model-Based Machine Learning?

<span style="color:blue">**A:**</span>
Searching the forums for two keyword phrases finds a total of three posts.

1. [*"mbml"*](https://forum.pyro.ai/search?q=%22mbml%22)
     - [Jan 3, '19 5:32 PM - Importance sampling and Empirical Margin](https://forum.pyro.ai/t/importance-sampling-and-empirical-margin/627)
     - [Oct 24, '18 2:28 PM - Model Based Machine Learning Book Chapter 2 Skills example in Pyro- tensor dimension issue](https://forum.pyro.ai/t/model-based-machine-learning-book-chapter-2-skills-example-in-pyro-tensor-dimension-issue/464)
2. [*"model based machine learning"*](https://forum.pyro.ai/search?q=%22model%20based%20machine%20learning%22)
     - [Feb 17, '19 9:49 AM - Model Based Machine Learning chapter 3, loss oscillating](https://forum.pyro.ai/t/model-based-machine-learning-chapter-3-loss-oscillating/738)
     - Oct 24, '18 2:28 PM - Model Based Machine Learning Book Chapter 2 Skills example in Pyro- tensor dimension issue

### Oct 24, '18 2:28 PM - Model Based Machine Learning Book Chapter 2 Skills example in Pyro- tensor dimension issue
* The post is very long and I think the original code from OG is not code to emulate, But the post did bring out other authors to contribute to cleaning up the code
* Need to be careful since pyro was a lot younger and some of the bugs or code might be work aronds
* [fritzo code](https://forum.pyro.ai/t/model-based-machine-learning-book-chapter-2-skills-example-in-pyro-tensor-dimension-issue/464/12) seems to be something to target
* The post also has some discussion about [tensor shape tutorial notebook](https://github.com/pyro-ppl/pyro/blob/dev/tutorial/source/tensor_shapes.ipynb), `to_event()`, `event_shape` see also [this post titled Dependency tracking in Pyro](https://forum.pyro.ai/t/dependency-tracking-in-pyro/500)
* Even though the code snippets are posted there are no repos shared

### Jan 3, '19 5:32 PM - Importance sampling and Empirical Margin](https://forum.pyro.ai/t/importance-sampling-and-empirical-margin/627
* post is more focused, but I think suffers from the earlier days of pyro
* the OG was able to make progress and posted his [Github for code](https://github.com/MicPie/pyro)
* [OG's repo](https://github.com/MicPie/pyro) has some great visualization for both chapters [1](https://github.com/MicPie/pyro/blob/master/MBML_Chapter1_MurderMystery.ipynb) and [2](https://github.com/MicPie/pyro/blob/master/MBML_Chapter2_PeoplesSkills.ipynb)
* OG links to [slides titled Introduction to Probabilistic Programming: Models and Inference in Pyro](https://mltrain.cc/wp-content/uploads/2018/02/MLTrain@UAI_-Introduction-to-Pyro.pdf) which he claims helped to get the program to work
* The slides are from a workshop titled [MLTrain @UAI 2018](https://mltrain.cc/events/mltrain-uai-2018/) with videos for the presentations on [YouTube](https://www.youtube.com/watch?v=f3GGwt4FD-g&list=PLqDaBXsXAF8px54HwZk8dWUfzfhYTrPDH).
* These slides and presentation are also linked to from the [post title Dependency tracking in Pyro](https://forum.pyro.ai/t/dependency-tracking-in-pyro/500/8)

### Feb 17, '19 9:49 AM - Model Based Machine Learning chapter 3, loss oscillating](https://forum.pyro.ai/t/model-based-machine-learning-chapter-3-loss-oscillating/738
* Has code snippets for chapter 3
* community seems to show progress with SVI but recommends MCMC

## Community members List
* [erlebach](https://forum.pyro.ai/u/erlebach)
* [Rubiel](https://forum.pyro.ai/u/rubiel/summary)
* [jeffmax](https://forum.pyro.ai/u/jeffmax)
* [MicPie](https://forum.pyro.ai/u/MicPie)

<span style="color:red">**Q:**</span> Are there any Github repositories outside the Pyro community working on a similar project?

<span style="color:blue">**A:**</span>
Searching via DuckDuckGo and Google I can find one dedicated repository and a mention to a course that teaches out of the MBML book.

1. DuckDuckGo search: github MBML "pyro" model based machine learning
     - [mengqvist/data_analysis_mbml](https://github.com/mengqvist/data_analysis_mbml)
2. Google search github MBML "pyro" model based machine learning:
     - [jmontalvo94/mbml-energy-predictor](https://github.com/jmontalvo94/mbml-energy-predictor)

### mengqvist/data_analysis_mbml
* The repo has all the data downloaded but not many examples finished
* There is a single [notebook](https://github.com/mengqvist/data_analysis_mbml/blob/master/MBML_book.ipynb) containing Chapter 1 and Chapter 2
* Looks like the author couldn't get past Chapter 2 *Moving to real data* ... *I know how you feel* 😭

### jmontalvo94/mbml-energy-predictor
* Mentions the course *42186 - Model-based Machine Learning* from the Technical University of Denmark (DTU)
* I can find several courses with the MBML title searching [Technical University of Denmark (DTU)](https://www.dtu.dk/english/resultat?qt=NetmesterSearch&fr=1&sw=42186%20-%20Model-based%20Machine%20Learning#tabs)
* [Course page for 42186 - Model-based Machine Learning from the Technical University of Denmark (DTU)](https://kurser.dtu.dk/course/42186)
