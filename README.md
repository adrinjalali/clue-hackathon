# clue-hackathon
This repository is what we used at Clue-WATTx hackathon, which won the best performing method.

- Website: http://cluehackathon.wattx.io/
- Original example submission repository (visible to attendees only): https://github.com/WATTx/cluehackathon

The submission and starting point of train scripts are copied from the above repository.

__IMPORTANT__: Do NOT add the sample or the synthesized data to this repository or any clones.
The data is the property of the Clue company (BioWink GmbH), and I believe we do not have
the permission to publish it publicly. It is safe to copy the data under a "data" folder in a locally cloned repo,
since it is included in the `.gitignore` file and won't automatically be added to the repo.
- Some gitignore documentation: https://help.github.com/articles/ignoring-files/ 

## Installation
You can run the code using a python virtual environment (http://docs.python-guide.org/en/latest/dev/virtualenvs/).


    virtualenv -p python3 venv
    source venv/bin/activate
    pip install -r requirements.txt
    ./run.sh


## Data Description
The dataset includes information about users, their cycles, and recorded symptoms within each cycle.
The data includes the following sections:
 - Users: Demographic information about users, including age, country, etc.
 - Active Days: consists of (user id, date, cycle id, day in cycle) tuples, showing days in which the user has had some interaction with the app.
 - Cycles: includes information about cycles, such as starting date, length, and cycle id
 - Tracking: this is the largest part of the dataset, including one row per (user id, cycle id, day in cycle, category, symptom).

There are almost 80 symotoms, which are in groups of four, each group called a category. In this analysis category is completely ignored, since symptom names are unique.

We approach the task as a regression problem, and not a classification problem. Then we take the predicted values as _probabilities_ and export them as output. One supporting argument is that the recorded symptoms are simply there when the symptom as passed a certain threshold, and the symptoms themselves behave in a continues way and are not discrete phenomena.

## Data Preparation
The data is transformed such that there is one row per user at the end. The tracking data is converted into columns, each column name being a (symptom, day in cycle) tuple. To better map cycles and day in cycles to eachother, all sycles are scaled into a 29 day cycle, which is the median cycle length. This is done by this transformation:

    day in cycle => (day in cycle / cycle length) * 29

and the cycle length is fetched from the `cycles` table. This results in a matrix which has rows as the number of users, and over 2500 (number of symptoms times 29) columns. At the end we attach the median number of active days per cycle for each user, to this matrix.

### `X` and `Y`
After maping all the cycles to the same length, we aggregate the data on cycle id, and count the number of times a user has reported a symptom on that each day of the cycle. This results in a `NaN` for days in the cycle for which the user has not ever reported the respective symptom, and an integer otherwise, reflecting the number of times the user has recorded that symptom on that day of the cycle. All the `NaN` values are later replaced by zeros.

Before creating the matrix, we exclude the last cycle of each user from the data, which gives us our input matrix, or `X`. And the data representing the last cycle results in a similar matrix, which we use as the output, or `Y`. Please note that the output mostly includes only ones and zeros, since the user has recorded a symptom on that day of the last cycle or not, unless the user has interacted with the app about a symptom more than once for the same recorded day, or that after scaling the cycle, two days are mapped into the same day.

The respective code for data preparation is done by `process_level2(.)` function in [pre_process.py](src/pre_process.py) file.

On top, we added several information from the users which could actually affect the occurency of a specific symptom.
With the function preprocess_users we have created the following demographic and cycle-specific variables:

- 'continent': which divides different countries to their original continent to identify differences
- 'first_havers': defined as females which just started to have their cycle with irregular periods
- 'menopause': a variable which takes into account women which will probably have soon menopause
- 'bmi': an index which is defined as weight divided by the square of height.
- 'age': three dummies for age, to identify different behaviours in using the app --> users <18, users 18-24 and users>24

These information are useful to the model to improve the predictability of the symptoms.

## Methods
Once the data is preprocessed and we have our `X` and `Y` matrices, we train a model for each symptom. This means the model takes the `X` matrix as input, and gives an output having dimensions `(n_users, 29)`.
If the model doesn't support a vector as an output, we need to train 29 separate models per symptom, and due to the computational cost of doing so we avoid such models.

The model is in this case a `scikit-learn` pipeline:

    pipeline = Pipeline([
        ('remove_low_variance_features', VarianceThreshold(threshold=0.0)),
        ('standard_scale', StandardScaler()),
        ('estimator', Lasso()),
    ])

    param_grid = {'estimator__alpha': [.1, .3, .5, .7, .8]}
    model = GridSearchCV(pipeline, param_grid = param_grid, n_jobs = 4,
                         verbose=2)
    model.fit(X, Y)

`Y` being the output for only one symptom. The actual code is found in [pipeline.py](src/pipeline.py) file. The `pipeline` can be changed easily to set the desired preprocessing and/or prediction model.

**Clustering:**
We believe that there is a great potential in doing clusters of different typologies of users.
In order to perform a cluster of users different variables have been taken into considerations:
there are two types of variables that have been used to cluster users, on one hand we focused on demographics, on the other hand on previous behaviours and attitude of using the app.
Given the fact that the PCA method uses continous variables, we built variables that could apply well to such task. The process can be found in src/clustering/preprocess_users and in users_features.
The demographic variables used are:
- 'bmi' (explained previously)
- 'age'

For as it concerns the modality of usage of the app, we built some variables which happens to be valuable for the analysis to extract patterns of utilisation:
- 'monthly activity': the number of interactions (symptom clicked) averaged per month
- 'relative activity': a percentage of the days that a user is normally active during the menstruation cycles (100% means that the user is active every day)
- 'menstrual_activity': a percentage which takes the interaction during the first 5 days of a woman and it divides out of the total interactions. This is very important to cluster people who just access the app for the menstruation from users which are constantly active.
- 'last_month_activity': similar as monthly activity, but it just look at the last month, to identify changing patterns of the user

Our clustering performance can be found inside clusters.py where we applied clustering by using a K-menans method on the two first principal components. 
The methodology is the following: use PCA analysis on those variable to identify group of people which behave similarly and then use K-means to extract information.
More analysis and comments (with interesting graphs) on this point can be found inside clusters+access_probability.pdf (in the main folder).
We tried then to do clustering the users beforehand to train a model for each cluster.

Inside the pdf file, there is also an interesting analysis: we tried to distinguish between the probability of access of the app of a user and the probability of clicking a symptom, given the fact that the user opened the app. More details and information on this analysis can be found in the file.

## Results
We compared and submitted a few models in our evaluations. At the end, the best performing model was a `LASSO` and a grid search on its `alpha` parameter. Other evaluated models include:
 - Linear Regression
 - Decision Trees with a grid search on its depth and max number of used input features
 - Random Forests
 - Support Vector Machines with an RBF kernel and grid search on the `gamma` parameter of the kernel and the `C` of the optimization problem (computation was too expensive and never finished in time)

## Potential Future Work
One important question dealing with this type of data is weather to train one single model for all the users, or a single model per user. Arguably clustering the users and training a model per cluster can be considered a middle ground.
We decided against training a model per user for two reasons:
- computationally intensive
- not enough data per user

An alternative approach is to train a general model for all the users, and then do further fitting for each given user. This uses the knowledge we have from the population, and then fits the model a bit further to the user. Some possible ways to do so are:
- Train a Gaussian process on the whole data, then further train the GP (for only a few iterations if it's an iterative method) given only the data from the user. One important detail to consider is that GPs are usually prune to overfitting unless they are sparse, or the covariance function imposes sparsity, or a feature selection is done prior to the GP fit.
- `scikit-learn` supports incremental fit for some models. The same strategy as explained above can be taken using these models. More information about `partial_fit` and the models can be found [here](http://scikit-learn.org/stable/modules/scaling_strategies.html)

There are also a few tasks that we could do on the preprocessing side of the code:
- The code in this repository assumes `1` whenever a symptom is tracked, and `0` otherwise. Then we sum over cycles of the user to calculate the input. Instead of taking the sum, we could take the average or the median of those values and see if it improves the performance. The downside of what we do is that users with more recorded cycles have higher values in their feature vector solely because they've been on the app longer, and not necessarily because they've been more active.
  - UPDATE: this did not improve the performance
- At the moment we scale the whole cycle linearly into a 29 day cycle. Whilst it might make sense for some of the symptoms, other symptoms might show a different behavior. For instance, it might be better to keep the last 2 weeks of the cycle as it is, and scale the rest of it linearly into 15 days. There is some evidence that this is a better transformation for many symptoms. One idea is to transform all the symptoms using both explained approaches, and then for each symptom, test which transformation makes the population closer to each other. The transformation resulting in less variance among the population is probably a better fit.
- The `StandardScaler` used in our code includes all the zeros in the estimated mean and variance before transformation, which in such a sparse data as we have in this task is wrong. As long as the prediction model does not assume having normally distributed input variables, this preprocessing step can be removed.
  - UPDATE: this did not improve the performance
- compare a constant small output to other models
  - UPDATE: this was implemented, but got failed and never got debugged (competition over). It is the `DummyModel` in (dump_results.py)[src/dump_results.py] file.

A final remark: inside [clusters+access_probability.pdf](clusters+access_probability.pdf) in the last part we tried to recreate profiles of click of a specific symptom given the fact that the user accessed the app. Such analysis it is really valuable as it shows the real behaviour of the sympthoms over time, and it can become especially powerful whenever users are clustered in different groups.
