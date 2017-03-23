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

## Methods
Once the data is pre-processed and we have our `X` and `Y` matrices, we train a model for each symptom. This means the model takes the `X` matrix as input, and gives an output having dimensions `(n_users, 29)`.
If the model doesn't support a vector as an output, we need to train 29 separate models per symptom, and due to the computational cost of doing so we avoid such models.

The model is in this case a `scikit-learn` pipeline:

    pipeline = Pipeline([
        ('remove_low_variance_features', VarianceThreshold(threshold=0.0)),
        ('standard_scale', StandardScaler()),
        ('estimator', DecisionTreeRegressor(max_depth=5)),
    ])

    pipeline.fit(X, Y)

`Y` being the output for only one symptom. The actual code is found in [pipeline.py](src/pipeline.py) file.

## Results
We compared and submitted a few models in our evaluations. At the end, the best performing model was a `LASSO` and a grid search on its `alpha` parameter. Other evaluated models include:
 - Linear Regression
 - Decision Trees with a grid search on its depth and max number of used input features
 - Random Forests
 - Support Vector Machines with an RBF kernel and grid search on the `gamma` parameter of the kernel and the `C` of the optimization problem (computation was too expensive and never finished in time)

## Potential Future Work
- GP iterative learning
- scikit-learn linear model partial learning
- modified pre-processing (avg instead of sum)
- different pre-processing per symptom
- standard scaler doesn't work for sparse data
- user clusters
- compare to all `0.01` or similar constant output performance
