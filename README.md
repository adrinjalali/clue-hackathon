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

## Data Preparation
The data is transformed such that there is one row per user at the end. The tracking data is converted into columns, each column name being a (symptom, day in cycle) tuple. To better map cycles and day in cycles to eachother, all sycles are scaled into a 29 day cycle, which is the median cycle length. This is done by this transformation:

    day in cycle => (day in cycle / cycle length) * 29

and the cycle length is fetched from the `cycles` table. This results in a matrix which has rows as the number of users, and over 2500 (number of symptoms times 29) columns. At the end we attach the median number of active days per cycle for each user, to this matrix.

### `X` and `Y`
After maping all the cycles to the same length, we aggregate the data on cycle id, and count the number of times a user has reported a symptom on that each day of the cycle. This results in a `NaN` for days in the cycle for which the user has not ever reported the respective symptom, and an integer otherwise, reflecting the number of times the user has recorded that symptom on that day of the cycle. All the `NaN` values are later replaced by zeros.

Before creating the matrix, we exclude the last cycle of each user from the data, which gives us our input matrix, or `X`. And the data representing the last cycle results in a similar matrix, which we use as the output, or `Y`. Please note that the output mostly includes only ones and zeros, since the user has recorded a symptom on that day of the last cycle or not, unless the user has interacted with the app about a symptom more than once for the same recorded day, or that after scaling the cycle, two days are mapped into the same day.

The respective code for data preparation is done by `process_level2(.)` function in [pre_process.py](src/pre_process.py) file.

## Methods
Ideas, from pre-processing to the end.

## Results
What we get out of it.