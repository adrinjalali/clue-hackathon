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

## Data Preparation
The 
## Methods
Ideas, from pre-processing to the end.

## Results
What we get out of it.