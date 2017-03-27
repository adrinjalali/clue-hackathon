import sys
import pandas as pd


def run(data_fname):

    # load data
    df = pd.read_csv(data_fname)

    # extract symptoms to predict
    symptoms = ['happy', 'pms', 'sad', 'sensitive_emotion', 'energized', 'exhausted',
                'high_energy', 'low_energy', 'cramps', 'headache', 'ovulation_pain',
                'tender_breasts', 'acne_skin', 'good_skin', 'oily_skin', 'dry_skin']
    df = df[df.symptom.isin(symptoms)]

    # predictions are simply the averages per user/day_of_cycle/symptom of the training data
    user_num_cycles = df[['user_id', 'cycle_id']].groupby(['user_id']).max()
    user_symptoms_total = df.groupby(['user_id', 'symptom', 'day_in_cycle']).count()[['cycle_id']]
    user_symptoms_rel = user_symptoms_total / user_num_cycles

    # create results
    user_symptoms_rel.reset_index(inplace=True)
    user_symptoms_rel = user_symptoms_rel.rename(columns={'cycle_id': 'probability'})
    results = user_symptoms_rel[['user_id', 'day_in_cycle', 'symptom', 'probability']]

    # save results in the correct format.
    results.to_csv('../result.txt', index=None)


if __name__ == '__main__':
    print('run training')
    data_fname = sys.argv[-1]
    run(data_fname)
