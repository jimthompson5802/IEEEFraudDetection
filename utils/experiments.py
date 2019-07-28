#
# utilities to extract experiment data
#

import pandas as pd
import mlflow
import mlflow.tracking


def extract_run_data_for_experiment(experiment_name):
    """
    Extract run data for specified experiment

    :param experiment_name: experiment name to extract run data
    :return: datafraame of run data for the specified experiment
    """

    def extract_run_data(run_info):
        """
        extracts run data for a specific run

        :param run_info: mlflow.entities.RunInfo
        :return: single data frame row with experiment run data
        """

        # get run specific data
        run = client.get_run(run_info.run_id)

        # extract run meta data
        p1 = pd.DataFrame(run.to_dictionary()['info'],
                          index=[0])[['experiment_id', 'run_id',
                                      'lifecycle_stage', 'start_time',
                                      'end_time', 'artifact_uri']]

        # Extract run tag data
        p2 = pd.DataFrame(run.to_dictionary()['data']['tags'], index=[0])

        # combine to create data frame row
        df_row = pd.concat([p1, p2], axis=1)

        return df_row

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    run_info_list = client.list_run_infos(experiment.experiment_id)

    experiment_runs = pd.concat([extract_run_data(run_info) for run_info in run_info_list],
                                axis=0, ignore_index=True)

    return experiment_runs