import google.auth
import pandas_gbq


def read_bigquery(table_name):
    credentials, project_id = google.auth.default()
    df = pandas_gbq.read_gbq('select * from `footy-323809.statistics.{}`'.format(table_name),
                             project_id=project_id,
                             credentials=credentials,
                             location='europe-west1')

    return df


def write(df, project_id, output_dataset_id, output_table_name, credentials, if_exists="replace"):
    print("write to bigquery")
    df.to_gbq(
        "{}.{}".format(output_dataset_id, output_table_name),
        project_id=project_id,
        if_exists=if_exists,
        credentials=credentials,
        progress_bar=None
    )
    print("Query complete. The table is updated.")
