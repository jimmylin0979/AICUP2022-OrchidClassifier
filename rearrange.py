import pandas as pd
import argparse

if __name__ == '__main__':

    #
    parser = argparse.ArgumentParser(description='AICUP - Orchid Classifier')
    parser.add_argument('--file', default='prediction.csv', type=str, required=True,
                                help='The name of the prediction file')
    args = parser.parse_args()

    # Step 5 : Rearrange
    # Rearrange the predictions into a dataframe.
    # output_file_path = 'prediction_model_swin_mixs_tmax101.csv'
    output_file_path = args.file
    df_predict = pd.read_csv(output_file_path)

    # 
    df_sample = pd.read_csv('./submission_template.csv')

    # 
    df_arrange = pd.merge(df_sample, df_predict, how='inner', on=['filename'])
    df_arrange = df_arrange.drop(['category_sample'], axis=1)
    df_arrange.to_csv(f'rearrange_{output_file_path}', index=False)
