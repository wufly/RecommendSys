from odps import ODPS
import pandas as pd
from tqdm import tqdm


o = ODPS(access_id='*********',
         secret_access_key='*************',
         project='********',
         endpoint='http://service.cn-shanghai.maxcompute.aliyun.com/api')


def get_data(table_name):
    table = o.get_table(table_name)
    data_list = list()
    with table.open_reader() as records:
        for record in tqdm(records):
            lt = list()
            for i in range(len(record)):
                lt.append(record[i])
            data_list.append(lt)
    data = pd.DataFrame(data_list, columns=table.schema.names)
    data.drop(data[pd.isnull(data['city_id'])].index, inplace=True)
    data.to_csv('./dataset/' + table_name + '.tsv', index=False)


if __name__ == '__main__':
    get_data('rank_model_test_data')



