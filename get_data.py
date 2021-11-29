from odps import ODPS
import pandas as pd
from tqdm import tqdm
from multiprocessing import Process
import gc


o = ODPS(access_id='*******',
         secret_access_key='**********',
         project='*****',
         endpoint='http://service.cn-shanghai.maxcompute.aliyun.com/api')


def get_data(table_name, num, total_step, step):
    table = o.get_table(table_name)
    process_list = list()
    field = table.schema.names
    with table.open_reader() as records:
        for i in range(num):
            p = Process(target=write_data, args=(i, total_step, step, records, field))
            p.start()
            process_list.append(p)

        for p in process_list:
            p.join()


def write_data(num, total_step, step, records, field):
    step_list = [i for i in range(num * total_step, (num + 1) * total_step, step)]
    for i in step_list:
        data_list = list()
        for record in tqdm(records[i: i + step]):
            lt = list()
            for j in range(len(field)):
                lt.append(record[j])
            data_list.append(lt)
        if len(data_list) == 0:
            break
        data = pd.DataFrame(data_list, columns=field)
        data.drop(data[pd.isnull(data['city_id'])].index, inplace=True)
        if 'ds' in data.columns:
            data.drop('ds', axis=1, inplace=True)
        data.to_csv('./dataset/rank_train_data.tsv', mode='a', sep='\t', index=False, header=False)
        del data
        del data_list
        gc.collect()


if __name__ == '__main__':
    get_data('pai_temp_201057_2026115_1', 5, 10000000, 1000000)




