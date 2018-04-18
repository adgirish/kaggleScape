
# coding: utf-8

# This kernel creates mapping between click_id  from current test.csv and click_id from test.csv uploaded initially (and re-uploaded officially as test_supplement.csv), which has complete data for time range 2017-Nov-10 12:00 - 2017-Nov-10 23:00 (Chinese local time).
# 
# The mapping can be useful to predict on test_supplement.csv data directly.
# 
# Old test data taken from [here](https://www.kaggle.com/tkm2261/old-test-data-on-talkingdata-adtracking), thanks tkm2261.

# In[ ]:



old_file_path = '../input/old-test-data-on-talkingdata-adtracking/test.csv'
file_path = '../input/talkingdata-adtracking-fraud-detection/test.csv'
output_file_path = 'mapping.csv'


def _split(line):
    line = line.strip()
    index = line.index(',')
    last_index = line.rindex(',')
    click_id = line[:index]
    payload = line[index:]
    time = line[last_index:]
    return click_id, payload, time


def _read_same_time(lines, unprocessed_line):
    click_id, payload, group_time = _split(unprocessed_line)
    click_id_dict = {payload: [click_id]}
    while True:
        unprocessed_line = lines.readline()
        if not unprocessed_line:
            return unprocessed_line, click_id_dict, group_time
        click_id, payload, click_time = _split(unprocessed_line)
        if group_time == click_time:
            if payload in click_id_dict:
                click_id_dict[payload].append(click_id)
            else:
                click_id_dict[payload] = [click_id]
        else:
            return unprocessed_line, click_id_dict, group_time


def _find_time(lines, group_time, unprocessed_line):
    if unprocessed_line:
        click_id, payload, time = _split(unprocessed_line)
        if group_time == time:
            return unprocessed_line
    while True:
        unprocessed_line = lines.readline()
        click_id, payload, time = _split(unprocessed_line)
        if group_time == time:
            return unprocessed_line


def _save(output, test_click_id_dict, old_test_click_id_dict):
    for payload, click_ids in test_click_id_dict.items():
        old_click_ids = old_test_click_id_dict[payload]
        if len(old_click_ids) != len(click_ids):
            print('Number of ids mismatch for "{}", test ids = {}, old test ids = {}'.format(payload, click_ids,
                                                                                             old_click_ids))
        for i in range(len(click_ids)):
            output.write('{},{}\n'.format(click_ids[i], old_click_ids[i]))


with open(file_path, "r", encoding="utf-8") as test:
    with open(old_file_path, "r", encoding="utf-8") as old_test:
        with open(output_file_path, "w", encoding="utf-8") as output:
            output.write('click_id,old_click_id\n')
            test.readline()  # skip header
            old_test.readline()  # skip header
            old_test_unprocessed_line = old_test.readline()
            test_unprocessed_line = test.readline()
            while test_unprocessed_line != '':
                test_unprocessed_line, test_click_id_dict, click_time = _read_same_time(test, test_unprocessed_line)
                old_test_unprocessed_line = _find_time(old_test, click_time, old_test_unprocessed_line)
                old_test_unprocessed_line, old_test_click_id_dict, _ = _read_same_time(old_test,
                                                                                       old_test_unprocessed_line)
                _save(output, test_click_id_dict, old_test_click_id_dict)
        pass


# In[1]:


import pandas as pd
mapping = pd.read_csv('../input/mapping-between-test-supplement-csv-and-test-csv/mapping.csv', dtype={'click_id': 'int32','old_click_id': 'int32'}, engine='c',
                na_filter=False,memory_map=True)


# In[ ]:


print('click id min {}'.format(mapping.click_id.min()))
print('click id max {}'.format(mapping.click_id.max()))
print('click id count {}'.format(mapping.click_id.count()))
print('click id unique count {}'.format(mapping.click_id.unique().shape[0]))


# In[ ]:


print('old click id min {}'.format(mapping.old_click_id.min()))
print('old click id max {}'.format(mapping.old_click_id.max()))
print('old click id count {}'.format(mapping.old_click_id.count()))
print('old click id unique count {}'.format(mapping.old_click_id.unique().shape[0]))

