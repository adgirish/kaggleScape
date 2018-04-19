
# coding: utf-8

# ## Below are feature engineering skills that I had been used in this competition, it improve my "Ownership" part model quite a lot
# 
# I have made some comments in this script, and hope it will help you in your next competition, if you have any question, please make comment below and I will response when I have time.

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[5]:


# create date related variable
def create_date_var(data, time):
    data[time] = pd.to_datetime(data[time])
    data['year'] = data[time].dt.year 
    data['month'] = data[time].dt.month 
    data['day'] = data[time].dt.day 
    data['dayofweek'] = data[time].dt.dayofweek 
    data['days_in_month'] = data[time].dt.days_in_month 
    data['weekofyear'] = data[time].dt.days_in_month
    data['quarter'] = data[time].dt.quarter
    
# create statistical variables
def get_stats_target(df, group_column, target_column, drop_raw_col=False):
    df_old = df.copy()
    grouped = df_old.groupby(group_column)
    the_stats = grouped[target_column].agg(['mean','median','max','min','std']).reset_index()
    
    the_stats.columns = [group_column[0], 
                       '_%s_mean_by_%s' % (target_column[0], group_column[0]),
                       '_%s_median_by_%s' % (target_column[0], group_column[0]),
                       '_%s_max_by_%s' % (target_column[0], group_column[0]),
                       '_%s_min_by_%s' % (target_column[0], group_column[0]),
                       '_%s_std_by_%s' % (target_column[0], group_column[0])]
    
    df_old = pd.merge(left=df_old, right=the_stats, on=group_column, how='left')
    if drop_raw_col:
        df_old.drop(group_column, axis=1, inplace=True)
    return df_old

# big part of feature engineering
def preprocess_data(rubbish_in, keep_is_missing=False):
    impute_missing = -1
    impute_missing_0 = 0
    impute_missing_1 = 1
    
    df_new = rubbish_in.copy()
    '''----------------------------------------------------------------------------------
    compute house count of last month, last week 
    ----------------------------------------------------------------------------------'''
    create_date_var(df_new, 'timestamp')
    df_new['timestamp_1'] = df_new.timestamp.apply(lambda x: x - pd.DateOffset(months=1))
    df_new['month_1'] = df_new['timestamp_1'].dt.month
    df_new['year_1'] = df_new['timestamp_1'].dt.year
    df_new['quarter_1'] = df_new['timestamp_1'].dt.quarter
    
    last_month_year = (df_new.timestamp_1.dt.month + df_new.timestamp_1.dt.year * 100)
    last_month_year_cnt_map = last_month_year.value_counts().to_dict()    
    df_new['_last_month_year_cnt'] = last_month_year.map(last_month_year_cnt_map)
    
    last_week_year = (df_new.timestamp_1.dt.weekofyear + df_new.timestamp_1.dt.year * 100)
    last_week_year_cnt_map = last_week_year.value_counts().to_dict()    
    df_new['_last_week_year_cnt'] = last_week_year.map(last_week_year_cnt_map)
    
    '''----------------------------------------------------------------------------------
    compute house count of this month, last week 
    ----------------------------------------------------------------------------------'''
    month_year = (df_new.timestamp.dt.month + df_new.timestamp.dt.year * 100)
    month_year_cnt_map = month_year.value_counts().to_dict()
    df_new['_month_year_cnt'] = month_year.map(month_year_cnt_map)
    
    week_year = (df_new.timestamp.dt.weekofyear + df_new.timestamp.dt.year * 100)
    week_year_cnt_map = week_year.value_counts().to_dict()
    df_new['_week_year_cnt'] = week_year.map(week_year_cnt_map)

    # drop useless variables
    df_new.drop(['timestamp_1', 'month_1', 'year_1', 'quarter_1'], axis=1, inplace=True)
    df_new['_num_of_missing'] = df_new.isnull().sum(axis=1)
    
    if keep_is_missing: 
        df_new['_missing_hospital_beds_raion'] = df_new['hospital_beds_raion'].isnull().astype(int)
    df_new['hospital_beds_raion'].fillna(impute_missing_0, inplace=True)

    if keep_is_missing: 
        df_new['_missing_cafe_500_info'] = df_new['cafe_avg_price_500'].isnull().astype(int)
    df_new['cafe_avg_price_500'].fillna(impute_missing_0, inplace=True)
    df_new['cafe_sum_500_max_price_avg'].fillna(impute_missing_0, inplace=True)
    df_new['cafe_sum_500_min_price_avg'].fillna(impute_missing_0, inplace=True)

    if keep_is_missing:
        df_new['_missing_cafe_1000_info'] = df_new['cafe_avg_price_1000'].isnull().astype(int)
    df_new['cafe_avg_price_1000'].fillna(impute_missing_0, inplace=True)
    df_new['cafe_sum_1000_max_price_avg'].fillna(impute_missing_0, inplace=True)
    df_new['cafe_sum_1000_min_price_avg'].fillna(impute_missing_0, inplace=True)

    if keep_is_missing:
        df_new['_missing_cafe_1500_info'] = df_new['cafe_avg_price_1500'].isnull().astype(int)
    df_new['cafe_avg_price_1500'].fillna(impute_missing_0, inplace=True)
    df_new['cafe_sum_1500_max_price_avg'].fillna(impute_missing_0, inplace=True)
    df_new['cafe_sum_1500_min_price_avg'].fillna(impute_missing_0, inplace=True)

    if keep_is_missing:
        df_new['_missing_cafe_2000_info'] = df_new['cafe_avg_price_2000'].isnull().astype(int)
    df_new['cafe_avg_price_2000'].fillna(impute_missing_0, inplace=True)
    df_new['cafe_sum_2000_max_price_avg'].fillna(impute_missing_0, inplace=True)
    df_new['cafe_sum_2000_min_price_avg'].fillna(impute_missing_0, inplace=True)

    if keep_is_missing:
        df_new['_missing_cafe_3000_info'] = df_new['cafe_avg_price_3000'].isnull().astype(int)
    df_new['cafe_avg_price_3000'].fillna(impute_missing_0, inplace=True)
    df_new['cafe_sum_3000_max_price_avg'].fillna(impute_missing_0, inplace=True)
    df_new['cafe_sum_3000_min_price_avg'].fillna(impute_missing_0, inplace=True)

    if keep_is_missing:
        df_new['_missing_cafe_5000_info'] = df_new['cafe_avg_price_5000'].isnull().astype(int)
    df_new['cafe_avg_price_5000'].fillna(impute_missing_0, inplace=True)
    df_new['cafe_sum_5000_max_price_avg'].fillna(impute_missing_0, inplace=True)
    df_new['cafe_sum_5000_min_price_avg'].fillna(impute_missing_0, inplace=True)

    df_new['preschool_quota'].fillna(impute_missing_0, inplace=True)
    df_new['school_quota'].fillna(impute_missing_0, inplace=True)
    
    if keep_is_missing:
        df_new['missing_build_info'] = df_new['build_count_block'].isnull().astype(int)
    df_new['build_count_block'].fillna(impute_missing_0, inplace=True)
    df_new['build_count_after_1995'].fillna(impute_missing_0, inplace=True)
    df_new['build_count_before_1920'].fillna(impute_missing_0, inplace=True)
    df_new['build_count_wood'].fillna(impute_missing_0, inplace=True)
    df_new['build_count_mix'].fillna(impute_missing_0, inplace=True)
    df_new['build_count_brick'].fillna(impute_missing_0, inplace=True)
    df_new['build_count_foam'].fillna(impute_missing_0, inplace=True)
    df_new['build_count_frame'].fillna(impute_missing_0, inplace=True)
    df_new['build_count_1921-1945'].fillna(impute_missing_0, inplace=True)
    df_new['build_count_monolith'].fillna(impute_missing_0, inplace=True)
    df_new['build_count_panel'].fillna(impute_missing_0, inplace=True)
    df_new['build_count_slag'].fillna(impute_missing_0, inplace=True)
    df_new['raion_build_count_with_material_info'].fillna(impute_missing_1, inplace=True)
    df_new['raion_build_count_with_builddate_info'].fillna(impute_missing_1, inplace=True)
    df_new['build_count_1946-1970'].fillna(impute_missing_0, inplace=True)
    df_new['build_count_1971-1995'].fillna(impute_missing_0, inplace=True)

#     df_new['prom_part_5000'].fillna(df_new['prom_part_5000'].median(), inplace=True)
#     df_new['metro_km_walk'].fillna(df_new['metro_km_walk'].median(), inplace=True)
#     df_new['metro_min_walk'].fillna(df_new['metro_min_walk'].median(), inplace=True)
#     df_new['id_railroad_station_walk'].fillna(df_new['id_railroad_station_walk'].median(), inplace=True)
#     df_new['railroad_station_walk_min'].fillna(df_new['railroad_station_walk_min'].median(), inplace=True)
#     df_new['railroad_station_walk_km'].fillna(df_new['railroad_station_walk_km'].median(), inplace=True)
#     df_new['green_part_2000'].fillna(df_new['green_part_2000'].median(), inplace=True)
    df_new['product_type'].fillna(df_new['product_type'].mode()[0], inplace=True)

    '''----------------------------------------------------------------------------------
        fill missing values in full_sq, life_sq, kitch_sq, num_room
        use apartment strategy to fill the most likely values
    ----------------------------------------------------------------------------------'''
    df_new.loc[df_new.full_sq == 5326.0, 'full_sq'] = 53
    df_new.loc[df_new.full_sq < 10, 'full_sq'] = np.nan
    df_new.loc[df_new.life_sq == 2, 'life_sq'] = np.nan
    df_new.loc[df_new.life_sq == 7478.0, 'life_sq'] = 48
    
    df_new['_missing_num_room'] = df_new['num_room'].isnull().astype(int)
    df_new['_missing_kitch_sq'] = df_new['kitch_sq'].isnull().astype(int)
    df_new['_missing_material'] = df_new['material'].isnull().astype(int)
    df_new['_missing_max_floor'] = df_new['max_floor'].isnull().astype(int)
    df_new['_apartment_name']=df_new['sub_area'] + df_new['metro_km_avto'].apply(lambda x: np.round(x)).astype(str)
    df_new['_apartment_name_drop']=df_new['sub_area'] + df_new['metro_km_avto'].apply(lambda x: np.round(x)).astype(str)
    df_new.life_sq.    fillna(df_new.groupby(['_apartment_name_drop'])['life_sq'].transform("median"), inplace=True)
    df_new.full_sq.    fillna(df_new.groupby(['_apartment_name_drop'])['full_sq'].transform("median"), inplace=True)
    df_new.kitch_sq.    fillna(df_new.groupby(['_apartment_name_drop'])['kitch_sq'].transform("median"), inplace=True)
    df_new.num_room.    fillna(df_new.groupby(['_apartment_name_drop'])['num_room'].transform("median"), inplace=True)
    df_new.life_sq.    fillna(df_new.groupby(['sub_area'])['life_sq'].transform("median"), inplace=True)
    df_new.full_sq.    fillna(df_new.groupby(['sub_area'])['full_sq'].transform("median"), inplace=True)
    df_new.kitch_sq.    fillna(df_new.groupby(['sub_area'])['kitch_sq'].transform("median"), inplace=True)
    df_new.num_room.    fillna(df_new.groupby(['sub_area'])['num_room'].transform("median"), inplace=True)
    
    '''----------------------------------------------------------------------------------
        fix wrong values
    ----------------------------------------------------------------------------------'''
    wrong_kitch_sq_index = df_new['kitch_sq'] > df_new['life_sq']
    df_new.loc[wrong_kitch_sq_index, 'kitch_sq'] = df_new.loc[wrong_kitch_sq_index, 'life_sq'] * 1 / 3

    wrong_life_sq_index = df_new['life_sq'] > df_new['full_sq']
    df_new.loc[wrong_life_sq_index, 'life_sq'] = df_new.loc[wrong_life_sq_index, 'full_sq'] * 3 / 5
    df_new.loc[wrong_life_sq_index, 'kitch_sq'] = df_new.loc[wrong_life_sq_index, 'full_sq'] * 1 / 5
    df_new.loc[df_new.life_sq.isnull(), 'life_sq'] = df_new.loc[df_new.life_sq.isnull(), 'full_sq'] * 3 / 5

    '''----------------------------------------------------------------------------------
        others
    ----------------------------------------------------------------------------------'''
    df_new['_rel_kitch_sq'] = df_new['kitch_sq'] / df_new['full_sq'].astype(float)
    
    df_new['_room_size'] = (df_new['life_sq'] - df_new['kitch_sq']) / df_new.num_room
    df_new['_room_size'] = df_new['_room_size'].apply(lambda x: 0 if x > 50 else x)
    df_new['_room_size'].fillna(0, inplace=True)
    
    df_new['_life_proportion'] = df_new['life_sq'] / df_new['full_sq']
    df_new['_kitch_proportion'] = df_new['kitch_sq'] / df_new['full_sq']
    
    df_new['_other_sq'] = df_new['full_sq'] - df_new['life_sq']
    df_new['_other_sq'] = df_new['_other_sq'].apply(lambda x: 0 if x <0 else x)
    
    df_new['max_floor'].fillna(1, inplace=True)
    df_new['floor'].fillna(1, inplace=True)
    
    wrong_max_floor_index = ((df_new['max_floor'] - df_new['floor']).fillna(-1)) < 0
    df_new['max_floor'][wrong_max_floor_index] = df_new['floor'][wrong_max_floor_index]
    df_new['max_floor'].fillna(1, inplace=True)
    
    df_new['_floor_from_top'] = df_new['max_floor'] - df_new['floor']
    df_new['_floor_by_top'] = df_new['floor'] / df_new['max_floor']

    # Year
    df_new.ix[df_new['build_year'] == 2, 'build_year'] = np.nan
    df_new.ix[df_new['build_year'] == 3, 'build_year'] = np.nan
    df_new.ix[df_new['build_year'] == 20, 'build_year'] = np.nan
    df_new.ix[df_new['build_year'] == 71, 'build_year'] = np.nan
    df_new.ix[df_new['build_year'] == 215, 'build_year'] = np.nan
    df_new.ix[df_new['build_year'] == 4965, 'build_year'] = 1956
    df_new.ix[df_new['id'] == (df_new.ix[df_new['build_year'] == 20052009]['id'].values[0]+1), 'build_year'] = 2009
    
    df_new.ix[df_new['build_year'] == 20052009, 'build_year'] = 2005
    df_new['_build_year_missing'] = df_new['build_year'].isnull().astype(int)
    df_new['build_year'].fillna(df_new.groupby(['sub_area', 'max_floor'])['build_year'].
                               transform('median'), inplace=True)
    df_new['build_year'].fillna(df_new.groupby(['sub_area'])['build_year'].
                               transform('median'), inplace=True)
    df_new['_age_of_house_before_sale'] = np.abs(df_new.timestamp.dt.year - df_new.build_year)
    df_new['_sale_before_build'] = ((df_new.timestamp.dt.year - df_new.build_year) < 0).astype(int)
    
    # State
    df_new['_missing_state'] = df_new['state'].isnull().astype(int)
    state_missing_map = {33:3, None:0}
    df_new['state'] = df_new.state.replace(state_missing_map)

    
    df_new.material.fillna(0, inplace=True)

    df_new = get_stats_target(df_new, ['sub_area'], ['max_floor'])
    df_new = get_stats_target(df_new, ['sub_area'], ['num_room'])
    df_new = get_stats_target(df_new, ['sub_area'], ['full_sq'])
    df_new = get_stats_target(df_new, ['sub_area'], ['life_sq'])
    df_new = get_stats_target(df_new, ['sub_area'], ['kitch_sq'])
    
    # 1m 2m 3m part
    df_new['_particular_1m_2m_3m_missing'] = df_new['_missing_num_room'] + df_new['_missing_kitch_sq']                                             + df_new['_missing_max_floor'] + df_new['_missing_material']
    
    sub_area_donot_contain_1m2m3m = ['Arbat',
                                     'Molzhaninovskoe',
                                     'Poselenie Filimonkovskoe',
                                     'Poselenie Kievskij',
                                     'Poselenie Mihajlovo-Jarcevskoe',
                                     'Poselenie Rjazanovskoe',
                                     'Poselenie Rogovskoe',
                                     'Poselenie Voronovskoe',
                                     'Vostochnoe']

    df_new['_particular_1m_2m_3m_sub_area'] = 1 - df_new.sub_area.isin(sub_area_donot_contain_1m2m3m).astype(int)
    df_new['_particular_1m_2m_3m_magic'] = df_new['_particular_1m_2m_3m_missing']*10 + df_new['_particular_1m_2m_3m_sub_area']
    df_new.drop(['_missing_num_room', '_missing_kitch_sq', '_missing_max_floor', '_missing_material','_particular_1m_2m_3m_sub_area'], axis=1, inplace=True)

    # create new feature
    # district
    df_new['_pop_density'] = df_new.raion_popul / df_new.area_m
    df_new['_hospital_bed_density'] = df_new.hospital_beds_raion / df_new.raion_popul
    df_new['_healthcare_centers_density'] = df_new.healthcare_centers_raion / df_new.raion_popul
    df_new['_shopping_centers_density'] = df_new.shopping_centers_raion / df_new.raion_popul
    df_new['_university_top_20_density'] = df_new.university_top_20_raion / df_new.raion_popul
    df_new['_sport_objects_density'] = df_new.sport_objects_raion / df_new.raion_popul
    df_new['_best_university_ratio'] = df_new.university_top_20_raion / (df_new.sport_objects_raion + 1)
    df_new['_good_bad_propotion'] = (df_new.sport_objects_raion + 1) / (df_new.additional_education_raion + 1)
    df_new['_num_schools'] = df_new.sport_objects_raion + df_new.additional_education_raion
    df_new['_schools_density'] = df_new._num_schools + df_new.raion_popul
    df_new['_additional_education_density'] = df_new.additional_education_raion / df_new.raion_popul
    
    df_new['_ratio_preschool'] = df_new.preschool_quota / df_new.children_preschool
    df_new['_seat_per_preschool_center'] = df_new.preschool_quota / df_new.preschool_education_centers_raion
    df_new['_seat_per_preschool_center'] = df_new['_seat_per_preschool_center'].apply(lambda x: df_new['_seat_per_preschool_center'].median() if x > 1e8 else x)
    
    df_new['_ratio_school'] = df_new.school_quota / df_new.children_school
    df_new['_seat_per_school_center'] = df_new.school_quota / df_new.school_education_centers_raion
    df_new['_seat_per_school_center'] = df_new['_seat_per_school_center'].apply(lambda x: df_new['_seat_per_preschool_center'].median() if x > 1e8 else x)
    
    
    df_new['_raion_top_20_school'] = df_new['school_education_centers_top_20_raion'] / df_new['school_education_centers_raion']
    df_new['_raion_top_20_school'].fillna(0, inplace=True)
    df_new['_maybe_magic'] =  df_new.product_type.apply(str) + '_' + df_new.id_metro.apply(str)
    df_new['_id_metro_line'] = df_new.id_metro.apply(lambda x: str(x)[0])
    
    df_new['_female_ratio'] = df_new.female_f / df_new.full_all
    df_new['_male_ratio'] = df_new.male_f / df_new.full_all
    df_new['_male_female_ratio_area'] = df_new.male_f / df_new.female_f
    df_new['_male_female_ratio_district'] = (df_new.young_male + df_new.work_male + df_new.ekder_male) /                                            (df_new.young_female + df_new.work_female + df_new.ekder_female)
    
    df_new['_young_ratio'] = df_new.young_all / df_new.raion_popul
    df_new['_young_female_ratio'] = df_new.young_female / df_new.raion_popul
    df_new['_young_male_ratio'] = df_new.young_male / df_new.raion_popul
    
    df_new['_work_ratio'] = df_new.work_all / df_new.raion_popul
    df_new['_work_female_ratio'] = df_new.work_female / df_new.raion_popul
    df_new['_work_male_ratio'] = df_new.work_male / df_new.raion_popul
    
    df_new['_children_burden'] = df_new.young_all / df_new.work_all
    df_new['_ekder_ratio'] = df_new.ekder_all / df_new.raion_popul
    df_new['_ekder_female_ratio'] = df_new.ekder_female / df_new.raion_popul
    df_new['_ekder_male_ratio'] = df_new.ekder_male / df_new.raion_popul
    
    sale_dict = dict(df_new[df_new.build_year > 3].groupby(['sub_area'])['timestamp'].count())
    df_new['_on_sale_known_build_year_ratio'] = df_new.sub_area.apply(lambda x: sale_dict[x]) / df_new.raion_build_count_with_builddate_info
    
    df_new['_congestion_metro'] = df_new.metro_km_avto / df_new.metro_min_avto
    df_new['_congestion_metro'].fillna(df_new['_congestion_metro'].mean(), inplace=True)
    df_new['_congestion_railroad'] = df_new.railroad_station_avto_km / df_new.railroad_station_avto_min
    
    df_new['_big_road1_importance'] = df_new.groupby(['id_big_road1'])['big_road1_km'].transform('mean')
    df_new['_big_road2_importance'] = df_new.groupby(['id_big_road2'])['big_road2_km'].transform('mean')
    df_new['_bus_terminal_importance'] = df_new.groupby(['id_bus_terminal'])['bus_terminal_avto_km'].transform('mean')
    
    df_new['_square_per_office_500'] = df_new.office_sqm_500 / df_new.office_count_500
    df_new['_square_per_trc_500'] = df_new.trc_sqm_500 / df_new.trc_count_500
    df_new['_square_per_office_1000'] = df_new.office_sqm_1000 / df_new.office_count_1000
    df_new['_square_per_trc_1000'] = df_new.trc_sqm_1000 / df_new.trc_count_1000
    df_new['_square_per_office_1500'] = df_new.office_sqm_1500 / df_new.office_count_1500
    df_new['_square_per_trc_1500'] = df_new.trc_sqm_1500 / df_new.trc_count_1500
    df_new['_square_per_office_2000'] = df_new.office_sqm_2000 / df_new.office_count_2000
    df_new['_square_per_trc_2000'] = df_new.trc_sqm_2000 / df_new.trc_count_2000    
    df_new['_square_per_office_3000'] = df_new.office_sqm_3000 / df_new.office_count_3000
    df_new['_square_per_trc_3000'] = df_new.trc_sqm_3000 / df_new.trc_count_3000 
    df_new['_square_per_office_5000'] = df_new.office_sqm_5000 / df_new.office_count_5000
    df_new['_square_per_trc_5000'] = df_new.trc_sqm_5000 / df_new.trc_count_5000 
    
    df_new['_square_per_trc_5000'].fillna(0, inplace=True)
    df_new['_square_per_office_5000'].fillna(0, inplace=True)
    df_new['_square_per_trc_3000'].fillna(0, inplace=True)
    df_new['_square_per_office_3000'].fillna(0, inplace=True)
    df_new['_square_per_trc_2000'].fillna(0, inplace=True)
    df_new['_square_per_office_2000'].fillna(0, inplace=True)
    df_new['_square_per_trc_1500'].fillna(0, inplace=True)
    df_new['_square_per_office_1500'].fillna(0, inplace=True)
    df_new['_square_per_trc_1000'].fillna(0, inplace=True)
    df_new['_square_per_office_1000'].fillna(0, inplace=True)
    df_new['_square_per_trc_500'].fillna(0, inplace=True)
    df_new['_square_per_office_500'].fillna(0, inplace=True)
    
    
    df_new['_cafe_sum_500_diff'] = df_new.cafe_sum_500_max_price_avg - df_new.cafe_sum_500_min_price_avg
    # replace it with ordering number
    ecology_map = {"satisfactory":5, "excellent":4, "poor":3, "good":2, "no data":1}
    df_new.ecology = df_new.ecology.map(ecology_map)
    
    return df_new


# In[3]:


train = pd.read_csv("../input/train.csv")
train.columns = map(str.lower, train.columns)
y = train.price_doc
y = y.apply(lambda x: 11111112 if x>111111111 else x)


test = pd.read_csv("../input/test.csv")
test.columns = map(str.lower, test.columns)

df = pd.concat([train, test], axis=0)


# In[6]:


df_preprocessed = preprocess_data(df)


# In[7]:


df_preprocessed.head()

