
import pandas as pd
import numpy as np
import tqdm
datafile = '/home/ysc/workspace/VisualTimeSeries/data/Anomaly_Classification/'
def make_energy_consumption_dataset():
    root = datafile+'SmartMeter_Energy_Consumption/'
    # # 读取CSV文件
    # df = pd.read_csv('/home/ysc/workspace/VisualTimeSeries/data/Anomaly_Classification/SmartMeter_Energy_Consumption/CC_LCL-FullData.csv')

    # # columns = df.columns()
    # # 过滤出type为ToU的行
    # df_filtered = df[df['stdorToU'] == 'ToU']

    # # # 按id和DateTime排序
    # # df_filtered = df_filtered.sort_values(by=['id', 'DateTime'])

    # # 获取唯一的id和DateTime
    # unique_ids = df_filtered['LCLid'].unique()
    # unique_datetimes = df_filtered['DateTime'].unique()

    # # 创建一个二维的numpy数组
    # data_array = np.zeros((len(unique_ids), len(unique_datetimes)))

    # # 填充numpy数组
    # id_to_index = {id_: idx for idx, id_ in enumerate(unique_ids)}
    # datetime_to_index = {dt: idx for idx, dt in enumerate(unique_datetimes)}

    # # for _, row in df_filtered.iterrows():
    # for _, row in tqdm.tqdm(df_filtered.iterrows(), total=len(df_filtered)):
    #     id_idx = id_to_index[row['LCLid']]
    #     datetime_idx = datetime_to_index[row['DateTime']]
    #     value = row['KWH/hh (per half hour) ']
    #     if value == 'Null':
    #         value = 0  # 或者使用 np.nan
    #     else:
    #         value = float(value)
    #     data_array[id_idx, datetime_idx] = value

    # # 保存id和numpy第一维索引的对应关系
    # id_index_mapping = {idx: id_ for id_, idx in id_to_index.items()}

    # # 保存datetime和numpy第二维索引的对应关系
    # datetime_index_mapping = {idx: dt for dt, idx in datetime_to_index.items()}

    # np.save(root + 'data.npy', data_array)
    # import json
    # with open(root + 'id_index_mapping.json', 'w') as f:
    #     json.dump(id_index_mapping, f, indent=4)
    # with open(root + 'datetime_index_mapping.json', 'w') as f:
    #     json.dump(datetime_index_mapping, f, indent=4)
    # print("Numpy Array:")
    # print(data_array)
    # print("ID to Index Mapping:")
    # print(id_index_mapping)
    # print("DateTime to Index Mapping:")
    # print(datetime_index_mapping)

    label_df = pd.read_csv(root + 'Tariffs.csv')
    # 读取索引映射文件
    import json
    with open(root+'datetime_index_mapping.json', 'r') as f:
        datetime_index_mapping = json.load(f)

    # 将索引映射文件转换为DataFrame
    index_df = pd.DataFrame(list(datetime_index_mapping.items()), columns=['Index', 'DateTime'])
    index_df['DateTime'] = pd.to_datetime(index_df['DateTime'])

    # 将标注文件中的日期转换为datetime格式
    label_df['TariffDateTime'] = pd.to_datetime(label_df['TariffDateTime'])
        # 过滤掉不存在的日期
    valid_dates = index_df['DateTime'].isin(label_df['TariffDateTime'])
    index_df = index_df[valid_dates]
    
    merged_df = pd.merge(index_df, label_df, left_on='DateTime', right_on='TariffDateTime', how='left')

    # 删除原本label_df中没有的行
    merged_df = merged_df.dropna(subset=['Tariff'])
    # 保留原来的DateTime信息
    merged_df['OriginalDateTime'] = merged_df['DateTime']

    # 将合并后的Index写入label_df
    label_df['Index'] = merged_df['Index']

    # 选择需要的列，并保留OriginalDateTime
    result_df = merged_df[['Index', 'Tariff', 'OriginalDateTime']]
    print(result_df)

    # 保存结果到CSV文件
    result_df.to_csv(root+'aligned_labels.csv', index=False)

def visuallize_energy_consumption():
    import matplotlib.pyplot as plt
    import os
    root = datafile+'SmartMeter_Energy_Consumption/'
    aligned_labels = pd.read_csv(root + 'aligned_labels.csv')

    # 添加一列包含格式化后的时间
    aligned_labels['FormattedTime'] = pd.to_datetime(aligned_labels['OriginalDateTime']).dt.strftime('%H:%M')

    label_mapping = {'Normal': 0.5, 'Low': 0, 'High': 1}
    aligned_labels['Tariff'] = aligned_labels['Tariff'].map(label_mapping)
    raw_data = np.load(root + 'data.npy')
    refined_data = raw_data[:,aligned_labels['Index']]
    #slinding window
    window_length = 480
    stride = 240
    refined_data = np.average(refined_data, axis=0)
    data_length = refined_data.shape[0]
    total_windows = (data_length - window_length) // stride + 1
    root = root + 'Images/'
    if not os.path.exists(root):
        os.makedirs(root)
    for i in range(total_windows):
        fig, ax = plt.subplots(figsize=(8, 3.2), dpi=100)
        date_time = aligned_labels['FormattedTime'][i * stride:i * stride + window_length]
        index_data = refined_data[i * stride:i * stride + window_length]
        ax.plot(index_data, label='Average Consumption')
        label_data = (index_data.max() - index_data.min()) * aligned_labels['Tariff'][i * stride:i * stride + window_length] + index_data.min()
        label_data = np.array(label_data)
        ax.plot(label_data, 'green', label='Price')

        # # 设置 x 轴标签的间隔
        # num_labels = 10  # 你可以根据需要调整这个值
        # step = max(1, len(date_time) // num_labels)
        # ax.set_xticks(date_time[::step])
        # ax.set_xticklabels(date_time[::step], rotation=45, ha='right')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Time Series Data')
        plt.legend()
        plt.savefig(root +f"{i}.png")
        plt.close()



def make_gas_water_dataset(Normalize=True):
    root = datafile+'Water_Gas_Attack/'
    import pandas as pd
    from scipy.io import arff
    from sklearn.preprocessing import MinMaxScaler
    for filename in ['water_final.arff.txt']:
    # for filename in ['gas_final.arff.txt', 'water_final.arff.txt']:
        data, meta = arff.loadarff(root + filename)
        df = pd.DataFrame(data)
        scalar = MinMaxScaler()
        laebl_types = ['Normal','NMRI','CMRI','MSCI','MPCI','MFCI','DOS','Recon']
        non_constant_features = ['command_address', 'response_address','resp_length','comm_read_function','resp_read_fun',
                               'sub_function','setpoint','control_mode','control_scheme','measurement']
        if filename == 'water_final.arff.txt':
            non_constant_features = ['command_address', 'response_address','resp_length','comm_read_function','resp_read_fun',
                               'sub_function','control_mode','control_scheme','measurement']
            non_constant_features.append(df.columns[14])
        print(df.columns)
        Input = df[non_constant_features]
        df_one_hot = pd.get_dummies(df['result'])
        df = pd.concat([df.iloc[:, :-1], df_one_hot], axis=1)   
        df_one_hot.columns = laebl_types
        error_ratio = [df_one_hot[label].sum()/df_one_hot.shape[0] for label in laebl_types]
        print(f"Error Ratio: {error_ratio}")
        
        if Normalize:
            df[df.columns[:-1]] = scalar.fit_transform(df[df.columns[:-1]])
        # Input = df.iloc[:, :-1]
        # Input = np.array(Input)
        # Target = df.iloc[:, -1].astype(int)
        # Target = np.array(Target)
        df = pd.concat([Input, df_one_hot], axis=1)
        df.to_csv(root + filename.split('.')[0] + '.csv')
        print(df.head())
        # print(f"Input Shape: {Input.shape}")
        # print(f"Target Shape: {Target.shape}")
def visuallize_gas_water():
    import matplotlib.pyplot as plt
    import os
    root = datafile+'Water_Gas_Attack/'
    df = pd.read_csv(root + 'water_final.csv')

    featur_label_mapping = {
        'command_address':['DOS'], 
        'response_address':['Recon'],
        'resp_length':['Recon'],
        'comm_read_function':['DOS'],
        'resp_read_fun':['CMRI'],
        'sub_function':['MFCI'],
        'setpoint':['MPCI'],
        'control_mode':['MSCI'],
        'control_scheme':['MSCI'],
        'measurement':['NMRI','CMRI'],
    }
    #slinding window
    window_length = 480
    stride = 240
    # refined_data = np.average(refined_data, axis=0)
    # data_length = refined_data.shape[0]
    data_length = df.shape[0]
    total_windows = (data_length - window_length) // stride + 1
    root = root + 'Images/'
    if not os.path.exists(root):
        os.makedirs(root)
    for i in range(total_windows):
        # fig, axs = plt.subplots(len(df.columns[:-8]), figsize=(8, 3.2), dpi=100)
        fig, axs = plt.subplots(10, 1, figsize=(15, 30), dpi=100, sharex=True)
        for j, feature in enumerate(df.columns[1:10]):  # 假设前10列是你要绘制的变量
            start_idx = i * stride
            end_idx = start_idx + window_length
            series =  np.array(df[feature][start_idx:end_idx])
            axs[j].plot(series, label=feature)
            for label in featur_label_mapping[feature]:
                feature_label = df[label][start_idx:end_idx] == True
                axs[j].scatter(np.where(feature_label)[0], series[feature_label], color='red', label=label)
                # axs[j].axvline(x=start_idx + df[df[label][start_idx:end_idx] == True].index[0], color='red', linestyle='--')
            axs[j].legend(loc='upper right')
            axs[j].set_ylabel(df.columns[j])

        
        axs[-1].set_xlabel('Time')
        
        plt.tight_layout()
        plt.savefig(os.path.join(root, f'subplot_{i}.png'))
        plt.close(fig)
            # ax.plot(df[feature][i * stride:i * stride + window_length], label=feature)
        # date_time = aligned_labels['FormattedTime'][i * stride:i * stride + window_length]
        # index_data = refined_data[i * stride:i * stride + window_length]
        # ax.plot(index_data, label='Average Consumption')
        # label_data = (index_data.max() - index_data.min()) * aligned_labels['Tariff'][i * stride:i * stride + window_length] + index_data.min()
        # label_data = np.array(label_data)
        # ax.plot(label_data, 'green', label='Price')

        # # 设置 x 轴标签的间隔
        # num_labels = 10  # 你可以根据需要调整这个值
        # step = max(1, len(date_time) // num_labels)
        # ax.set_xticks(date_time[::step])
        # ax.set_xticklabels(date_time[::step], rotation=45, ha='right')
        # plt.xlabel('Time')
        # plt.ylabel('Value')
        # plt.title('Time Series Data')
        # plt.legend()
        # plt.savefig(root +f"{i}.png")
        # plt.close()
def find_anomaly_intervals(label):
    intervals = []
    start = None
    for i, val in enumerate(label):
        if val == 1 and start is None:
            start = i
        elif val == 0 and start is not None:
            intervals.append([start, i - 1])
            start = None
    if start is not None:
        intervals.append([start, len(label) - 1])
    return intervals

def visuallize_timeeval():
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import os
    import glob
    dataroot = '/home/ysc/workspace/VisualTimeSeries/data/timeeval-datasets/univariate/'
    image_root = os.path.join(dataroot, 'img')
    done = ['CalIt2','CATSv2','Daphnet','Dodgers']
    # dataroot = '/home/ysc/workspace/VisualTimeSeries/data/timeeval-datasets/multivariate/CalIt2/CalIt2-traffic.test.csv'
    for file in os.listdir(dataroot):
    # for file in good_pattern:
        if file in done:
            continue
        # image_file= os.path.join(image_root,file)
        datafile = os.path.join(dataroot,file)
        csv_files = glob.glob(os.path.join(datafile, '*.csv'))
        # image_files = glob.glob(os.path.join(datafile, '*.png'))
        # print(csv_files)
        for csv_file in csv_files:
            specifile = csv_file.split('/')[-1]
            specifile = (specifile.split('.')[0]+'.'+specifile.split('.')[1] if len(specifile.split('.'))>2 else specifile.split('.')[0])
            # if specifile in ['CalIt2-traffic.test','CATSv2.train']:
            #     continue
            if specifile.split('.')[-1] == 'train':
                continue
            print(specifile)
            image_file = os.path.join(image_root, file, specifile)
            if not os.path.exists(image_file):
                os.makedirs(image_file)
            data = pd.read_csv(csv_file)
            columns = data.columns
            anomalies = data[columns[-1]].astype(int)
            events = np.where(anomalies == 1)[0]
            anomaly_rate = len(events) / len(anomalies)
            # print(anomalies)
            for column in columns[:-1]:
                if column == 'timestamp':
                    continue
                print(f"    {column}")
                vis_data = data[column]
                plt.figure(figsize=(200, 4))
                plt.plot(vis_data, label='Data', color='blue')
                for event in events:
                    plt.scatter(event, vis_data[event], color='red', label='Anomaly' if event == events[0] else "")
                intervals = find_anomaly_intervals(anomalies)
                # plt.plot(data, label='Data', color='blue')
                # for interval in intervals:
                #     plt.axvspan(interval[0], interval[1], color='red', alpha=0.3, label='Anomaly' if interval == intervals[0] else "")
                # plt.title('Data with Anomalies')
                # plt.legsend()
                if '/' in column:
                    column = column.replace('/','_')
                plt.title(f"Column: {column} with Anomalies rate: {anomaly_rate}")
                plt.tick_params(axis='x', width=1)
                # plt.legend()
                plt.savefig(os.path.join(image_file,column+'_scatter.png'))
                plt.close()

if __name__ == '__main__':
    # make_energy_consumption_dataset()
    # make_gas_water_dataset(Normalize=False)
    # visuallize_energy_consumption()
    # visuallize_gas_water()
    visuallize_timeeval()
    
