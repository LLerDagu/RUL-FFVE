import sys
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
import pandas as pd
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib as mpl
import imageio
import os


# --------------------------------------- DATA PRE-PROCESSING ---------------------------------------
def add_remaining_useful_life(df):
    # 1. Get the total number of cycles for each unit
    grouped_by_unit = df.groupby(by="unit_nr")
    max_cycle = grouped_by_unit["time_cycles"].max()

    # 2. Merge the max cycle back into the original frame
    result_frame = df.merge(max_cycle.to_frame(name='max_cycle'), left_on='unit_nr', right_index=True)

    # 3. Calculate remaining useful life for each row
    remaining_useful_life = result_frame["max_cycle"] - result_frame["time_cycles"]
    result_frame["RUL"] = remaining_useful_life
    
    # 4. drop max_cycle as it's no longer needed
    result_frame = result_frame.drop("max_cycle", axis=1)

    return result_frame

def add_operating_condition(df):
    df_op_cond = df.copy()

    df_op_cond['setting_1'] = abs(df_op_cond['setting_1'].round())
    df_op_cond['setting_2'] = abs(df_op_cond['setting_2'].round(decimals=2))
    
    # converting settings to string and concatanating makes the operating condition into a categorical variable
    df_op_cond['op_cond'] = df_op_cond['setting_1'].astype(str) + '_' + \
                        df_op_cond['setting_2'].astype(str) + '_' + \
                        df_op_cond['setting_3'].astype(str)
    
    return df_op_cond

def condition_scaler(df_train, df_test, sensor_names):
    # apply operating condition specific scaling
    scaler = StandardScaler()
    for condition in df_train['op_cond'].unique():
        scaler.fit(df_train.loc[df_train['op_cond'] == condition, sensor_names])
        df_train.loc[df_train['op_cond'] == condition, sensor_names] = scaler.transform(df_train.loc[df_train['op_cond'] == condition, sensor_names])
        df_test.loc[df_test['op_cond'] == condition, sensor_names] = scaler.transform(df_test.loc[df_test['op_cond'] == condition, sensor_names])
    return df_train, df_test


def exponential_smoothing(df, sensors, n_samples, alpha=0.4):
    df = df.copy()
    # 1. take the exponential weighted mean
    df[sensors] = df.groupby('unit_nr')[sensors].apply(lambda x: x.ewm(alpha=alpha).mean()).reset_index(level=0, drop=True)
    
    # 2. drop first n_samples of each unit_nr to reduce filter delay
    def create_mask(data, samples):
        result = np.ones_like(data)
        result[0:samples] = 0
        return result
    
    mask = df.groupby('unit_nr')['unit_nr'].transform(create_mask, samples=n_samples).astype(bool)
    df = df[mask]
    
    return df


def gen_train_data(df, sequence_length, columns):
    data = df[columns].values
    num_elements = data.shape[0]

    # -1 and +1 because of Python indexing
    for start, stop in zip(range(0, num_elements-(sequence_length-1)), range(sequence_length, num_elements+1)):
        yield data[start:stop, :]
        
def gen_data_wrapper(df, sequence_length, columns, unit_nrs=np.array([])):
    if unit_nrs.size <= 0:
        unit_nrs = df['unit_nr'].unique()
        
    data_gen = (list(gen_train_data(df[df['unit_nr']==unit_nr], sequence_length, columns))  # batch_size的划分只在同编号发动机范围内
               for unit_nr in unit_nrs)
    data_array = np.concatenate(list(data_gen)).astype(np.float32)
    return data_array

def gen_labels(df, sequence_length, label):
    data_matrix = df[label].values
    num_elements = data_matrix.shape[0]

    # -1 because I want to predict the rul of that last row in the sequence, not the next row
    return data_matrix[sequence_length-1:num_elements, :]  

def gen_label_wrapper(df, sequence_length, label, unit_nrs=np.array([])):
    if unit_nrs.size <= 0:
        unit_nrs = df['unit_nr'].unique()
        
    label_gen = [gen_labels(df[df['unit_nr']==unit_nr], sequence_length, label) 
                for unit_nr in unit_nrs]
    label_array = np.concatenate(label_gen).astype(np.float32)
    return label_array

def gen_test_data(df, sequence_length, columns, mask_value):
    if df.shape[0] < sequence_length:
        data_matrix = np.full(shape=(sequence_length, len(columns)), fill_value=mask_value) # pad
        idx = data_matrix.shape[0] - df.shape[0]
        data_matrix[idx:,:] = df[columns].values  # fill with available data
    else:
        data_matrix = df[columns].values
        
    # specifically yield the last possible sequence
    stop = data_matrix.shape[0]
    start = stop - sequence_length
    for i in list(range(1)):
        yield data_matrix[start:stop, :]


def gen_test_data_forOne(df, sequence_length, columns, mask_value, step=1):
    if df.shape[0] < sequence_length:
        data_matrix = np.full(shape=(sequence_length, len(columns)), fill_value=mask_value)  # pad
        idx = data_matrix.shape[0] - df.shape[0]
        data_matrix[idx:, :] = df[columns].values  # fill with available data
    else:
        data_matrix = df[columns].values

    # specifically yield the last possible sequence
    stop = data_matrix.shape[0]
    start = (stop - sequence_length) % step

    for i in list(range(start, stop, step)):
        if(i + sequence_length > stop):
            break
        yield data_matrix[i:i+sequence_length, :]

def get_data(dataset, sensors, sequence_length, alpha, threshold):
    # 1. files
    dir_path = './data/'
    train_file = 'train_'+dataset+'.txt'
    test_file = 'test_'+dataset+'.txt'

    # 2. columns
    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = ['s_{}'.format(i+1) for i in range(0,21)]
    col_names = index_names + setting_names + sensor_names

    # 3. data readout
    # 3.1 training Data: comprising both training and validation sets
    train = pd.read_csv((dir_path+train_file), sep=r'\s+', header=None, names=col_names)
    # 3.2 test dataset
    test = pd.read_csv((dir_path+test_file), sep=r'\s+', header=None, names=col_names)
    y_test = pd.read_csv((dir_path+'RUL_'+dataset+'.txt'), sep=r'\s+', header=None, names=['RemainingUsefulLife']).clip(upper=threshold)


    # 4. create RUL values according to the piece-wise target function
    train = add_remaining_useful_life(train)
    train['RUL'].clip(upper=threshold, inplace=True)


    # 5. remove unused sensors
    drop_sensors = [element for element in sensor_names if element not in sensors]

    # 6. scale with respect to the operating condition
    # 'op_cond'
    X_train_pre = add_operating_condition(train.drop(drop_sensors, axis=1)) # train.drop(drop_sensors, axis=1)删除不使用的传感器
    X_test_pre = add_operating_condition(test.drop(drop_sensors, axis=1))
    # Z-score
    X_train_pre, X_test_pre = condition_scaler(X_train_pre, X_test_pre, sensors)

    # 7. exponential smoothing
    X_train_pre= exponential_smoothing(X_train_pre, sensors, 0, alpha)
    X_test_pre = exponential_smoothing(X_test_pre, sensors, 0, alpha)

    # 8. train-val split
    gss = GroupShuffleSplit(n_splits=1, train_size=0.80, random_state=42)
    # generate the train/val for *each* sample -> for that we iterate over the train and val units we want
    # this is a for that iterates only once and in that iterations at the same time iterates over all the values we want,
    # i.e. train_unit and val_unit are not a single value but a set of training/vali units
    for train_unit, val_unit in gss.split(X_train_pre['unit_nr'].unique(), groups=X_train_pre['unit_nr'].unique()):
        train_unit = X_train_pre['unit_nr'].unique()[train_unit]  # gss returns indexes and index starts at 1
        val_unit = X_train_pre['unit_nr'].unique()[val_unit]

        x_train = gen_data_wrapper(X_train_pre, sequence_length, sensors, train_unit)
        y_train = gen_label_wrapper(X_train_pre, sequence_length, ['RUL'], train_unit)  # shape=(xxx, 1)
        
        x_val = gen_data_wrapper(X_train_pre, sequence_length, sensors, val_unit)
        y_val = gen_label_wrapper(X_train_pre, sequence_length, ['RUL'], val_unit)

    # 9. create sequences for test
    test_gen = (list(gen_test_data(X_test_pre[X_test_pre['unit_nr']==unit_nr], sequence_length, sensors, 0.))
               for unit_nr in X_test_pre['unit_nr'].unique())
    x_test = np.concatenate(list(test_gen)).astype(np.float32)

    return x_train, y_train, x_val, y_val, x_test, y_test['RemainingUsefulLife']

def get_oneUnitData(dataset, sensors, sequence_length, alpha, threshold, step=1, unit_nr=1):
    # 1. files
    dir_path = './data/'
    train_file = 'train_' + dataset + '.txt'
    test_file = 'test_' + dataset + '.txt'

    # 2. columns
    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = ['s_{}'.format(i + 1) for i in range(0, 21)]
    col_names = index_names + setting_names + sensor_names  # 表头：数据集每一列的"属性名"

    # 3. data readout
    # 3.1 training Data: comprising both training and validation sets
    train = pd.read_csv((dir_path + train_file), sep=r'\s+', header=None, names=col_names)
    # 3.2 test dataset
    test = pd.read_csv((dir_path + test_file), sep=r'\s+', header=None, names=col_names)
    y_test = pd.read_csv((dir_path + 'RUL_' + dataset + '.txt'), sep=r'\s+', header=None,
                         names=['RemainingUsefulLife']).clip(upper=threshold)

    # 4. create RUL values according to the piece-wise target function
    train = add_remaining_useful_life(train)
    train['RUL'].clip(upper=threshold, inplace=True)

    # 5. remove unused sensors
    drop_sensors = [element for element in sensor_names if element not in sensors]

    # 6. scale with respect to the operating condition
    # 'op_cond'
    X_train_pre = add_operating_condition(train.drop(drop_sensors, axis=1))
    X_test_pre = add_operating_condition(test.drop(drop_sensors, axis=1))
    # Z-score
    X_train_pre, X_test_pre = condition_scaler(X_train_pre, X_test_pre, sensors)

    # 7. exponential smoothing
    X_train_pre = exponential_smoothing(X_train_pre, sensors, 0, alpha)
    X_test_pre = exponential_smoothing(X_test_pre, sensors, 0, alpha)   # size(41214, 11)

    # 8. create sequences for test
    test_gen = list(
        gen_test_data_forOne(X_test_pre[X_test_pre['unit_nr'] == unit_nr], sequence_length, sensors, 0., step=step)
    )
    x_test = np.array(test_gen).astype(np.float32).reshape(-1,30,len(sensors))

    return x_test
# ---------------------------------------------------------------------------------------------------
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=True, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf  # evaluation metric: Could be either total_loss or RMSE or SCORE
        self.val_lossArray_withMin = []
        self.delta = delta

    def __call__(self, val_loss, val_lossArray, model):
        score = -val_loss

        # 1. recording initial score
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, val_lossArray, model)
        # 2. if the current model shows no improvement
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        # 3. if the current model's score is smaller
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, val_lossArray, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_lossArray, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.model = model
        self.val_loss_min = val_loss
        self.val_lossArray_withMin = val_lossArray

# ---------------------------------------  VISUALIZATION ---------------------------------------
def viz_latent_space_3Dtrend(args, encoder, reparameterize, data, targets=[], epoch='Final', device='cpu', save=False, show=True, path='.'):
    # 1. Figure setup
    fig = plt.figure(figsize=(9, 6), dpi=300)

    axes = plt.axes(projection='3d')
    axes.set_xlabel('z1')
    axes.set_ylabel('z2')
    axes.set_zlabel('z3')

    # Adjusting scaling ratio for each axis
    axes.set_box_aspect(args.viz_box_aspect)

    # 3D plot rotation angle
    if args.viz_diy:
        axes.view_init(args.viz_view_init[0], args.viz_view_init[1])
    # print(axes.elev)
    # print(axes.azim)

    # Removing axes
    # plt.axis('off')

    # Removing axis ticklabels
    # axes.axes.xaxis.set_ticklabels([])
    # axes.axes.yaxis.set_ticklabels([])
    # axes.axes.zaxis.set_ticklabels([])

    # Set major ticks for the axes
    if args.viz_diy:
        axes.xaxis.set_major_locator(plt.MultipleLocator(args.viz_major_locator[0]))
        axes.yaxis.set_major_locator(plt.MultipleLocator(args.viz_major_locator[1]))
        axes.zaxis.set_major_locator(plt.MultipleLocator(args.viz_major_locator[2]))

    # Axis value range
    if args.viz_diy:
        axes.set_xlim(args.viz_xlim)
        axes.set_ylim(args.viz_ylim)
        axes.set_zlim(args.viz_zlim)

    # Set the image background to white/transparent
    axes.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0))
    axes.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0))
    axes.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0))

    # Set colorbar
    cmap = mpl.cm.viridis
    # other colorbar
    # clist = ['#F0A6A8', '#CDD9EF', '#8FBEE1']
    # cmap = LinearSegmentedColormap.from_list('free_style',clist)
    norm = mpl.colors.Normalize(vmin=0, vmax=125)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                 ax=axes, orientation='vertical', label='RUL',
                 shrink=0.7, pad=0.13)

    # 2. Plotting: Preventing excessive memory usage, plotting 1000 data points at a time
    for i in range(0, int(data.shape[0]/1000)):
        # dataset
        x = data[i*1000:(i+1)*1000,:,:]
        y = targets[i*1000:(i+1)*1000]

        mu, var = encoder(x)
        z = reparameterize(mu, var)
        z = z.cpu().detach().numpy()

        axes.scatter(z[:,0], z[:,1], z[:,2], c=y.cpu().detach().numpy(), alpha = 0.1, cmap=cmap)

    if i*1000 <= data.shape[0]:
         x = data[i*1000:data.shape[0], :, :]
         y = targets[i*1000:data.shape[0]]
         mu, var = encoder(x)
         z = reparameterize(mu, var)
         z = z.cpu().detach().numpy()
         axes.scatter(z[:, 0], z[:, 1], z[:, 2], c=y.cpu().detach().numpy(), alpha = 0.1, cmap=cmap)

    if save:
        plt.savefig(path+'/latent_space_3D_epoch'+str(epoch)+'.png', transparent = True)
    if show:
        plt.show()

    return z

def create_gif(image_list, gif_name, duration=0.35):
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return

def viz_latent_space_3Dtrend_lineGif(args, encoder, reparameterize, train_data, predict_data, train_target=[], epoch='Final', device='cpu', save=False, show=True, path='.'):
    # 1. Figure setup
    fig = plt.figure(figsize=(9, 6), dpi=300)

    axes = plt.axes(projection='3d')
    axes.set_xlabel('z1')
    axes.set_ylabel('z2')
    axes.set_zlabel('z3')

    # Adjusting scaling ratio for each axis
    axes.set_box_aspect(args.viz_box_aspect)

    # 3D plot rotation angle
    if args.viz_diy:
        axes.view_init(args.viz_view_init[0], args.viz_view_init[1])

    # Set major ticks for the axes
    if args.viz_diy:
        axes.xaxis.set_major_locator(plt.MultipleLocator(args.viz_major_locator[0]))
        axes.yaxis.set_major_locator(plt.MultipleLocator(args.viz_major_locator[1]))
        axes.zaxis.set_major_locator(plt.MultipleLocator(args.viz_major_locator[2]))

    # Axis value range
    if args.viz_diy:
        axes.set_xlim(args.viz_xlim)
        axes.set_ylim(args.viz_ylim)
        axes.set_zlim(args.viz_zlim)

    # Set the image background to white/transparent
    axes.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0))
    axes.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0))
    axes.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0))

    # Set colorbar
    cmap = mpl.cm.viridis
    # clist = ['#F0A6A8', '#CDD9EF', '#8FBEE1']
    # cmap = LinearSegmentedColormap.from_list('free_style', clist)
    norm = mpl.colors.Normalize(vmin=0, vmax=125)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                 ax=axes, orientation='vertical', label='RUL',
                 shrink = 0.7, pad= 0.13)

    # 2. Plotting: Preventing excessive memory usage, plotting 1000 data points at a time
    i = 0
    for i in range(0, int(train_data.shape[0] / 1000)):
        # dataset
        x = train_data[i * 1000:(i + 1) * 1000, :, :]
        y = train_target[i * 1000:(i + 1) * 1000]

        mu, var = encoder(x)
        z = reparameterize(mu, var)
        z = z.cpu().detach().numpy()

        axes.scatter(z[:, 0], z[:, 1], z[:, 2], c=y.cpu().detach().numpy(), cmap=cmap, alpha=args.viz_gif_alpha)

    if i * 1000 <= train_data.shape[0]:
        x = train_data[i * 1000:train_data.shape[0], :, :]
        y = train_target[i * 1000:train_data.shape[0]]
        mu, var = encoder(x)
        z = reparameterize(mu, var)
        z = z.cpu().detach().numpy()
        axes.scatter(z[:, 0], z[:, 1], z[:, 2], c=y.cpu().detach().numpy(), cmap=cmap, alpha=args.viz_gif_alpha)

    # 3. Plotting: Path of objects under evaluation
    image_list = []
    if save:
        os.makedirs(path + '/gif', exist_ok=True)
    x = predict_data[:, :, :]   # dataset

    mu, var = encoder(x)
    z = reparameterize(mu, var)
    z = z.cpu().detach().numpy()
    for point in range(0, z.shape[0]-1):
        # scatter
        axes.scatter(z[point,0], z[point,1], z[point,2], c='red')
        # arrow
        axes.quiver(
            z[point,0], z[point,1], z[point,2],  # <-- starting point of vector
            z[point+1,0]-z[point,0], z[point+1,1]-z[point,1], z[point+1,2]-z[point,2],  # <-- directions of vector
            color='black'
        )

        if point == z.shape[0]-2:
            axes.scatter(z[point+1, 0], z[point+1, 1], z[point+1, 2], c='red')

        if save:
            file_name = path + '/gif/latent_space_3D_line_p' + str(point) + '.png'
            image_list.append(file_name)
            plt.savefig(file_name)

    if show:
        plt.show()

    # 4. creating GIF image
    orgin = path + '/gif'
    gif_name = orgin + '/GIF.gif'  # gif_name
    duration = 0.35
    create_gif(image_list, gif_name, duration)

    return

# --------------------------------------------- RESULTS  --------------------------------------------
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def score(y_true, y_hat):
    res = 0
    res_array = []
    subs_array = []
    for true, hat in zip(y_true, y_hat):
        subs = (hat - true).cpu().detach().numpy()
        subs_array.append(subs)
        # print(subs)
        if subs < 0:
            res_array.append(np.exp(-subs/13)-1)
            res = res + np.exp(-subs/13)-1
        else:
            res_array.append(np.exp(subs/10)-1)
            res = res + np.exp(subs/10)-1
    return res, res_array, subs_array

