import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

def read_log_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    patterns = {
        'train_loss': r'Loss:\s(\d+\.\d+)',
        'train_i2t_acc': r'Image2Text Acc:\s(\d+\.\d+)',
        'train_t2i_acc': r'Text2Image Acc:\s(\d+\.\d+)',
        'valid_loss': r'Valid Loss:\s(\d+\.\d+)',
        'valid_i2t_acc': r'Image2Text Acc:\s(\d+\.\d+)',
        'valid_t2i_acc': r'Text2Image Acc:\s(\d+\.\d+)'
    }
    pa_temp = r'Valid Loss:\s(\d+\.\d+)'
    data = {key: [] for key in patterns.keys()}

    for line in lines:
        for key, pattern in patterns.items():
            match = re.search(pattern, line)
            if key == 'valid_i2t_acc':
                match_temp = re.search(pa_temp, line)
                #print("1")
                if match and match_temp:
                    data[key].append(float(match.group(1)))
                continue
            if key == 'valid_t2i_acc':
                match_temp = re.search(pa_temp, line)
                if match and match_temp:
                    data[key].append(float(match.group(1)))
                continue
            if match:
                data[key].append(float(match.group(1)))

    #print(len(data['valid_loss']))
    #print(data['valid_loss'])
    #print(len(data['valid_i2t_acc']))
    #print(len(data['valid_t2i_acc']))
          
    return data

def plot_loss_and_acc(data):
    epochs_train = range(1, len(data['train_loss']) + 1)
    epochs_valid = range(1, len(data['valid_loss']) + 1)

    original_x = np.arange(1, 173)
    new_x = np.linspace(1, 25633, 172)
    f_1 = interp1d(original_x, data['valid_loss'], kind='linear', fill_value="extrapolate")
    xpanded_data_points_1 = f_1(new_x)
    f_2 = interp1d(original_x, data['valid_i2t_acc'], kind='linear', fill_value="extrapolate")
    xpanded_data_points_2 = f_2(new_x)
    f_3 = interp1d(original_x, data['valid_t2i_acc'], kind='linear', fill_value="extrapolate")
    xpanded_data_points_3 = f_3(new_x)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs_train, data['train_loss'], 'b', label='Training loss')
    plt.plot(new_x, data['valid_loss'], 'g', label='Validation loss',linewidth=5)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(epochs_train, data['train_i2t_acc'], 'r', label='Training Image2Text accuracy')
    plt.plot(new_x, data['valid_i2t_acc'], 'm', label='Validation Image2Text accuracy',linewidth=5)
    plt.title('Training and Validation Image2Text Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs_train, data['train_t2i_acc'], 'c', label='Training Text2Image accuracy')
    plt.plot(new_x, data['valid_i2t_acc'], 'y', label='Validation Text2Image accuracy',linewidth=5)
    plt.title('Training and Validation Text2Image Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.savefig('loss_acc_plot.png')

if __name__ == "__main__":
    log_file_path = '/home/dengyiru/Chinese-CLIP-master/data_path/experiments/vip_finetune_vit-b-16_roberta-base_bs128_4gpu/out_2023-04-12-03-48-12.log'
    data = read_log_file(log_file_path)
    plot_loss_and_acc(data)
