import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras

from config import Config
from model import Riboformer

def main():
    
    parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter
        )

    test = parser.add_argument_group(title = 'Test parameters',
                                     description = 'Parameters for testing model')
    test.add_argument('-e', '--epoch', default = 10, type = int, 
                      help = 'epoch number for model training')
    test.add_argument('-b', '--batch', default = 64, type = int, 
                      help = 'batch size for model training')
    test.add_argument('-s', '--split', default = 0.7, type = float,
                      help = 'proportion for the training dataset')
    test.add_argument('-l', '--learning', default = 0.0005, type = float, 
                      help = 'learning rate for model training')
    test.add_argument('-save', '--save', action = 'store_true', 
                      help = 'Save the model and training results')

    run = parser.add_argument_group(title = 'Prediction parameters',
                                    description = 'Parameters for Prediction using model')
    run.add_argument('-o', '--output_h5', default = 'cm_mg_model2', help = 'Output keras model')
    run.add_argument('-i', '--input_folder', default = 'GSE119104_Mg_buffer', help='Input data folder')

    args = parser.parse_args()
    
    
    # initiate the model & the model configs
    model_config = Config()
    
    model = Riboformer(model_config)
    
    wsize = model_config.wsize

    print("--------------------------------------------------\nData loading")

    path = os.getcwd()
    parpath = os.path.dirname(path)
    datapath = parpath + '/datasets/' + args.input_folder + '/'
    print(f"Current dictionary: {datapath}")

    all_files = os.listdir(datapath)
    xc_files = [f for f in all_files if f.endswith('xc.txt')]
    yc_files = [f for f in all_files if f.endswith('yc.txt')]
    x_c = np.loadtxt(datapath + xc_files[0], delimiter="\t")
    y_c = np.loadtxt(datapath + yc_files[0], delimiter="\t")

    x_c[:,:wsize] = x_c[:,:wsize]/100 - 5
    y_c = y_c/100 - 5
    
    data_size = x_c.shape[0]
    num_train = int(args.split * data_size)
    num_val = (data_size - num_train) // 2

    # randomly shuffle the data
    indices = np.random.permutation(data_size)
    training_idx, test_idx, val_idx = np.split(indices, [num_train, num_train + num_val])

    x_train, y_train = x_c[training_idx], y_c[training_idx]
    x_val, y_val = x_c[val_idx], y_c[val_idx]
    x_test, y_test = x_c[test_idx], y_c[test_idx]

    print(len(x_train), "Training sequences")
    print(len(x_val), "Validation sequences")
    print(len(x_test), "Testing sequences")

    x_train = [x_train[:, -wsize:], x_train[:, :wsize]]
    x_test = [x_test[:, -wsize:], x_test[:,:wsize]]
    x_val = [x_val[:, -wsize:], x_val[:, :wsize]]

    starter_learning_rate = args.learning
    decay_steps = args.epoch * data_size / args.batch

    learning_rate_fn = tf.keras.optimizers.schedules.CosineDecay(
        starter_learning_rate,
        decay_steps,
        alpha = 0.0
    )

    opt = tf.keras.optimizers.Adam(learning_rate = learning_rate_fn)
    model.compile(optimizer = opt, loss = "mean_squared_error", metrics = ["accuracy"])

    print("--------------------------------------------------\ntraining")

    history = model.fit(
        x_train, 
        y_train,
        batch_size = args.batch, 
        epochs = args.epoch,
        validation_data = (x_val, y_val),
        verbose = 1,
    )

    val_acc_per_epoch = history.history['val_loss']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    
    predict_test = model.predict(x_test).reshape(-1)
    predict_train = model.predict(x_train).reshape(-1)
    
    print(predict_test.shape)

    model_results = np.zeros((num_val, 2))
    model_results[:,0] = y_test
    model_results[:,1] = predict_test

    # save the model prediction in the test dataset
    if args.save:
        np.savetxt(datapath + 'model_prediction.txt', model_results, delimiter = '\t')

    # output all the correlations
    corr = np.corrcoef(predict_test, y_test)[0,1]
    print("model prediction", corr)
    
    corr = np.corrcoef(predict_train, y_train)[0,1]
    print("model training", corr)
    
    corr = np.corrcoef(x_test[1][:num_val, wsize//2], y_test[:num_val])[0,1]
    print("original correlation", corr)

    # save the model
    if args.save:
        model.save(parpath + "/models/" + args.output_h5, save_format = 'tf')

    print("--------------------------------------------------\nfinished!")       

if __name__ == "__main__":
    main()
