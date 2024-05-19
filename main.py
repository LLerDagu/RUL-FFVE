import utils
from models.model_FFVE import Encoder, Decoder, reparameterize, total_loss, total_loss_forTest
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from data.dataset import RUL_Dataset
from utils import EarlyStopping
from copy import deepcopy
import os
import argparse
import matplotlib.pyplot as plt
import sys
import time

if __name__ == "__main__":
    # ----------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--save_dir', type=str, default='./checkpoint/FD001/FFVE_test/op_4')
    parser.add_argument('--dataset', type=str, default='FD001')
    parser.add_argument('--itr', type=int, default=5, help='experiments times')
    parser.add_argument('--threshold', type=int, default=125)
    parser.add_argument('--latent_dim', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--early_stopping_with_loss', type=str, default='val_rmse', help='criteria for early stopping: [val_loss, val_rmse, val_score]')
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda:0')

    parser.add_argument('--e_layers', type=int, default=1, help='num of encoder layers')
    parser.add_argument('--norm', type=int, default=False, help='whether to apply LayerNorm')
    parser.add_argument('--fac_C', type=int, default=True, help='whether to apply factorized channel interaction')
    parser.add_argument('--short_res', type=int, default=True, help='short range residual operation')
    parser.add_argument('--long_res', type=int, default=False, help='long range high-low level feature fusion operation')

    args = parser.parse_args()

    dataset = args.dataset
    save_folder = args.save_dir
    os.makedirs(save_folder, exist_ok=True)

    sys.stdout = utils.Logger(save_folder+'/log.log', sys.stdout)
    sys.stderr = utils.Logger(save_folder+'/log.log_file', sys.stderr)
    print(args)
    # ------------------------------ (1) DATA -----------------------------------


    # sensors to work with
    sensors = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_11', 's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21']    # 14
    # windows length
    sequence_length = 30
    # smoothing intensity
    alpha = 0.1
    batch_size = args.batch_size
    # max RUL
    threshold = args.threshold

    # Average_record
    avg_testLoss = np.array([])
    avg_testRMSE = np.array([])
    avg_testScore = np.array([])
    avg_trainTime = np.array([])

    for itr in range(0, args.itr):
        print("Itr: ", itr)
        # Load Dataset
        x_train, y_train, x_val, y_val, x_test, y_test = utils.get_data(dataset, sensors, sequence_length, alpha, threshold)
        tr_dataset = RUL_Dataset(x_train, y_train)
        val_dataset = RUL_Dataset(x_val, y_val)
        test_dataset = RUL_Dataset(x_test, y_test)

        # Load Loader
        tr_loader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        # --------------------------------------- (2) MODEL ----------------------------------------
        timesteps = x_train.shape[1]  # timesteps = window_size
        input_dim = x_train.shape[2]  # input_dim = channel_size
        latent_dim = args.latent_dim
        epochs = 100000
        device = args.device
        lr = args.lr
        patience = args.patience
        early_stopping_with_loss = args.early_stopping_with_loss

        # Initialize model
        encoder = Encoder(configs=args, sequence_length=sequence_length, input_dim=len(sensors),
                          latent_dim=latent_dim).to(device)
        decoder = Decoder(latent_dim=latent_dim).to(device)

        # --------------------------- (3) Optimizer and Early Stopping ---------------------------
        optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        early = EarlyStopping(patience=patience)

        # ----------------------------------- (4) Record -----------------------------------------
        val_loss_record = []
        val_rmse_record = []

        # ---------------------------- (5) Train and Validation ----------------------------------
        time_now = time.time()  # record training time
        for epoch in range(epochs):
            # 1). Model training phase
            # 1. Train
            encoder.train()
            decoder.train()

            # 2. Initialization record
            tr_loss = 0.

            # 3. Training the model
            for tr_x, tr_y in tr_loader:
                # switching devices
                tr_x, tr_y = tr_x.to(device), tr_y.to(device)

                # Model: Forward
                optimizer.zero_grad()   # reset gradients
                # 3.1 encoder
                mu, var = encoder(tr_x)
                # 3.2 reparameterize: obtain latent vector z
                z = reparameterize(mu, var).float()
                # 3.3 decoder: output RUL (scalar)
                out = decoder(z).view(-1)
                # 3.4 loss function
                loss, _, _ = total_loss(out, tr_y, mu, var)
                loss.backward()  # bp
                # 3.5 optimizer: update parameters
                optimizer.step()
                # 3.6 record loss
                tr_loss += loss.item() / len(tr_loader)
            # ---------------------------------

            # 2). Model validation
            # 1. Validation
            encoder.eval()
            decoder.eval()
            # 2. Initialization record
            val_loss = 0.  # average loss for the validation set
            val_rmse = 0.  # average RMSE for the validation set
            val_score = 0.
            # 3. Evaluating the model
            for val_x, val_y in val_loader:
                # val_x=(batch_size, sequence_len, input_dim), val_y=(batch_size)
                val_x, val_y = val_x.to(device), val_y.to(device)

                with torch.no_grad():
                    mu, var = encoder(val_x)
                    z = reparameterize(mu, var)
                    out = decoder(z).view(-1)

                loss, kl_loss, rmse_loss = total_loss(out, val_y, mu, var)

                val_loss += loss / len(val_loader)
                val_rmse += rmse_loss.item() / len(val_loader)
                val_score += utils.score(val_y, out)[0]
            # ---------------------------------
            scheduler.step()    # learning rate decay

            # 3). Model performance: output results on the training and validation sets
            print('Epoch %d : tr_loss %.2f, val_loss %.2f, val_rmse %.2f, val_score %.2f' % (
            epoch, tr_loss, val_loss, val_rmse, val_score))
            # 1. Record: param, val_loss, val_rmse
            param_dict = {'encoder': deepcopy(encoder.state_dict()), 'decoder': deepcopy(decoder.state_dict())}
            val_loss_record.append(val_loss.cpu().detach().numpy())
            val_rmse_record.append(val_rmse)

            # 2. Early Stopping
            if early_stopping_with_loss == 'val_loss':
                early(val_loss, [val_loss, val_rmse, val_score], param_dict)
            elif early_stopping_with_loss == 'val_rmse':  # default
                early(val_rmse, [val_loss, val_rmse, val_score], param_dict)
            elif early_stopping_with_loss == 'val_score':
                early(val_score, [val_loss, val_rmse, val_score], param_dict)

            # 3. If the model has converged, training can be terminated early
            if early.early_stop == True:
                break
            # ---------------------------------
        trainTime = time.time()-time_now  # training time
        # Plotting: RMSE - Epoch iteration graph
        plt.plot(val_rmse_record)
        plt.xlabel('epoch')
        plt.ylabel('RMSE')
        plt.savefig(args.save_dir + '/rmse_epoch_itr'+str(itr)+'.png')
        # plt.show()
        plt.clf()

        # Save Best Model
        torch.save(early.model, os.path.join(save_folder, 'best_model_itr'+str(itr)+'.pt'))

        # --------------------------------- (6) Test --------------------------------
        # 1. Loading the model
        encoder.load_state_dict(early.model['encoder'])
        decoder.load_state_dict(early.model['decoder'])

        # 2. Evaluation mode
        encoder.eval()
        decoder.eval()

        # 3. Initialization record
        test_loss = 0.
        test_rmse = 0.
        test_score = 0.

        # 4. Prediction: test set
        for test_x, test_y in test_loader:
            test_x, test_y = test_x.to(device), test_y.to(device)

            # Model: Forward
            with torch.no_grad():
                mu, var = encoder(test_x)
                z = reparameterize(mu, var)
                out = decoder(z).view(-1)

            # loss function
            loss, kl_loss, rmse_loss = total_loss_forTest(out, test_y, mu, var)

            # record loss
            test_loss += loss / len(test_loader)
            test_rmse += rmse_loss.item() / len(test_loader)

        # RMSE
        test_rmse = np.sqrt(test_rmse)
        # Score
        mu, var = encoder(torch.tensor(x_test).to(device))
        z = reparameterize(mu, var)
        y_test_hat = decoder(z)
        test_score, res_array, subs_array = utils.score(y_test, y_test_hat)

        # plotting the error graph for SCORE
        plt.plot(res_array)
        plt.xlabel('engine_number')
        plt.ylabel('score')
        plt.savefig(args.save_dir + '/score_of_engineUnit_itr'+str(itr)+'.png')
        # plt.show()
        plt.clf()
        plt.plot(subs_array)
        plt.xlabel('engine_number')
        plt.ylabel('subs')
        plt.savefig(args.save_dir + '/subs_of_engineUnit_itr'+str(itr)+'.png')
        # plt.show()
        plt.clf()

        # 5. Model performance: output results on the test set
        avg_testLoss = np.append(avg_testLoss, test_loss.cpu().detach().numpy())
        avg_testRMSE = np.append(avg_testRMSE, test_rmse)
        avg_testScore = np.append(avg_testScore, test_score)
        avg_trainTime = np.append(avg_trainTime, trainTime)  # training time
        print('+----------------------------------------------------------------+')
        print('Itr: %d, Training Time: %.2f seconds' % (itr, trainTime))
        print('Itr: %d, Final Result : test loss %.2f, test_rmse %.2f, test_score %.2f' % (itr, test_loss, test_rmse, test_score))
        with open(os.path.join(save_folder, 'result.txt'), 'a') as f:
            f.writelines('Itr: %d, Training Time: %.2f seconds\n' % (itr, trainTime))
            f.writelines('Itr: %d, Final Result : val loss %.2f, val_rmse %.2f, val_score %.2f\n' % (itr, early.val_lossArray_withMin[0], early.val_lossArray_withMin[1], early.val_lossArray_withMin[2]))
            f.writelines('Itr: %d, Final Result : test loss %.2f, test_rmse %.2f, test_score %.2f\n' % (itr, test_loss, test_rmse, test_score))
        print('+----------------------------------------------------------------+')

    lossAvg, rmseAvg, scoreAvg = (avg_testLoss.mean(), avg_testRMSE.mean(), avg_testScore.mean())
    print('Average Training Time : %.2f seconds' % avg_trainTime.mean())
    print('Final Average Result : test loss %.2f, test_rmse %.2f, test_score %.2f' % (lossAvg, rmseAvg, scoreAvg))
    print('+----------------------------------------------------------------+')
    with open(os.path.join(save_folder, 'result.txt'), 'a') as f:
        f.writelines('Average Training Time : %.2f seconds\n' % avg_trainTime.mean())
        f.writelines('Final Average Result : test loss %.2f, test_rmse %.2f, test_score %.2f\n' % (lossAvg, rmseAvg, scoreAvg))