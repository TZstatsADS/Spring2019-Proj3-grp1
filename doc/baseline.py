import pandas as pd
import numpy as np
import os
import cv2
import random
import time
import sys
from sklearn.ensemble import GradientBoostingRegressor


def Features(LR_img, HR_img, n_points = 1000, each_pixel = False):

    if len(LR_img) != len(HR_img):
        print("train and test not the same length")
        sys.exit(0)

    datanum = len(LR_img)
    features = []
    responses = []
    
    # find features for a certain pixel in LR
    def surrounding(img, r, c):
        # adding zero edges
        # central = img[r, c]
        central = 0
        new_img = np.insert(img, [0,img.shape[0]-1], values=0, axis=0)
        new_img = np.insert(new_img, [0,img.shape[1]-1], values=0, axis=1)
        r += 1; c += 1
        
        surr = np.array(
        [new_img[r-1,c-1]-central, 
        new_img[r-1,c]-central,
        new_img[r-1,c+1]-central,
        new_img[r,c-1]-central,
        new_img[r,c+1]-central, 
        new_img[r+1,c-1]-central,
        new_img[r+1,c]-central,
        new_img[r+1,c+1]-central])

        return(surr)
    
    # find corresponding pixels(4) in HR
    def corresponding(img, r, c):
        raw = img[2*(r+1)-2:2*(r+1), 2*(c+1)-2:2*(c+1)]
        nraw = np.concatenate(raw)
        return(nraw)
    
    # capture only features
    if each_pixel:
        for ind in range(datanum):
            row = np.array(range(LR_img[ind].shape[0]))
            col = np.array(range(LR_img[ind].shape[1]))
            for r, c in zip(row, col):
                features.append(surrounding(LR_img[ind], r, c))

        return(np.array(features))

    else:
        # make sure you have the right shape of features 
        # feat : n_files x n_points, 8, 3
        # resp : n_files x n_points, 4, 3
        for ind in range(datanum):
            row = LR_img[ind].shape[0]
            col = LR_img[ind].shape[1]

            row = np.random.choice(range(row), size=n_points)
            col = np.random.choice(range(col), size=n_points)
            # feats = []
            # subpixels = []

            for r, c in zip(row, col):
                features.append(surrounding(LR_img[ind], r, c))
                responses.append(corresponding(HR_img[ind], r, c))

            # features.append(np.array(feats))
            # responses.append(np.array(subpixels))
        return(np.array(features), np.array(responses))


def Train(feature, response, depth = 3):
    model_list = []

    params = {'n_estimators': 200, 'max_depth': depth, 
    'min_samples_split': 2,
    'learning_rate': 0.01, 'loss': 'ls'}
    model = GradientBoostingRegressor(**params)

    for i in range(12):
        c1 = i % 4 
        c2 = (i-c1) // 4 
        # 12 different GBMs
        X = feature[:,:,c2] 
        y = response[:,c1,c2]
        model_list.append(model.fit(X, y))
    
    return(model_list)

def Test(model_list, dat):

    predArr = np.zeros([dat.shape[0], 4, 3])
    for i in range(12):
        fit_train = model_list[i]
        ### calculate column and channel
        c1 = i % 4 
        c2 = (i-c1) // 4 
        featMat = dat[:,:, c2]
        ### make predictions
        predArr[:, c1, c2] = fit_train.predict(featMat)

    return(predArr)


def CrossValidation(X, y, depth, K):
    n = y.shape[0]
    n_fold = K
    # n_fold cross validation
    s = np.random.choice(range(n_fold), n)  
    cv_error = np.zeros(n_fold)

    # print(s)
    for i in range(n_fold):
        flag = [i != x for x in s]
        train_dat = X[flag, :, :]
        train_lab = y[flag, :, :]
        test_dat = X[np.invert(flag), :, :]
        test_lab = y[np.invert(flag), :, :]
        
        fit = Train(train_dat, train_lab, depth)
        pred = Test(fit, test_dat)  
        cv_error[i] = np.mean((pred - test_lab)**2)   

    return((np.mean(cv_error), np.std(cv_error)))

def SuperResolution(LR_img, HR_dir, model_list):
    feat = Features(LR_img, HR_img, each_pixel = True)
    pred = Test(model_list, feat)
    # save prediction



if __name__ == "__main__":
    
    start = time.time()

    print(os.curdir)
    dir = "/Users/tianchenwang/Git/proj3/train_set"
    os.chdir(dir)

    LR_dir = dir + "/LR/"
    HR_dir = dir + "/HR/"
    LR_img = os.listdir(LR_dir)[:10]
    HR_img = os.listdir(HR_dir)[:10]

    # read image as numpy array
    LR_img = [cv2.imread(LR_dir+i) for i in LR_img]
    HR_img = [cv2.imread(HR_dir+i) for i in HR_img]

    print(len(LR_img))
    print(len(HR_img))

    ## features extraction
    feat, resp = Features(LR_img, HR_img)

    ## models training 
    # models = Train(feat, resp)

    ## cross validation
    # """
    para_depth = [5,6,7]
    crr = np.zeros(len(para_depth))
    for dep in para_depth:
        print("dep: ", dep, CrossValidation(feat, resp, dep, 3))

    # """


    end = time.time()
    print("time: ", end - start)