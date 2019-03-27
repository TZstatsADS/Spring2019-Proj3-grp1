import pandas as pd
import numpy as np
import os
import cv2
import random
import time
import sys
from joblib import dump, load
from sklearn.ensemble import GradientBoostingRegressor

def Features(LR_img, HR_img, n_points = 1000, each_pixel = False):

    datanum = len(LR_img)
    features = []
    responses = []
    
    # find features for a certain pixel in LR
    def surrounding(img, r, c):
        # adding zero edges
#         central = img[r, c]
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
      
    def surroundingfaster(new_img ,r, c):
        r += 1; c += 1
        central = 0
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
#         print(raw)
        nraw = np.concatenate(raw)
        return(nraw)
    
    # capture only features
    ##
    ### 
    #### too slow here.
    
    if each_pixel:
        for ind in range(datanum):
            img = LR_img[ind]
            row = np.array(range(img.shape[0]))
            col = np.array(range(img.shape[1]))
            
            new_img = np.insert(img, [0,img.shape[0]-1], values=0, axis=0)
            new_img = np.insert(new_img, [0,img.shape[1]-1], values=0, axis=1)
            for r in row:
                for c in col:
                    features.append(surroundingfaster(new_img, r, c))
            if ind % 100 == 0:
              print("featuring goes to number", ind)

        return(np.array(features))
    
    #####

    else:
        # make sure you have the right shape of features 
        # feat : n_files x n_points, 8, 3
        # resp : n_files x n_points, 4, 3
        if len(LR_img) != len(HR_img):
          print("train and test not the same length")
          sys.exit(0)
        
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
            if ind % 10 == 0:
              print("featuring goes to number", ind)
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
        featMat = dat[:, :, c2]
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

# input a single img!!
def SuperResolution(LR_img, HR_dir, model_list):

    def reshape(prediction, numr, numc):
        new_pic = np.zeros([2*numr, 2*numc, 3])
        # print("predic", prediction)
        rpos = 0
        cpos = 0
        for i, pred in enumerate(prediction):
            if i!=0 and i%numc == 0:
                rpos += 2
                cpos = 0

            new_pic[rpos:(rpos+2),cpos:(cpos+2),:] = pred.reshape(2,2,3)
            cpos += 2

        print("newpic", new_pic.shape)
        return(new_pic)
      
    numr = LR_img.shape[0]
    numc = LR_img.shape[1]
    
    t0 = time.clock()
    feat = Features([LR_img], [], each_pixel = True)
    print("feature extraction time: ", time.clock() - t0)
    
#     t0 = time.clock()
    pred = Test(model_list, feat)
#     print("test time: ", time.clock() - t0)
    
#     t0 = time.clock()
    predicted_img = reshape(pred, numr, numc)
#     print("reshape time: ", time.clock() - t0)
    
    # save prediction
    print("save predicted pic")
    cv2.imwrite(HR_dir, predicted_img)

if __name__ == "__main__":
    
    start = time.time()

    print(os.curdir)
    dir = "/Users/tianchenwang/Git/proj3/"
    os.chdir(dir)

    LR_dir = dir + "train_set/LR/"
    HR_dir = dir + "train_set/HR/"
    # flower
    LR_img_dir = os.listdir(LR_dir)[1000:1500]
    HR_img_dir = os.listdir(HR_dir)[1000:1500]

    # read image as numpy array
    LR_img = [cv2.imread(LR_dir+i) for i in LR_img_dir]
    HR_img = [cv2.imread(HR_dir+i) for i in HR_img_dir]
    
    print("data loaded...")
    # print(LR_img[1].shape)
    # print(HR_img[1].shape)

    ## features extraction
        # center = 0
    feat, resp = Features(LR_img, HR_img)

    ## models training 
    models = Train(feat, resp, depth=5)
    # save model
    for i, model in enumerate(models):
        dump(model, 'models/fruit_'+str(i)+'.joblib') 
    
    print("models saved...")

    ## generate new imgs
    # pred_dir = '/Users/tianchenwang/Git/proj3/prediction/3.jpg'
    # SuperResolution(LR_img[3], pred_dir, models)

    ## cross validation
    """
    para_depth = [5,6,7]
    crr = np.zeros(len(para_depth))
    for dep in para_depth:
        print("dep: ", dep, CrossValidation(feat, resp, dep, 3))

    """


    end = time.time()
    print("time: ", end - start)