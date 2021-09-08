import keras
from sklearn.metrics import roc_curve, auc, confusion_matrix, roc_auc_score
import numpy as np
import os
import cv2 as cv
import random
from fileStream import writeToFile, initFile


class Histories(keras.callbacks.Callback):

    def __init__(self, val_data, filename):
        super(Histories, self).__init__()
        self.x_batch = []
        self.y_batch = []
        for i in range(len(val_data)):
            x, y = val_data.__getitem__(i)
            self.x_batch.extend(x)
            self.y_batch.extend(np.ndarray.astype(y, int))
        self.aucs = []
        self.specificity = []
        self.sensitivity = []
        self.losses = []
        self.filename = filename
        return

    def on_train_begin(self, logs={}):
        initFile(self.filename)
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        y_pred = self.model.predict(np.asarray(self.x_batch))
        con_mat = confusion_matrix(np.asarray(self.y_batch).argmax(axis=-1), y_pred.argmax(axis=-1))
        tn, fp, fn, tp = con_mat.ravel()
        sens = tp/(tp+fn)
        spec = tn/(tn+fp)
        auc_score = roc_auc_score(np.asarray(self.y_batch).argmax(axis=-1), y_pred.argmax(axis=-1))
        #auc_score = roc_auc_score(np.asarray(self.y_batch).argmax(axis=-1), y_pred[:, 1])
        print("Specificity: %f Sensitivity: %f AUC: %f"%(spec, sens, auc_score))
        print(con_mat)
        self.sensitivity.append(sens)
        self.specificity.append(spec)
        self.aucs.append(auc_score)
        writeToFile(self.filename, epoch, auc_score, spec, sens, self.losses[epoch])
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


def specificity_class_calc(con_mat, y_class=0):
    tn_i = 0
    for i in range(con_mat.shape[0]):
        for j in range(con_mat.shape[0]):
            if i != y_class and j != y_class:
                tn_i += con_mat[i, j]

    fp_i = 0
    for j in range(con_mat.shape[0]):
        if j != y_class:
            fp_i += con_mat[j, y_class]

    return tn_i/(tn_i+fp_i)


def sensitivity_class_calc(con_mat, y_class=0):
    tp_i = con_mat[y_class, y_class]
    fn_i = 0
    for i in range(con_mat.shape[0]):
        if i != y_class:
            fn_i += con_mat[y_class, i]

    if tp_i+fn_i == 0:
        print("Division by 0, sensitivity calculation returned 0")
        return 0

    return tp_i/(tp_i+fn_i)


def avg_macro_spec_sens(con_mat):
    sens_all = 0
    spec_all = 0
    print(con_mat)
    for classes in range(con_mat.shape[0]):
        sens_all += sensitivity_class_calc(con_mat, classes)
        spec_all += specificity_class_calc(con_mat, classes)

    return spec_all/con_mat.shape[0], sens_all/con_mat.shape[0]


def calculate_measures(y, y_pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(y.shape[1]):
        for n in range(y.shape[0]):
            if y[n, i] != i and y_pred[n, i] != i:
                TN += 1
            elif y[n, i] != i and y_pred[n, i] == i:
                FP += 1
            elif y[n, i] == i and y_pred[n, i] != i:
                FN += 1
            elif y[n, i] == i and y_pred[n, i] == i:
                TP += 1
    print("TP: %f, FP: %f, TN: %f, FN: %f" % (TP, FP, TN, FN))
    return TP, FP, TN, FN


def sensitivity_specificity(y, y_pred):
    TP, FP, TN, FN = calculate_measures(y, y_pred)
    return TP / (TP + FN), TN / (TN + FP)


# TN / (TN + FP)
def specificity(y, y_pred):
    TP, FP, TN, FN = calculate_measures(y, y_pred)
    return TN / (TN + FP)


def auc_own(y_true, y_pred):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(y_true.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    return roc_auc["micro"]


def get_model_memory_usage(batch_size, model):
    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    number_size = 4.0
    if K.floatx() == 'float16':
         number_size = 2.0
    if K.floatx() == 'float64':
         number_size = 8.0

    total_memory = number_size*(batch_size*shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes


def image_augmentation(img, aug_nr):

    if aug_nr == 0:
        (h, w) = img.shape[:2]
        center = (w / 2, h / 2)
        scale = 1.0
        M = cv.getRotationMatrix2D(center, random.randint(10, 45), scale)
        return cv.warpAffine(img, M, (h, w))
    elif aug_nr == 1:
        (h, w) = img.shape[:2]
        center = (w / 2, h / 2)
        scale = 1.0
        M = cv.getRotationMatrix2D(center, 180, scale)
        return cv.warpAffine(img, M, (h, w))
    elif aug_nr == 2:
        (h, w) = img.shape[:2]
        center = (w / 2, h / 2)
        scale = 1.0
        M = cv.getRotationMatrix2D(center, random.randint(315, 350), scale)
        return cv.warpAffine(img, M, (h, w))
    elif aug_nr == 3:
        return cv.flip(img, 0)
    elif aug_nr == 4:
        (h, w) = img.shape[:2]
        M = np.array([[1, 0, random.randint(5, 20)], [0, 1, 0]], dtype=float)
        return cv.warpAffine(img, M, (h, w))
    elif aug_nr == 5:
        (h, w) = img.shape[:2]
        M = np.array([[1, 0, 0], [0, 1, random.randint(5, 20)]], dtype=float)
        return cv.warpAffine(img, M, (h, w))
    else:
        return cv.flip(img, 1)


def aug_generator(src_path, dst_path, n):

    if not os.path.isdir(dst_path):
        os.makedirs(dst_path)

    for file in os.listdir(src_path):
        img = cv.imread(src_path+file)
        cv.imwrite(dst_path+file, img)

    count = len(os.listdir(dst_path))

    while count < n:
        for file in os.listdir(src_path):
            img = cv.imread(src_path+file)
            cv.imwrite(dst_path+file[0:-5]+str(count)+".jpeg", image_augmentation(img, random.randint(0, 6)))
            count += 1
            print(count)
            if count > n:
                break

    return print("Done augmenting")


def aug_check(search_path):
    same = []
    for file in os.listdir(search_path):
        if file[file.index('t')+1] == '.':
            for file2 in os.listdir(search_path):
                if file[:file.index('t')] == file2[:file.index('t')]:
                    if not np.bitwise_xor(cv.imread(search_path+file), cv.imread(search_path+file2)).any()\
                            and not file==file2:
                        same.append((file, file2))
                    else:
                        continue
        else:
            continue
    print("Number of duplicates are:", len(same))
    return same


def model_evaluater(model, test_generator, average='macro'):
    x_batch = []
    y_batch = []
    for i in range(len(test_generator)):
        x, y = test_generator.__getitem__(i)
        x_batch.extend(x)
        y_batch.extend(np.ndarray.astype(y, int))

    y_pred = model.predict(np.asarray(x_batch))
    con_mat = confusion_matrix(np.asarray(y_batch).argmax(axis=-1), y_pred.argmax(axis=-1))
    tn, fp, fn, tp = con_mat.ravel()
    spec = tn/(tn+fp)
    sens = tp/(tp+fn)
    auc = roc_auc_score(np.asarray(y_batch).argmax(axis=-1), y_pred.argmax(axis=-1), average=average)
    print("Specificity: %f Sensitivity: %f AUC: %f" %(spec, sens, auc))
    print(con_mat)
    return spec, sens, auc, con_mat

