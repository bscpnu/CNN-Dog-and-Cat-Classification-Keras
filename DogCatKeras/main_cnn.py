import dataset
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers. normalization import BatchNormalization
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import numpy as np


def cnn_model(data_train, img_size):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_size, img_size, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(96, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(96, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.summary()

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(data_train.images, data_train.labels, validation_split=0.16, batch_size=50, epochs=100, verbose=1)

    return model, history


def summary_result(data_act, data_pred):

    cm = confusion_matrix(data_act, data_pred)
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    classNames = ['Dog', 'Cat']
    plt.title('Confusion Matrix of Dog and Cat Classification')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames)
    plt.yticks(tick_marks, classNames)
    s = [['TN', 'FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(s[i][j]) + " = " + str(cm[i][j]))
    plt.show()

    # calculate AUC
    auc = roc_auc_score(data_act, data_pred)
    print('AUC score = %.3f' % auc)

    # calculate roc curve
    fpr, tpr, thresholds = roc_curve(data_act, data_pred)

    # plot no skill
    plt.title('ROC Curve')
    plt.plot([0, 1], [0, 1], linestyle='--')
    # plot the roc curve for the model
    plt.plot(fpr, tpr, marker='.')
    # show the plot
    plt.show()

    acc = accuracy_score(data_act, data_pred)
    print("Accuracy score = ", acc)

if __name__=='__main__':

    img_size = 128
    class_list = ['dogs', 'cats']

    train_path = 'data/train/'
    test_path = 'data/test/'
    checkpoint_dir = "models/"

    data_train = dataset.read_data_sets(train_path, img_size, class_list)
    data_test = dataset.read_data_sets(test_path, img_size, class_list)

    print("Size of:")
    print("- Training-set:\t\t{}".format(len(data_train.labels)))
    print("- Test-set:\t\t{}".format(len(data_test.labels)))

    model, history = cnn_model(data_train, img_size)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    #testing phase
    pred_class = model.predict_classes(data_test.images)
    pred_proba = model.predict_proba(data_test.images)

    data_id_test = pd.DataFrame(data_test.ids, columns=['image name'])
    data_labels_test = pd.DataFrame(data_test.labels, columns=['actual', 'actual_label'])
    data_class_predict = pd.DataFrame(pred_class, columns=['predicted'])
    data_class_proba = pd.DataFrame(pred_proba)

    data_act = np.where(data_labels_test['actual_label'] == 1, 'cat', 'dog')
    data_pred = np.where(data_class_predict['predicted']== 1, 'cat', 'dog')

    df_data_act = pd.DataFrame(data_act, columns=['actual'])
    df_data_pred = pd.DataFrame(data_pred, columns=['predicted'])

    result_test = pd.concat([data_id_test, df_data_act, df_data_pred], axis=1)

    df_result_test = pd.DataFrame(result_test)
    df_result_test.to_csv("result_test.csv")


    summary_result(data_labels_test['actual_label'], data_class_predict['predicted'])