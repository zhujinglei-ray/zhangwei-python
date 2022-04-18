import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier


# 输出 confusion 矩阵
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    #如果需要 看图片 输入 plt.show()
    # plt.show()

class Model:
    def __init__(self, data):
        self.Y = data['target']
        self.X = data[['Gender', 'Reality', 'ChldNo_1', 'ChldNo_2More', 'wkphone',
                       'gp_Age_high', 'gp_Age_highest', 'gp_Age_low',
                       'gp_Age_lowest', 'gp_worktm_high', 'gp_worktm_highest',
                       'gp_worktm_low', 'gp_worktm_medium', 'occyp_hightecwk',
                       'occyp_officewk', 'famsizegp_1', 'famsizegp_3more',
                       'houtp_Co-op apartment', 'houtp_Municipal apartment',
                       'houtp_Office apartment', 'houtp_Rented apartment',
                       'houtp_With parents', 'edutp_Higher education',
                       'edutp_Incomplete higher', 'edutp_Lower secondary', 'famtp_Civil marriage',
                       'famtp_Separated', 'famtp_Single / not married', 'famtp_Widow']]
        self.X_train, self.X_test, self.y_train, self.y_test = self._smote_deal_with_sample_imbalance()

    # Using Synthetic Minority Over-Sampling Technique(SMOTE) to overcome sample imbalance problem.
    def _smote_deal_with_sample_imbalance(self):
        self.Y = self.Y.astype('int')
        X_balance, Y_balance = SMOTE().fit_resample(self.X, self.Y)
        X_balance = pd.DataFrame(X_balance, columns=self.X.columns)
        # After over sampling, the number between 1 and 0 is balanced. It can be seen from the confusion matrix

        X_train, X_test, y_train, y_test = train_test_split(X_balance, Y_balance,
                                                            stratify=Y_balance, test_size=0.3,
                                                            random_state=10086)

        return X_train, X_test, y_train, y_test

    # log(𝑝1−𝑝)=𝛽0+𝛽1𝑥1+⋅⋅⋅+𝛽𝑞𝑥𝑞
    def logistic_regression(self):
        model = LogisticRegression(C=0.8,
                                   random_state=0,
                                   solver='lbfgs')
        model.fit(self.X_train, self.y_train)
        y_predict = model.predict(self.X_test)

        print('Accuracy Score is {:.5}'.format(accuracy_score(self.y_test, y_predict)))
        print(pd.DataFrame(confusion_matrix(self.y_test, y_predict)))

        sns.set_style('white')
        class_names = ['0', '1']
        plot_confusion_matrix(confusion_matrix(self.y_test, y_predict),
                              classes=class_names, normalize=True,
                              title='Normalized Confusion Matrix: Logistic Regression')
        # model.predict()
        print(self.X_train.head(1))
        return model

    def simple_decision_tree(self):
        class_names = ['0', '1']
        model = DecisionTreeClassifier(max_depth=12,
                                       min_samples_split=8,
                                       random_state=1024)

        model.fit(self.X_train, self.y_train)
        y_predict = model.predict(self.X_test)

        print('Accuracy Score is {:.5}'.format(accuracy_score(self.y_test, y_predict)))
        print(pd.DataFrame(confusion_matrix(self.y_test, y_predict)))

        plot_confusion_matrix(confusion_matrix(self.y_test, y_predict),
                              classes=class_names, normalize=True,
                              title='Normalized Confusion Matrix: CART')

    def random_forest(self):
        class_names = ['0', '1']
        model = RandomForestClassifier(n_estimators=250,
                                       max_depth=12,
                                       min_samples_leaf=16
                                       )
        model.fit(self.X_train, self.y_train)
        y_predict = model.predict(self.X_test)

        print('Accuracy Score is {:.5}'.format(accuracy_score(self.y_test, y_predict)))
        print(pd.DataFrame(confusion_matrix(self.y_test, y_predict)))

        plot_confusion_matrix(confusion_matrix(self.y_test, y_predict),
                              classes=class_names, normalize=True,
                              title='Normalized Confusion Matrix: Random Forests')

        # https://www.jianshu.com/p/71fde5d90136

    def simple_ann(self):
        class_names = ['0', '1']
        clf = MLPClassifier(solver='sgd', activation='identity', max_iter=10, alpha=1e-5, hidden_layer_sizes=(100, 50),
                            random_state=1, verbose=True)

        clf.fit(self.X_train, self.y_train)

        y_predict = clf.predict(self.X_test)
        print(clf.predict(self.X_test))
        print('Accuracy Score is {:.5}'.format(accuracy_score(self.y_test, y_predict)))
        plot_confusion_matrix(confusion_matrix(self.y_test, y_predict),
                              classes=class_names, normalize=True,
                              title='A N N')
